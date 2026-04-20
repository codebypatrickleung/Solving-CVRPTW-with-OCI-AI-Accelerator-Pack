"""
CVRPTW utility functions for the intro.ipynb and gh200.ipynb notebooks.

Functions
---------
solve               - Submit a payload and poll for a solution in one call.
solution_eval       - Compare cuOpt solution against the best-known result.
plot_routes         - Visualise vehicle routes on a two-panel map.
create_from_file    - Parse a Gehring & Homberger benchmark file.
build_cost_matrix   - Compute a pairwise Euclidean distance matrix.
build_fleet_data    - Build the fleet_data section of a cuOpt payload.
build_task_data     - Build the task_data section of a cuOpt payload.
build_payload       - Build a complete cuOpt payload (no solver_config).
summarise_results   - Display a summary table for multiple time-limit runs.
plot_instance       - Visualise a benchmark dataset (locations + time-window distribution).
"""

import logging
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.spatial import distance


# ---------------------------------------------------------------------------
# Solve helper
# ---------------------------------------------------------------------------

def solve(base_url: str, payload: dict, time_limit: int) -> dict:
    """
    Submit a cuOpt payload and poll for the solution in one step.

    Args:
        base_url (str): Base URL of the cuOpt server.
        payload (dict): Complete cuOpt request payload. A ``solver_config`` key
            with the given ``time_limit`` is added automatically.
        time_limit (int): Solver time budget in seconds.

    Returns:
        dict: The ``solver_response`` sub-dict from the API response, or an
            empty dict if the request failed.
    """
    payload['solver_config'] = {'time_limit': time_limit}
    logging.info(f'Submitting optimization problem (time_limit={time_limit}s)...')

    # Submit the problem payload
    req_id = None
    try:
        response = requests.post(f"{base_url}/cuopt/request", json=payload)
        response.raise_for_status()
        data = response.json()
        req_id = data.get("reqId")
        logging.info(f"Request submitted successfully. Request ID: {req_id}")
    except requests.RequestException as e:
        logging.error(f"Error submitting problem: {e}")
        return {}

    if not req_id:
        return {}

    # Poll the solution endpoint until the solver finishes
    solution_url = f"{base_url}/cuopt/solution/{req_id}"
    max_retries = 300
    logging.info('Polling for solution...')
    for _ in range(max_retries):
        try:
            response = requests.get(solution_url)
            if response.status_code == 200:
                result = response.json()
                if "response" in result:
                    logging.info("Optimization Complete")
                    solver_response = result["response"]
                    return solver_response.get('solver_response', {})
            elif response.status_code == 404:
                # 404 indicates the GPU is still processing the request
                logging.info("Solving...")
            else:
                logging.error(f"Server Error: {response.status_code} - {response.text}")
                return {}
        except requests.RequestException as e:
            logging.error(f"Network Error: {e}")
            return {}
        time.sleep(1)

    logging.error(f"Timed out after {max_retries} seconds.")
    return {}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def solution_eval(vehicles, cost, best_known_solution):
    """
    Print a comparison between cuOpt's solution and the best-known solution.

    Args:
        vehicles (int): Number of vehicles used in the cuOpt solution.
        cost (float): Total travel cost of the cuOpt solution.
        best_known_solution (dict): Dict with 'n_vehicles' and 'cost' from
            the literature.
    """
    print(f"- cuOpt provides a solution using {vehicles} vehicles")
    print(f"- This represents {vehicles - best_known_solution['n_vehicles']} "
          f"more than the best known solution")
    print(f"- Vehicle Percent Difference "
          f"{(vehicles / best_known_solution['n_vehicles'] - 1) * 100:.2f}% \n")
    print(f"- In addition cuOpt provides a solution cost of {cost}")
    print(f"- Best known solution cost is {best_known_solution['cost']}")
    print(f"- Cost Percent Difference "
          f"{(cost / best_known_solution['cost'] - 1) * 100:.2f}%")


def summarise_results(results_by_limit, best_known_solution,
                      benchmark_name="200-Customer Benchmark (R1_2_1)"):
    """
    Display a summary table comparing cuOpt results across time limits.

    Args:
        results_by_limit (dict): Mapping of ``time_limit (int) -> dict``
            where each dict has keys:
            - ``'num_vehicles'``  – vehicles used in the solution
            - ``'solution_cost'`` – total travel cost
        best_known_solution (dict): Dict with 'n_vehicles' and 'cost'.
        benchmark_name (str): Label shown in the printed table header.

    Returns:
        pd.DataFrame: The summary table (also printed to stdout).
    """
    bk_veh  = best_known_solution['n_vehicles']
    bk_cost = best_known_solution['cost']

    rows = []
    for time_limit in sorted(results_by_limit):
        r        = results_by_limit[time_limit]
        n_veh    = r.get('num_vehicles', 'N/A')
        cost     = r.get('solution_cost', 'N/A')

        veh_diff = (n_veh - bk_veh) if isinstance(n_veh, (int, float)) else 'N/A'
        veh_pct  = (
            f"{(n_veh / bk_veh - 1) * 100:.2f}%"
            if isinstance(n_veh, (int, float)) else 'N/A'
        )
        cost_pct = (
            f"{(cost / bk_cost - 1) * 100:.2f}%"
            if isinstance(cost, (int, float)) else 'N/A'
        )
        cost_str = f"{cost:.2f}" if isinstance(cost, (int, float)) else cost

        rows.append({
            'Time Limit (s)'      : time_limit,
            'Vehicles Used'       : n_veh,
            'Best Known Vehicles' : bk_veh,
            'Extra Vehicles'      : veh_diff,
            'Vehicle Gap (%)'     : veh_pct,
            'Total Cost'          : cost_str,
            'Best Known Cost'     : f"{bk_cost:.2f}",
            'Cost Gap (%)'        : cost_pct,
        })

    columns = [
        'Time Limit (s)', 'Vehicles Used', 'Best Known Vehicles',
        'Extra Vehicles', 'Vehicle Gap (%)', 'Total Cost',
        'Best Known Cost', 'Cost Gap (%)',
    ]
    df = pd.DataFrame(rows, columns=columns).set_index('Time Limit (s)')

    print("=" * 80)
    print(f"cuOpt Performance Summary — {benchmark_name}")
    print("=" * 80)

    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())

    return df


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_routes(solver_response, orders, title='cuOpt Solution Routes',
                max_routes=20, detail_routes=5):
    """
    Visualise vehicle routes from a cuOpt solver response.

    Left panel  – all routes overlaid on the full customer map (up to
    ``max_routes``).
    Right panel – directional arrow-diagram for the first ``detail_routes``
    vehicles.

    Args:
        solver_response (dict | None): The ``solver_response`` dict returned
            by ``solve()`` (the nested sub-dict, not the
            top-level response wrapper).
        orders (pd.DataFrame): Location data; must contain ``xcord`` and
            ``ycord`` columns (depot at index 0).
        title (str): Figure suptitle.
        max_routes (int): Maximum routes drawn on the overview map.
        detail_routes (int): Routes drawn in the detailed arrow panel.
    """
    if solver_response is None or solver_response.get('status') != 0:
        print('No valid solution available to visualise.')
        return

    vehicle_data = solver_response.get('vehicle_data', {})
    if not vehicle_data:
        print('No vehicle_data found in solver_response.')
        return

    coords   = orders[['xcord', 'ycord']].values
    depot_xy = coords[0]
    n_used   = len(vehicle_data)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── Left: overview of all routes ─────────────────────────────────
    ax = axes[0]
    ax.scatter(coords[1:, 0], coords[1:, 1],
               s=10, c='lightgray', zorder=2, label='Customer')
    ax.scatter(*depot_xy, s=200, c='red', marker='*', zorder=6, label='Depot')

    vids    = list(vehicle_data.keys())[:max_routes]
    cmap_ov = [plt.cm.tab20(i / max(len(vids), 1)) for i in range(len(vids))]

    for idx, vid in enumerate(vids):
        route = vehicle_data[vid].get('route', [])
        if len(route) < 2:
            continue
        rc    = np.array([coords[loc] for loc in route])
        color = cmap_ov[idx]
        ax.plot(rc[:, 0], rc[:, 1], '-', color=color, linewidth=0.9, alpha=0.7)
        ax.scatter(rc[1:-1, 0], rc[1:-1, 1], s=15, color=color, zorder=4)

    ax.set_title(
        f'{title}\n'
        f'(showing {len(vids)} of {n_used} routes | '
        f'cost: {solver_response.get("solution_cost", "N/A"):.2f})',
        fontsize=11
    )
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='upper right', fontsize=8)

    # ── Right: detailed arrow-diagram for first N vehicles ───────────
    ax2 = axes[1]
    ax2.scatter(coords[1:, 0], coords[1:, 1], s=10, c='lightgray', zorder=2)
    ax2.scatter(*depot_xy, s=200, c='red', marker='*', zorder=6)
    ax2.annotate('Depot', depot_xy, textcoords='offset points',
                 xytext=(6, 4), fontsize=8, fontweight='bold', color='red')

    detail_vids = list(vehicle_data.keys())[:detail_routes]
    palette     = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    leg_patches = []

    for idx, vid in enumerate(detail_vids):
        route = vehicle_data[vid].get('route', [])
        if len(route) < 2:
            continue
        rc      = np.array([coords[loc] for loc in route])
        color   = palette[idx % len(palette)]
        n_stops = len(route) - 2

        for j in range(len(rc) - 1):
            ax2.annotate('', xy=rc[j + 1], xytext=rc[j],
                         arrowprops=dict(arrowstyle='->',
                                         color=color,
                                         lw=1.5,
                                         mutation_scale=12))
        ax2.scatter(rc[1:-1, 0], rc[1:-1, 1], s=70, color=color, zorder=5)
        for k, (x, y) in enumerate(rc[1:-1], 1):
            ax2.annotate(str(k), (x, y), fontsize=7, ha='center',
                         va='center', color='white',
                         fontweight='bold', zorder=6)

        leg_patches.append(
            mpatches.Patch(
                color=color, label=f'Vehicle {vid} ({n_stops} stops)'
            )
        )

    leg_patches.append(
        plt.Line2D([0], [0], marker='*', color='w',
                   markerfacecolor='red', markersize=12, label='Depot')
    )
    ax2.legend(handles=leg_patches, loc='upper right', fontsize=8)
    ax2.set_title(
        f'Detailed View — First {len(detail_vids)} Vehicle Routes\n'
        f'(numbers = visit order along each route)',
        fontsize=11
    )
    ax2.set_xlabel('X Coordinate')

    plt.tight_layout()
    plt.show()


def plot_instance(orders, instance_name, instance_type):
    """
    Visualise a benchmark dataset with two panels.

    Left panel  – customer locations coloured by demand with the depot marked.
    Right panel – histogram of customer time-window widths with mean and median
    reference lines.

    Args:
        orders (pd.DataFrame): Location data; must contain ``xcord``, ``ycord``,
            ``demand``, ``earliest_time``, and ``latest_time`` columns.
            Row 0 is the depot; subsequent rows are customers.
        instance_name (str): Instance identifier shown in plot titles
            (e.g., ``'C1_2_1'``).
        instance_type (str): Distribution description shown in plot titles
            (e.g., ``'Clustered'`` or ``'Random'``).
    """
    n_locations = orders['demand'].shape[0] - 1

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Left: Customer locations coloured by demand ───────────────────────
    ax      = axes[0]
    depot_x = orders['xcord'].values[0]
    depot_y = orders['ycord'].values[0]
    cust_x  = orders['xcord'].values[1:]
    cust_y  = orders['ycord'].values[1:]
    demands = orders['demand'].values[1:]

    scatter = ax.scatter(cust_x, cust_y, c=demands, cmap='viridis',
                         s=20, alpha=0.7, zorder=5)
    ax.scatter(depot_x, depot_y, s=300, c='red', marker='*',
               zorder=6, label='Depot')
    plt.colorbar(scatter, ax=ax, label='Demand per customer')
    ax.set_title(
        f'{instance_name}: {instance_type} Customer Locations\n'
        f'({n_locations} customers, 1 depot — colour = demand)',
        fontsize=12
    )
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(loc='upper right')

    # ── Right: Time-window width distribution ─────────────────────────────
    ax2      = axes[1]
    tw_width = orders['latest_time'].values[1:] - orders['earliest_time'].values[1:]

    ax2.hist(tw_width, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(tw_width.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {tw_width.mean():.1f}')
    ax2.axvline(np.median(tw_width), color='orange', linestyle=':', linewidth=1.5,
                label=f'Median: {np.median(tw_width):.1f}')
    ax2.set_title(
        f'Distribution of Customer Time-Window Widths\n'
        f'({instance_name} — {instance_type} Instance)',
        fontsize=12
    )
    ax2.set_xlabel('Time-Window Width (time units)')
    ax2.set_ylabel('Number of Customers')
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def create_from_file(file_path, is_pdp=False):
    """
    Parse a Gehring & Homberger benchmark file into a pandas DataFrame.

    The file format has:
      - Line 5: vehicle count and capacity (for VRP), or line 1 for PDP
      - Lines 10+: location data rows (for VRP), or all remaining lines for PDP

    Args:
        file_path (str): Path to the benchmark problem file.
        is_pdp (bool): If True, parse Pickup & Delivery Problem format.

    Returns:
        tuple: ``(df, vehicle_capacity, vehicle_num)``
            - ``df``               – DataFrame with one row per location
            - ``vehicle_capacity`` – capacity of each vehicle (int)
            - ``vehicle_num``      – number of vehicles available (int)
    """
    node_list = []
    vehicle_num = None
    vehicle_capacity = None

    with open(file_path, "rt") as f:
        for count, line in enumerate(f, start=1):
            if is_pdp and count == 1:
                vehicle_num, vehicle_capacity, _speed = line.split()
                # _speed is present in some PDP file variants but is not used
                # in the CVRPTW problem formulation
            elif not is_pdp and count == 5:
                vehicle_num, vehicle_capacity = line.split()
            elif is_pdp and count > 1:
                node_list.append(line.split())
            elif not is_pdp and count >= 10:
                node_list.append(line.split())

    vehicle_num      = int(vehicle_num)
    vehicle_capacity = int(vehicle_capacity)

    columns = [
        "vertex", "xcord", "ycord", "demand",
        "earliest_time", "latest_time", "service_time",
    ]
    rows = []
    for item in node_list:
        row = {
            "vertex"        : int(item[0]),
            "xcord"         : float(item[1]),
            "ycord"         : float(item[2]),
            "demand"        : int(item[3]),
            "earliest_time" : int(item[4]),
            "latest_time"   : int(item[5]),
            "service_time"  : int(item[6]),
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=columns)
    return df, vehicle_capacity, vehicle_num


# ---------------------------------------------------------------------------
# Payload building helpers
# ---------------------------------------------------------------------------

def build_cost_matrix(orders):
    """
    Compute a pairwise Euclidean distance matrix from location coordinates.

    Args:
        orders (pd.DataFrame): DataFrame with ``xcord`` and ``ycord`` columns.
            Row 0 is the depot; subsequent rows are customers.

    Returns:
        np.ndarray: Square distance matrix of shape ``(len(orders), len(orders))``,
            where ``orders`` includes the depot at row 0.
    """
    coords = list(zip(orders['xcord'].tolist(), orders['ycord'].tolist()))
    return distance.cdist(coords, coords, 'euclidean')


def build_fleet_data(n_vehicles, vehicle_capacity):
    """
    Build the ``fleet_data`` section of a cuOpt payload.

    All vehicles start and end at the depot (location index 0) and share the
    same capacity (homogeneous fleet).

    Args:
        n_vehicles (int): Number of vehicles.
        vehicle_capacity (int | float): Capacity of each vehicle.

    Returns:
        dict: ``fleet_data`` dict ready for insertion into a cuOpt payload.
    """
    return {
        "vehicle_locations": [[0, 0]] * n_vehicles,
        "capacities"       : [[vehicle_capacity] * n_vehicles],
    }


def build_task_data(orders, n_locations):
    """
    Build the ``task_data`` section of a cuOpt payload.

    Includes customer locations, demands, time windows, and service times.
    The depot (index 0) is excluded from task data.

    Args:
        orders (pd.DataFrame): DataFrame with columns ``demand``,
            ``earliest_time``, ``latest_time``, and ``service_time``.
            Row 0 must be the depot.
        n_locations (int): Number of customer locations (excluding depot).

    Returns:
        dict: ``task_data`` dict ready for insertion into a cuOpt payload.
    """
    location_demand  = orders['demand'].values.astype(int).tolist()
    earliest_times   = orders['earliest_time'].values.astype(int).tolist()
    latest_times     = orders['latest_time'].values.astype(int).tolist()
    service_times    = orders['service_time'].values.astype(int).tolist()

    return {
        "task_locations"    : list(range(1, n_locations + 1)),
        "demand"            : [location_demand[1:]],
        "task_time_windows" : [
            [earliest_times[i], latest_times[i]]
            for i in range(1, n_locations + 1)
        ],
        "service_times"     : service_times[1:],
    }


def build_payload(orders, n_locations, n_vehicles, vehicle_capacity):
    """
    Build a complete cuOpt payload (without ``solver_config``).

    Combines the cost matrix, fleet data, and task data into a single dict
    that can be passed directly to ``solve()`` after
    adding a ``solver_config`` entry.

    Args:
        orders (pd.DataFrame): Location data (depot at row 0, customers
            at rows 1+).  Must have columns ``xcord``, ``ycord``, ``demand``,
            ``earliest_time``, ``latest_time``, and ``service_time``.
        n_locations (int): Number of customer locations (excluding depot).
        n_vehicles (int): Number of vehicles.
        vehicle_capacity (int | float): Capacity per vehicle.

    Returns:
        dict: cuOpt payload with ``cost_matrix_data``, ``fleet_data``, and
            ``task_data`` keys populated.
    """
    cost_matrix = build_cost_matrix(orders)

    payload = {
        "cost_matrix_data": {
            "data": {"0": cost_matrix.astype(np.float32).tolist()}
        },
        "fleet_data": build_fleet_data(n_vehicles, vehicle_capacity),
        "task_data" : build_task_data(orders, n_locations),
    }
    return payload
