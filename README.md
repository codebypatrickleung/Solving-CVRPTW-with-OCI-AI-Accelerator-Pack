# CVRPTW with OCI AI Accelerator Pack

This directory contains a Jupyter notebook and Python utilities for solving the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)** using the **Vehicle Route Optimizer** from the **OCI AI Accelerator Pack**, powered by **NVIDIA cuOpt** on Oracle Cloud Infrastructure (OCI).

## Contents

| File / Directory | Description |
|---|---|
| `cvrptw.ipynb` | Step-by-step notebook: data loading, payload building, API calls, and result visualisation |
| `cvrptw-utils/utils.py` | Reusable helper functions imported by the notebook |

## Prerequisites

- Python 3.10–3.12
- A deployed **Vehicle Route Optimizer** stack (OCI AI Accelerator Pack)
- Network access to the deployed API endpoint (VCN, bastion, or VPN)

Install the required Python packages before running the notebook:

```bash
pip install requests numpy pandas scipy matplotlib
```

## Quick Start

1. **Deploy** the Vehicle Route Optimizer via **OCI Console → Analytics & AI → AI Accelerator Pack → Vehicle Route Optimizer**.
2. Note the **API endpoint** from the stack Outputs after deployment (~30–45 min).
3. Open `cvrptw.ipynb` and set `BASE_URL` to your API endpoint.
4. Run all cells.

## Helper Functions (`cvrptw-utils/utils.py`)

| Function | Purpose |
|---|---|
| `solve` | Submit a payload and poll for a cuOpt solution in one call |
| `solution_eval` | Compare cuOpt results against a best-known solution |
| `plot_routes` | Visualise vehicle routes on a two-panel map |
| `create_from_file` | Parse a Gehring & Homberger benchmark instance file |
| `build_cost_matrix` | Compute a pairwise Euclidean distance matrix |
| `build_fleet_data` | Build the `fleet_data` section of a cuOpt payload |
| `build_task_data` | Build the `task_data` section of a cuOpt payload |
| `build_payload` | Assemble a complete cuOpt payload |
| `summarise_results` | Print a performance summary table across multiple time limits |
| `plot_instance` | Visualise benchmark dataset: customer locations and time-window distribution |

## Benchmarks

The notebook demonstrates CVRPTW solving on two standard **Gehring & Homberger** instances (200 customers each):

| Instance | Distribution | Best-known vehicles | Best-known cost |
|---|---|---|---|
| `C1_2_1` | Clustered | 20 | 2704.57 |
| `R1_2_1` | Random | 20 | 4772.63 |

Each instance is solved with both a 2-second and a 10-second solver time limit to illustrate the cost/quality trade-off.

## Cleanup

To avoid ongoing GPU/OKE costs, run **Destroy** on the Resource Manager stack in the OCI Console after you finish testing.
