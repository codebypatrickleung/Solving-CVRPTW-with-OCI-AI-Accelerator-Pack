# CVRPTW with OCI AI Accelerator Pack

This directory contains a Jupyter notebook and Python utilities for solving the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)** using the **Vehicle Route Optimizer** from **OCI AI Accelerator Pack**, powered by **NVIDIA cuOpt** on Oracle Cloud Infrastructure (OCI). Read the full blog post [here](./Faster-Vehicle-Routing-Solving-CVRPTW-with-OCI-AI-Accelerator-Pack.pdf)

## Contents

| File / Directory | Description |
|---|---|
| `cvrptw.ipynb` | Step-by-step notebook: data loading, payload building, API calls, and result visualisation |
| `cvrptw-utils/utils.py` | Reusable helper functions imported by the notebook |

## Prerequisites

- Python 3.10–3.12
- A deployed **Vehicle Route Optimizer** stack (OCI AI Accelerator Pack)
- Network access to the deployed API endpoint 

Install the required Python packages before running the notebook:

```bash
pip install requests numpy pandas scipy matplotlib
```

## Quick Start

1. **Deploy** the Vehicle Route Optimizer via **OCI Console → Analytics & AI → AI Accelerator Pack → Vehicle Route Optimizer**.
2. Note the **API endpoint** from the stack Outputs after deployment (~30–45 min).
3. Open `cvrptw.ipynb` and set `BASE_URL` to your API endpoint.
4. Run all cells.

## Benchmarks

The notebook demonstrates CVRPTW solving on two standard **Gehring & Homberger** instances (200 customers each):

| Instance | Distribution | Best-known vehicles | Best-known cost |
|---|---|---|---|
| `C1_2_1` | Clustered | 20 | 2704.57 |
| `R1_2_1` | Random | 20 | 4784.11 |

Each instance is solved with both a 2-second and a 10-second solver time limit to illustrate the cost/quality trade-off.

## Cleanup

To avoid ongoing GPU/OKE costs, run **Destroy** on the Resource Manager stack in the OCI Console after you finish testing.
