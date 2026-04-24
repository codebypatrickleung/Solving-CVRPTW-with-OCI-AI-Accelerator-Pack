# CVRPTW with OCI AI Accelerator Pack

This directory contains a Jupyter notebook and Python utilities for solving the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)** using the **Vehicle Route Optimizer** from **OCI AI Accelerator Pack**, powered by **NVIDIA cuOpt** on Oracle Cloud Infrastructure (OCI). Read the full blog post [here](./Faster-Vehicle-Routing-Solving-CVRPTW-with-OCI-AI-Accelerator-Pack.pdf)

## Contents

| File / Directory | Description |
|---|---|
| [Introduction](./1-intro.ipynb) | Step-by-step notebook: data loading, payload building, API calls, and result visualisation |
| [Gehring & Homberger 200-customer instances](./2-gh200-test.ipynb) | Benchmark notebook: tests all 60 Gehring & Homberger 200-customer instances |

## Prerequisites

- Python 3.14
- A deployed **Vehicle Route Optimizer** stack (OCI AI Accelerator Pack)
- Network access to the deployed API endpoint 

Install the required Python packages before running the notebook:

```bash
pip install requests numpy pandas scipy matplotlib
```

## Quick Start

1. **Deploy** the Vehicle Route Optimizer via **OCI Console → Analytics & AI → AI Accelerator Pack → Vehicle Route Optimizer**.
2. Note the **API endpoint** from the stack Outputs after deployment (~30–45 min).
3. Open notebook and set `BASE_URL` to your API endpoint.
4. Run all cells.

## Acknowledgement
I would like to thank Gehring and Homberger for the benchmark instances and [SINTEF for maintaining the VRPTW benchmark repository](https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark), which served as the source for the instance definitions and best-known solutions used in this notebook.

- Gehring, H. and Homberger, J. (1999). "A Parallel Hybrid Evolutionary Metaheuristic for the Vehicle Routing Problem with Time Windows." Proceedings of the EURO-Gen99, pp. 80-89
- Gehring, H. and Homberger, J. (2001). "A Parallel Two-phase Metaheuristic for Routing Problems with Time Windows." Asia-Pacific Journal of Operational Research, 18, 35-47

## Cleanup

To avoid ongoing GPU/OKE costs, run **Destroy** on the Resource Manager stack in the OCI Console after you finish testing.
