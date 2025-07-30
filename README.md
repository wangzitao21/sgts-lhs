# SGTS-LHS Framework

This repository provides the official implementation and replication package for the paper "Enhancing Surrogate Assisted Optimization with SHAP Guided Two-Stage Sampling".

It includes the Python implementation of the SGTS-LHS framework and scripts to reproduce the results from the case studies.

## Installation

1.  Clone the repository:

```bash
git clone https://github.com/wangzitao21/sgts-lhs.git
cd sgts-lhs
```

2.  Install the required Python packages:

It is recommended to create a virtual environment first.

```bash
pip install -r requirements.txt
```

## Run the Examples

The three case studies presented in the paper correspond to three executable scripts.

### Case 1

To run the first example, execute the following command in your terminal:

```bash
python run_example1.py
```

### Case 2

To run the second example, execute:

```bash
python run_example2.py
```

### Case 3

This case study requires the MODFLOW 6 executable.

1.  Download MODFLOW 6:

Download the appropriate executable for your operating system from the official repository:

- [https://github.com/MODFLOW-ORG/executables](https://github.com/MODFLOW-ORG/executables)

2.  Place the executable:

Place the downloaded executable file into the `bin/` directory at the root of this project.

- For Windows, the file should be named `mf6.exe`.
- For Linux, it should be named `mf6`.

The final path should look like this: `./bin/mf6.exe` or `./bin/mf6`.

3.  Run the script:

```bash
python run_example3.py
```

**Note on Case 3:** This example involves a number of iterations and may take a long time to complete. For convenience, we have included a  `.ipynb` with the computed results and plots. You can view `run_plot_example3.ipynb`.

This allows you to inspect the results without running the full simulation.