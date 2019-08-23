# ddp

## Introduction

A python-based web app that shows a simple 2 stage ($t_0$ and $t_1$) deterministic dual dynamic programming (ddp) hydro-thermal scheduling problem.

<p align="center">
<img src="./decision_tree.png" alt="Decision Tree" width="600">
</p>


This is 100% based on a excel toy example used by [PSR](www.psr-inc.com). 

This toy example contains one hydro generator and three thermal generators with different generation costs. It simulates the optimal dispatch scheduling in $t_0$ taking into account the future cost (in $t_1$) of generation.

<p align="center">
<img src="./diagram.png" alt="toy example" width="400">
</p>


| Generator      | Capacity     |  Cost         | Vmax         | Prod. Factor  |
| :------------- |:-------------| :-------------|:-------------|:-----|
| Hydro          | 11           |  -            | 130          | 0.2  |
| Thermo 1       | 5            |  8            | -            | -    |
| Thermo 2 	     | 5            |  12           | -            | -    |
| Thermo 3       | 20           |  15           | -            | -    |


The formulation of this problem is:

<p align="center">
<img src="./equation.png" alt="LP" width="600">
</p>



## Contents

### ddp.py

This script has functions to create the LP and process results. It also defines the power plant objects.

- `save_results_df`: converts dictionary with results to a data frame
- `print_summary`: writes dictionary of results as a formatted text table
- `set_lp`: creates LP and returns dictionary with vectors and matrices (in CVX format).
- `compute_line_cut`: computes the resulting new cut after a new iteration to be plotted.


### dash_test.py

This script implements the web app interface using Dash


