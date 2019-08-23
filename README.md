# ddp

## Introduction

A python-based web app that shows a simple 2 stage ($t_0$ and $t_1$) deterministic dual dynamic programming (ddp) hydro-thermal scheduling problem.

This is 100% based on a excel toy example used by PSR (www.PSR.inc). 

This toy example contains one hydro generator and three thermal generators with different generation costs. It simulates the optimal dispatch scheduling in $t_0$ taking into account the future cost (in $t_1$) of generation.

<img src="./diagram.png" alt="toy example" width="400">

## Contents

### ddp.py

This script creates and solves the optimization problem

### dash_test.py

This script implements the web app interface using Dash


