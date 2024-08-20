# Elevator Dispatching Optimization using Q-Learning

## Overview
This repository contains the implementation of an Elevator Dispatching system optimized using Reinforcement Learning (RL), specifically Q-Learning. The project simulates a multi-floor building and an elevator system, where the agent learns to minimize waiting times by making optimal decisions on whether to move up, down, or stay at the current floor.

## Features
- **Elevator Environment:** A simulated environment that models an elevator in a multi-floor building, complete with passenger requests and rewards based on the elevator's actions.
- **Q-Learning Agent:** An agent that learns the optimal policy for minimizing passenger waiting time using a tabular Q-Learning approach.
- **Training and Testing:** Scripts to train the agent over multiple episodes and evaluate its performance.
- **Performance Analysis:** Visualizations of training progress and Q-values, including the impact of different hyperparameters on training performance.

## Requirements
- Python 3.x
- NumPy
- Matplotlib

To install the required libraries, you can run:
```bash
pip install numpy matplotlib
