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
```

## How to Run

To run the Elevator Dispatching optimization code, navigate to the `src` directory in your terminal and execute the `Elevator_Dispatcher.py` script. This script initializes the elevator environment, trains the Q-learning agent, and evaluates its performance.

### Steps to Run:
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/iburdo/AE4350.git
    ```
2. Navigate to the `src` directory:
    ```bash
    cd AE4350/src
    ```
3. Run the `Elevator_Dispatcher.py` script:
    ```bash
    python Elevator_Dispatcher.py
    ```
4. Observe the training progress and final results, which will be output in the terminal and through visualizations.

### Example:
```bash
cd AE4350/src
python Elevator_Dispatcher.py
