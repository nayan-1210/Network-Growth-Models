# Network Growth Model

This project provides a **real-time simulation and visualization of growing networks** using a customizable attachment kernel and a known **target degree distribution**. The GUI is built with Tkinter and uses Matplotlib for interactive network and statistics plots.

## Features

- **Interactive GUI** for controlling simulation parameters:
  - Choose degree distribution: Power Law, Exponential, or Custom
  - Set distribution parameters, number of edges per node (`m`), and maximum nodes
  - Control simulation speed and progress
- **Real-time visualization**:
  - Network growth animation
  - Attachment kernel plot
  - Current and target degree distributions
  - Convergence plot (Total Variation Distance)
- **Statistics panel**: Displays current step, network size, mean degree, TVD, and more
- **Efficient sampling**: Uses a Fenwick Tree for weighted node selection

## Requirements

- Python 3.7+
- `matplotlib`
- `networkx`
- `numpy`
- No external GUI dependencies (uses Tkinter, included with Python)

Install dependencies with:

```bash
pip install matplotlib networkx numpy
```

## Usage

1. Run the script:

    ```bash
    python Network_Growth_Model.py
    ```

2. Use the GUI to:
    - Select the desired degree distribution and parameters
    - Set simulation speed and network size
    - Start, pause, stop, or reset the simulation
    - Observe real-time network growth and statistics

## How It Works

- The simulation starts with a small core network.
- New nodes are added one by one, each connecting to `m` existing nodes.
- Node selection for attachment is based on a computed kernel to match the target degree distribution.
- The GUI updates plots and statistics in real time.

## Files

- `Network_Growth_Model.py`: Main source code for the GUI and simulation.

## Customization

- You can modify the `create_target_distribution` method to implement other degree distributions.
- The attachment kernel logic is in `compute_attachment_kernel`.

## Sample Screenshots

- The sample screenshots can be found in the images directory 
