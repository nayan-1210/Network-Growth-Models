import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import threading
import time
import queue
from typing import Dict, List, Tuple

class FenwickTree:
    """Binary Indexed Tree for efficient weighted sampling"""
    def __init__(self, n: int):
        self.n = n
        self.tree = [0.0] * (n + 1)
    
    def update(self, idx: int, delta: float):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & (-idx)
    
    def query(self, idx: int) -> float:
        result = 0.0
        while idx > 0:
            result += self.tree[idx]
            idx -= idx & (-idx)
        return result
    
    def total_sum(self) -> float:
        return self.query(self.n)
    
    def weighted_sample(self) -> int:
        total = self.total_sum()
        if total <= 0:
            return np.random.randint(1, self.n + 1)
        
        target = np.random.random() * total
        left, right = 1, self.n
        
        while left < right:
            mid = (left + right) // 2
            if self.query(mid) < target:
                left = mid + 1
            else:
                right = mid
        return left

class NetworkSimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Growing Network Simulation - Real-time Visualization")
        self.root.geometry("1400x900")
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.simulation_thread = None
        self.data_queue = queue.Queue()
        
        # Network data
        self.G = None
        self.target_P = {}
        self.A_k = {}
        self.m = 1
        self.current_step = 0
        self.max_steps = 1000
        self.node_degrees = []
        self.pos = None  # Network layout positions
        
        # History for plotting
        self.degree_history = []
        self.attachment_history = []
        self.step_history = []
        
        self.setup_ui()
        
        # Start the update loop
        self.root.after(50, self.update_plots)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Simulation Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Parameters frame
        params_frame = ttk.Frame(control_frame)
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Distribution selection
        ttk.Label(params_frame, text="Distribution:").grid(row=0, column=0, padx=5, sticky=tk.W)
        self.dist_var = tk.StringVar(value="power_law")
        dist_combo = ttk.Combobox(params_frame, textvariable=self.dist_var, 
                                 values=["power_law", "exponential", "custom"], width=12)
        dist_combo.grid(row=0, column=1, padx=5)
        
        # Parameters
        ttk.Label(params_frame, text="Parameter:").grid(row=0, column=2, padx=5, sticky=tk.W)
        self.param_var = tk.StringVar(value="2.5")
        param_entry = ttk.Entry(params_frame, textvariable=self.param_var, width=8)
        param_entry.grid(row=0, column=3, padx=5)
        
        ttk.Label(params_frame, text="m (edges/node):").grid(row=0, column=4, padx=5, sticky=tk.W)
        self.m_var = tk.StringVar(value="2")
        m_entry = ttk.Entry(params_frame, textvariable=self.m_var, width=6)
        m_entry.grid(row=0, column=5, padx=5)
        
        ttk.Label(params_frame, text="Max nodes:").grid(row=0, column=6, padx=5, sticky=tk.W)
        self.max_nodes_var = tk.StringVar(value="1000")
        max_nodes_entry = ttk.Entry(params_frame, textvariable=self.max_nodes_var, width=8)
        max_nodes_entry.grid(row=0, column=7, padx=5)
        
        # Speed control
        ttk.Label(params_frame, text="Speed (ms):").grid(row=1, column=0, padx=5, sticky=tk.W)
        self.speed_var = tk.StringVar(value="100")
        speed_entry = ttk.Entry(params_frame, textvariable=self.speed_var, width=8)
        speed_entry.grid(row=1, column=1, padx=5)
        
        # Buttons frame
        buttons_frame = ttk.Frame(control_frame)
        buttons_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(buttons_frame, text="Start Simulation", 
                                   command=self.start_simulation, style="Accent.TButton")
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(buttons_frame, text="Pause", 
                                   command=self.pause_simulation, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(buttons_frame, text="Stop", 
                                  command=self.stop_simulation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(buttons_frame, text="Reset", command=self.reset_simulation)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready to start simulation")   
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Visualization frame
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with subplots
        self.fig = plt.figure(figsize=(14, 8))
        
        # Network visualization (top left)
        self.ax_network = plt.subplot2grid((2, 3), (0, 0), colspan=1)
        self.ax_network.set_title("Network Growth")
        self.ax_network.set_aspect('equal')
        
        # Attachment kernel (top middle)
        self.ax_kernel = plt.subplot2grid((2, 3), (0, 1), colspan=1)
        self.ax_kernel.set_title("Attachment Kernel A_k")
        self.ax_kernel.set_xlabel("Degree k")
        self.ax_kernel.set_ylabel("Attractiveness A_k")
        
        # Current degree distribution (top right)
        self.ax_current_dist = plt.subplot2grid((2, 3), (0, 2), colspan=1)
        self.ax_current_dist.set_title("Current Degree Distribution")
        self.ax_current_dist.set_xlabel("Degree k")
        self.ax_current_dist.set_ylabel("Probability P(k)")
        
        # Degree distribution convergence (bottom, spanning all columns)
        self.ax_convergence = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        self.ax_convergence.set_title("Degree Distribution Convergence")
        self.ax_convergence.set_xlabel("Simulation Steps")
        self.ax_convergence.set_ylabel("Total Variation Distance from Target")
        self.ax_convergence.set_yscale('log')
        
        plt.tight_layout()
        
        # Embed matplotlib in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Real-time Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Statistics labels
        self.stats_labels = {}
        stats_items = [
            ("Current Step:", "0"),
            ("Network Size:", "0 nodes, 0 edges"),
            ("Mean Degree:", "0.00"),
            ("Target Mean:", "0.00"),
            ("TVD from Target:", "N/A"),
            ("Last Added Node Degree:", "N/A")
        ]
        
        for i, (label, value) in enumerate(stats_items):
            row, col = i // 3, (i % 3) * 2
            ttk.Label(stats_grid, text=label, font=("TkDefaultFont", 9, "bold")).grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=2)
            self.stats_labels[label] = tk.StringVar(value=value)
            ttk.Label(stats_grid, textvariable=self.stats_labels[label], 
                     font=("TkDefaultFont", 9)).grid(
                row=row, column=col+1, sticky=tk.W, padx=5, pady=2)
    
    def create_target_distribution(self):
        """Create target distribution based on user selection"""
        dist_type = self.dist_var.get()
        param = float(self.param_var.get())
        self.m = int(self.m_var.get())
        
        if dist_type == "power_law":
            # Power law: P(k) ∝ k^(-γ)
            gamma = param
            k_min, k_max = self.m, 50
            degrees = range(k_min, k_max + 1)
            probs = [k**(-gamma) for k in degrees]
            
        elif dist_type == "exponential":
            # Exponential: P(k) ∝ exp(-λk)
            lam = param
            k_min, k_max = self.m, 30
            degrees = range(k_min, k_max + 1)
            probs = [np.exp(-lam * k) for k in degrees]
        
        else:  # custom
            # Simple custom distribution for demo
            degrees = range(self.m, 20)
            probs = [1.0 / (k + 1) for k in degrees]
        
        # Normalize and adjust for consistency condition <k> = 2m
        total_prob = sum(probs)
        current_mean = sum(k * p for k, p in zip(degrees, probs)) / total_prob
        
        # Scale to achieve target mean
        scale_factor = (2 * self.m) / current_mean
        scaled_probs = [p * scale_factor for p in probs]
        
        # Re-normalize
        total_scaled = sum(scaled_probs)
        final_probs = [p / total_scaled for p in scaled_probs]
        
        self.target_P = dict(zip(degrees, final_probs))
        
        # Update target mean statistic
        target_mean = sum(k * p for k, p in self.target_P.items())
        self.stats_labels["Target Mean:"].set(f"{target_mean:.3f}")
    
    def compute_attachment_kernel(self):
        """Compute attachment kernel A_k"""
        self.A_k = {}
        
        # Compute tail probabilities
        tail_prob = {}
        cumsum = 0.0
        for k in sorted(self.target_P.keys(), reverse=True):
            tail_prob[k] = cumsum
            cumsum += self.target_P[k]
        
        # Compute attachment kernel
        for k in self.target_P:
            if self.target_P[k] > 0:
                self.A_k[k] = tail_prob[k] / self.target_P[k]
            else:
                self.A_k[k] = 0.0
    
    def create_initial_network(self):
        """Create initial network"""
        self.G = nx.Graph()
        n0 = 5  # Initial core size
        
        if self.m == 1:
            # Star graph
            self.G.add_node(0)
            self.node_degrees = [n0 - 1]
            for i in range(1, n0):
                self.G.add_edge(0, i)
                self.node_degrees.append(1)
        else:
            # Small complete graph
            for i in range(n0):
                for j in range(i + 1, n0):
                    self.G.add_edge(i, j)
            self.node_degrees = [n0 - 1] * n0
        
        # Create network layout positions (spring layout, fixed)
        self.pos = nx.spring_layout(self.G, k=2, iterations=50)
        
        self.current_step = len(self.node_degrees)
        
        # Initialize data structures
        self.degree_history = []
        self.attachment_history = []
        self.step_history = []
        
        # Initialize Fenwick tree
        self.fenwick = FenwickTree(self.max_steps)
        for i, degree in enumerate(self.node_degrees):
            attractiveness = self.A_k.get(degree, 0.0)
            self.fenwick.update(i + 1, attractiveness)
    
    def simulation_worker(self):
        """Worker thread for running the simulation"""
        try:
            while self.is_running and self.current_step < self.max_steps:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Simulation step
                new_node = self.current_step
                self.G.add_node(new_node)
                
                # Sample m distinct targets
                targets = set()
                attempts = 0
                max_attempts = 10 * self.m
                
                while len(targets) < self.m and attempts < max_attempts:
                    try:
                        target_idx = self.fenwick.weighted_sample() - 1
                        if target_idx not in targets and target_idx < new_node:
                            targets.add(target_idx)
                    except:
                        target_idx = np.random.randint(0, new_node)
                        targets.add(target_idx)
                    attempts += 1
                
                # Fill remaining targets randomly if needed
                while len(targets) < self.m:
                    target_idx = np.random.randint(0, new_node)
                    targets.add(target_idx)
                
                # Create edges and update degrees
                for target in targets:
                    self.G.add_edge(new_node, target)
                    
                    # Update target node's degree
                    old_degree = self.node_degrees[target]
                    new_degree = old_degree + 1
                    self.node_degrees[target] = new_degree
                    
                    # Update Fenwick tree
                    old_attractiveness = self.A_k.get(old_degree, 0.0)
                    new_attractiveness = self.A_k.get(new_degree, 0.0)
                    self.fenwick.update(target + 1, new_attractiveness - old_attractiveness)
                
                # Add new node
                self.node_degrees.append(self.m)
                new_attractiveness = self.A_k.get(self.m, 0.0)
                self.fenwick.update(new_node + 1, new_attractiveness)
                
                # Update layout for new node (simple positioning)
                if new_node not in self.pos:
                    # Place new node at a random position near existing nodes
                    if len(self.pos) > 0:
                        center_x = np.mean([pos[0] for pos in self.pos.values()])
                        center_y = np.mean([pos[1] for pos in self.pos.values()])
                        angle = np.random.random() * 2 * np.pi
                        radius = 0.5 + 0.3 * np.random.random()
                        self.pos[new_node] = (center_x + radius * np.cos(angle), 
                                            center_y + radius * np.sin(angle))
                    else:
                        self.pos[new_node] = (np.random.random() - 0.5, np.random.random() - 0.5)
                
                self.current_step += 1
                
                # Put data in queue for GUI update
                simulation_data = {
                    'step': self.current_step,
                    'network_copy': self.G.copy(),
                    'degrees_copy': self.node_degrees.copy(),
                    'pos_copy': self.pos.copy()
                }
                self.data_queue.put(simulation_data)
                
                # Sleep based on speed setting
                try:
                    delay = int(self.speed_var.get()) / 1000.0
                    time.sleep(delay)
                except:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self.is_running = False
    
    def start_simulation(self):
        """Start the simulation"""
        if self.is_running:
            return
        
        try:
            self.max_steps = int(self.max_nodes_var.get())
            self.create_target_distribution()
            self.compute_attachment_kernel()
            self.create_initial_network()
            
            self.is_running = True
            self.is_paused = False
            
            # Update button states
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.simulation_worker, daemon=True)
            self.simulation_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {str(e)}")
    
    def pause_simulation(self):
        """Pause/resume simulation"""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        self.pause_btn.config(text="Resume" if self.is_paused else "Pause")
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.is_paused = False
        
        # Update button states
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="Pause")
        self.stop_btn.config(state=tk.DISABLED)
        
        self.progress_var.set("Simulation stopped")
    
    def reset_simulation(self):
        """Reset the simulation"""
        self.stop_simulation()
        
        # Clear data
        self.G = None
        self.current_step = 0
        self.node_degrees = []
        self.degree_history = []
        self.attachment_history = []
        self.step_history = []
        
        # Clear plots
        for ax in [self.ax_network, self.ax_kernel, self.ax_current_dist, self.ax_convergence]:
            ax.clear()
        
        self.ax_network.set_title("Network Growth")
        self.ax_kernel.set_title("Attachment Kernel A_k")
        self.ax_kernel.set_xlabel("Degree k")
        self.ax_kernel.set_ylabel("Attractiveness A_k")
        self.ax_current_dist.set_title("Current Degree Distribution")
        self.ax_current_dist.set_xlabel("Degree k")
        self.ax_current_dist.set_ylabel("Probability P(k)")
        self.ax_convergence.set_title("Degree Distribution Convergence")
        self.ax_convergence.set_xlabel("Simulation Steps")
        self.ax_convergence.set_ylabel("Total Variation Distance from Target")
        
        self.canvas.draw()
        
        # Reset progress
        self.progress_bar.config(value=0)
        self.progress_var.set("Ready to start simulation")
        
        # Reset statistics
        for key in self.stats_labels:
            if key == "Target Mean:":
                self.stats_labels[key].set("0.00")
            elif "Current Step:" in key:
                self.stats_labels[key].set("0")
            elif "Network Size:" in key:
                self.stats_labels[key].set("0 nodes, 0 edges")
            else:
                self.stats_labels[key].set("N/A")
    
    def calculate_tvd(self, empirical_dist):
        """Calculate Total Variation Distance from target"""
        all_degrees = set(self.target_P.keys()) | set(empirical_dist.keys())
        tvd = 0.5 * sum(abs(empirical_dist.get(k, 0) - self.target_P.get(k, 0)) 
                       for k in all_degrees)
        return tvd
    
    def update_plots(self):
        """Update plots with new data from simulation thread"""
        try:
            # Get latest data from queue (non-blocking)
            while not self.data_queue.empty():
                try:
                    data = self.data_queue.get_nowait()
                    
                    step = data['step']
                    G_copy = data['network_copy']
                    degrees_copy = data['degrees_copy']
                    pos_copy = data['pos_copy']
                    
                    # Update network plot
                    self.ax_network.clear()
                    self.ax_network.set_title(f"Network Growth (Step {step})")
                    
                    if len(G_copy.nodes()) > 0:
                        # Color nodes by degree
                        degrees = dict(G_copy.degree())
                        node_colors = [degrees.get(node, 0) for node in G_copy.nodes()]
                        
                        nx.draw(G_copy, pos_copy, ax=self.ax_network, 
                               node_color=node_colors, node_size=30, 
                               edge_color='lightgray', width=0.5, 
                               cmap=plt.cm.viridis, with_labels=False)
                    
                    # Update attachment kernel plot
                    self.ax_kernel.clear()
                    self.ax_kernel.set_title("Attachment Kernel A_k")
                    self.ax_kernel.set_xlabel("Degree k")
                    self.ax_kernel.set_ylabel("Attractiveness A_k")
                    
                    if self.A_k:
                        degrees = sorted(self.A_k.keys())
                        attractiveness = [self.A_k[k] for k in degrees]
                        self.ax_kernel.plot(degrees, attractiveness, 'bo-', markersize=4)
                        self.ax_kernel.grid(True, alpha=0.3)
                    
                    # Update current degree distribution
                    self.ax_current_dist.clear()
                    self.ax_current_dist.set_title(f"Current vs Target Distribution (Step {step})")
                    self.ax_current_dist.set_xlabel("Degree k")
                    self.ax_current_dist.set_ylabel("Probability P(k)")
                    
                    if degrees_copy:
                        # Calculate empirical distribution
                        degree_counts = Counter(degrees_copy)
                        total_nodes = len(degrees_copy)
                        empirical_dist = {k: count/total_nodes for k, count in degree_counts.items()}
                        
                        # Plot both distributions
                        all_degrees = sorted(set(self.target_P.keys()) | set(empirical_dist.keys()))
                        target_probs = [self.target_P.get(k, 0) for k in all_degrees]
                        empirical_probs = [empirical_dist.get(k, 0) for k in all_degrees]
                        
                        self.ax_current_dist.plot(all_degrees, target_probs, 'ro-', 
                                                label='Target P(k)', markersize=4, alpha=0.7)
                        self.ax_current_dist.plot(all_degrees, empirical_probs, 'bs-', 
                                                label='Current P(k)', markersize=4, alpha=0.7)
                        self.ax_current_dist.legend()
                        self.ax_current_dist.grid(True, alpha=0.3)
                        
                        # Calculate TVD for convergence plot
                        tvd = self.calculate_tvd(empirical_dist)
                        self.step_history.append(step)
                        self.degree_history.append(tvd)
                        
                        # Update statistics
                        mean_degree = np.mean(degrees_copy) if degrees_copy else 0
                        self.stats_labels["Current Step:"].set(str(step))
                        self.stats_labels["Network Size:"].set(f"{G_copy.number_of_nodes()} nodes, {G_copy.number_of_edges()} edges")
                        self.stats_labels["Mean Degree:"].set(f"{mean_degree:.3f}")
                        self.stats_labels["TVD from Target:"].set(f"{tvd:.4f}")
                        if degrees_copy:
                            self.stats_labels["Last Added Node Degree:"].set(str(degrees_copy[-1]))
                    
                    # Update convergence plot
                    if len(self.step_history) > 1:
                        self.ax_convergence.clear()
                        self.ax_convergence.set_title("Degree Distribution Convergence")
                        self.ax_convergence.set_xlabel("Simulation Steps")
                        self.ax_convergence.set_ylabel("Total Variation Distance from Target")
                        self.ax_convergence.set_yscale('log')
                        
                        self.ax_convergence.plot(self.step_history, self.degree_history, 'g-', linewidth=2)
                        self.ax_convergence.grid(True, alpha=0.3)
                    
                    # Update progress bar
                    progress = (step / self.max_steps) * 100
                    self.progress_bar.config(value=progress)
                    self.progress_var.set(f"Step {step}/{self.max_steps} ({progress:.1f}%)")
                    
                except queue.Empty:
                    break
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            print(f"Plot update error: {e}")
        
        # Schedule next update
        if not self.root.winfo_exists():
            return
        self.root.after(50, self.update_plots)

def main():
    root = tk.Tk()
    app = NetworkSimulationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()