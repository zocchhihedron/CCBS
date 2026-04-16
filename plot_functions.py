import numpy as np
import matplotlib.pyplot as plt

def plot_hypercolumn_activations(nn):
    o_array = np.array(nn.o_history)
    time_array = np.array(nn.time_axis)
    
    plt.figure(figsize=(12, 8))
    
    # Track labels for the y-axis
    y_ticks = []
    y_labels = []

    for h in range(nn.hypercolumns):
        for m in range(nn.minicolumns):
            # Calculate global index in the flat array
            unit_idx = h * nn.minicolumns + m
            
            # Calculate shifted y-position with a gap between hypercolumns
            y_pos = (h * (nn.minicolumns + 1)) + m
            
            # Find when this unit was active
            active_indices = np.where(o_array[:, unit_idx] == 1)[0]
            
            if len(active_indices) > 0:
                plt.scatter(time_array[active_indices], 
                            np.ones_like(active_indices) * y_pos, 
                            marker='s', s=40, color='black')
            
            # Record tick position and label
            y_ticks.append(y_pos)
            y_labels.append(f"H{h}:M{m}")

    # Set custom ticks to show Hypercolumn and Minicolumn IDs
    plt.yticks(y_ticks, y_labels, fontsize=8)
    plt.ylabel("Hypercolumn (H) : Minicolumn (M)")
    plt.xlabel("Time (s)")
    plt.title("BCPNN Activation: Grouped by Hypercolumns")
    
    # Optional: Add horizontal lines to separate hypercolumns visually
    for h in range(1, nn.hypercolumns):
        line_pos = h * (nn.minicolumns + 1) - (1 / 2)
        plt.axhline(y=line_pos, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_current_history(nn):
    s_array = np.array(nn.s_history)
    time_array = np.array(nn.time_axis)
    
    plt.figure(figsize=(12, 8))
    
    # Track labels for the y-axis
    y_ticks = []
    y_labels = []

    for h in range(nn.hypercolumns):
        for m in range(nn.minicolumns):
            # Calculate global index in the flat array
            unit_idx = h * nn.minicolumns + m
            
            # Calculate shifted y-position with a gap between hypercolumns
            y_pos = (h * (nn.minicolumns + 1)) + m
            
            # Find when this unit was active
            active_indices = np.where(s_array[:, unit_idx] == 1)[0]
            
            if len(active_indices) > 0:
                plt.scatter(time_array[active_indices], 
                            np.ones_like(active_indices) * y_pos, 
                            marker='s', s=40, color='black')
            
            # Record tick position and label
            y_ticks.append(y_pos)
            y_labels.append(f"H{h}:M{m}")

    # Set custom ticks to show Hypercolumn and Minicolumn IDs
    plt.yticks(y_ticks, y_labels, fontsize=8)
    plt.ylabel("Hypercolumn (H) : Minicolumn (M)")
    plt.xlabel("Time (s)")
    plt.title("BCPNN Activation: Grouped by Hypercolumns")
    
    # Optional: Add horizontal lines to separate hypercolumns visually
    for h in range(1, nn.hypercolumns):
        line_pos = h * (nn.minicolumns + 1) - (1 / 2)
        plt.axhline(y=line_pos, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_pattern_evolution(nn, trained_seq, recall_start_time):
    """
    Visualizes how well the network matches each trained pattern over time.
    """
    o_history = np.array(nn.o_history) # (time_steps, 100)
    time_axis = np.array(nn.time_axis)
    n_patterns = len(trained_seq)
    
    # Calculate overlap (dot product) between state and each pattern
    # Since patterns are one-hot, dot product counts matching active units
    overlaps = np.zeros((len(time_axis), n_patterns))
    for i, pattern in enumerate(trained_seq):
        # We normalize by number of hypercolumns so 1.0 = perfect match
        overlaps[:, i] = np.dot(o_history, pattern) / nn.hypercolumns

    plt.figure(figsize=(14, 6))
    
    # Create a heatmap of pattern activations
    plt.imshow(overlaps.T, aspect='auto', origin='lower', 
               extent=[time_axis[0], time_axis[-1], 0, n_patterns-1],
               cmap='viridis')
    
    plt.colorbar(label="Pattern Match Strength")
    
    # Mark the start of recall
    plt.axvline(x=recall_start_time, color='red', linestyle='--', linewidth=2, label='Recall Starts')
    
    plt.yticks(range(n_patterns), [f"Patt {i}" for i in range(n_patterns)])
    plt.xlabel("Time (s)")
    plt.ylabel("Learned Patterns")
    plt.title("Sequence Evolution (Training to Recall)")
    plt.legend()
    plt.tight_layout()
    plt.show()