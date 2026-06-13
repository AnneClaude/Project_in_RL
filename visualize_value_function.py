import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. Check if Q-tables exist, otherwise train them
if not os.path.exists("q_table_qlearning.npy") or not os.path.exists("q_table_sarsa.npy"):
    print("Q-tables not found. Running compare_algorithms.py to generate them...")
    from compare_algorithms import run_comparison
    run_comparison()

# 2. Load the Q-tables
q_table_ql = np.load("q_table_qlearning.npy")
q_table_sarsa = np.load("q_table_sarsa.npy")

print(f"Loaded Q-tables with shape: Q-Learning={q_table_ql.shape}, SARSA={q_table_sarsa.shape}")

def plot_heatmap(ax, q_table, light_state, title):
    """
    Plots a 2D heatmap of the state-value function V(s) = max_a Q(s, a)
    for a given light_state. Annotates each cell with the value and optimal action.
    """
    # Grid dimensions: ns_queue (0 to 5) vs ew_queue (0 to 5)
    V = np.zeros((6, 6))
    Actions = np.zeros((6, 6), dtype=int)
    
    for ns in range(6):
        for ew in range(6):
            q_vals = q_table[ns, ew, light_state]
            V[ns, ew] = np.max(q_vals)
            Actions[ns, ew] = np.argmax(q_vals)
            
    # Draw heatmap
    # origin='lower' ensures y-axis is NS Queue (0 at bottom, 5 at top)
    # x-axis is EW Queue (0 at left, 5 at right)
    im = ax.imshow(V, cmap="RdYlGn", origin='lower', aspect='equal', vmin=V.min(), vmax=V.max())
    
    # Configure grid lines and ticks
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(np.arange(6))
    ax.set_yticklabels(np.arange(6))
    
    ax.set_xlabel("EW Queue Length", fontsize=10, labelpad=5)
    ax.set_ylabel("NS Queue Length", fontsize=10, labelpad=5)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    
    # Add annotations
    min_val, max_val = V.min(), V.max()
    val_range = max_val - min_val if max_val != min_val else 1.0
    
    for ns in range(6):
        for ew in range(6):
            val = V[ns, ew]
            act = Actions[ns, ew]
            
            # Text color mapping for readability: use white text for low (red) values
            normalized_val = (val - min_val) / val_range
            text_color = "white" if normalized_val < 0.35 else "black"
            
            act_text = "S" if act == 1 else "K"
            label = f"{val:.1f}\n[{act_text}]"
            
            ax.text(ew, ns, label, ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
            
            # Highlight decision boundaries (Switch actions) with a dashed border
            if act == 1:
                rect = plt.Rectangle((ew-0.45, ns-0.45), 0.9, 0.9, fill=False, edgecolor='blue', linewidth=1.5, linestyle='--')
                ax.add_patch(rect)
                
    return im

def generate_visualizations():
    # Setup a 2x2 grid of subplots for direct comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Q-Learning plots (Top Row)
    im1 = plot_heatmap(axes[0, 0], q_table_ql, 0, "Q-Learning: NS Green (Phase 0)")
    im2 = plot_heatmap(axes[0, 1], q_table_ql, 1, "Q-Learning: EW Green (Phase 1)")
    
    # SARSA plots (Bottom Row)
    im3 = plot_heatmap(axes[1, 0], q_table_sarsa, 0, "SARSA: NS Green (Phase 0)")
    im4 = plot_heatmap(axes[1, 1], q_table_sarsa, 1, "SARSA: EW Green (Phase 1)")
    
    # Add colorbars for each row
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar1 = fig.colorbar(im1, cax=cbar_ax1)
    cbar1.set_label("State-Value V(s)", fontsize=10, labelpad=8)
    
    cbar_ax2 = fig.add_axes([0.92, 0.12, 0.02, 0.35])
    cbar2 = fig.colorbar(im3, cax=cbar_ax2)
    cbar2.set_label("State-Value V(s)", fontsize=10, labelpad=8)
    
    # Create custom legend for action indicators
    legend_elements = [
        Patch(facecolor='none', edgecolor='blue', linestyle='--', linewidth=1.5, label='Optimal Action: Switch Phase (a=1)'),
        Patch(facecolor='none', edgecolor='none', label='[K] = Keep Green  |  [S] = Switch to Yellow/Other')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.96), ncol=2, fontsize=10, frameon=True)
    
    plt.suptitle("Traffic Signal RL: Learned State-Value Functions V(s) & Decision Boundaries", fontsize=15, fontweight='bold', y=0.98)
    
    # Adjust spacing to avoid overlap
    plt.subplots_adjust(right=0.90, top=0.90, hspace=0.3, wspace=0.3)
    
    output_filename = "value_functions_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Successfully generated and saved comparison heatmap to '{output_filename}'!")

if __name__ == "__main__":
    generate_visualizations()
