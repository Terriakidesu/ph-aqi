import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_node(ax, x, y, label, color, is_square=True):
    if is_square:
        rect = patches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6, boxstyle='round,pad=0.05', facecolor='none', edgecolor=color, linewidth=2.5, zorder=10)
        ax.add_patch(rect)
    else:
        circle = patches.Circle((x, y), radius=0.3, facecolor='none', edgecolor=color, linewidth=2.5, zorder=10)
        ax.add_patch(circle)
    
    ax.text(x, y, label, ha='center', va='center', fontsize=14, fontweight='normal', color='black', zorder=11)

def main():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor('white')
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')

    yellow, teal, red = '#f1c40f', '#1abc9c', '#e74c3c'

    # State Flows (Black lines, zorder=1)
    ax.plot([1, 13], [6, 6], 'k-', lw=2, zorder=1)
    ax.plot([1, 13], [2, 2], 'k-', lw=2, zorder=1)
    ax.text(0.5, 6, "$LTM_{t-1}$", fontsize=14, color='black', ha='right', fontweight='bold', zorder=1)
    ax.text(13.5, 6, "$LTM_{t}$", fontsize=14, color='black', fontweight='bold', zorder=1)
    ax.text(0.5, 2, "$STM_{t-1}$", fontsize=14, color='black', ha='right', fontweight='bold', zorder=1)
    ax.text(13.5, 2, "$STM_{t}$", fontsize=14, color='black', fontweight='bold', zorder=1)

    # Input node
    ax.plot([3.5, 3.5], [0.5, 2], 'k-', lw=2, zorder=1)
    ax.text(3.5, 0.2, "$E_t$", fontsize=14, color='black', ha='center', fontweight='bold', zorder=1)

    # Nodes (zorder=10 for patch, 11 for text)
    draw_node(ax, 3.5, 3.5, "$\\sigma$", yellow) # Forget gate
    draw_node(ax, 3.5, 6, "$\\times$", yellow) # Forget mul
    
    draw_node(ax, 5.5, 3.5, "$\\sigma$", teal) # Input gate
    draw_node(ax, 7, 3.5, "$\\tanh$", teal) # Cell candidate
    draw_node(ax, 7, 4.8, "$\\times$", teal) # Input mul
    
    draw_node(ax, 8.5, 6, "$+$", red) # Cell add
    
    draw_node(ax, 8.5, 4.5, "$\\tanh$", teal) # Output tanh
    draw_node(ax, 10, 2, "$\\sigma$", teal) # Output gate
    draw_node(ax, 11, 2, "$\\times$", teal) # Output mul

    # Connectivity (Arrows - Black, zorder=1)
    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=1)

    arrow((1, 6), (3.1, 6))
    arrow((3.5, 3.8), (3.5, 5.7))
    arrow((3.5, 1.7), (3.5, 3.2))
    arrow((3.9, 6), (8.1, 6))
    
    arrow((3.5, 2), (5.5, 3.2))
    arrow((3.5, 2), (7, 3.2))

    arrow((5.5, 3.8), (5.5, 4.5))
    arrow((5.9, 4.8), (6.7, 4.8))
    arrow((7, 5.1), (8.1, 6))
    
    arrow((8.5, 5.7), (8.5, 4.8))
    arrow((8.5, 4.2), (8.5, 2.3))
    arrow((10.4, 2), (10.7, 2))
    
    plt.title("LSTM Cell Structure", fontsize=20, color='black', pad=20, loc='left', fontweight='bold')
    
    output_dir = Path("figs/model_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "lstm_architecture_ref.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved restructured, white-background LSTM cell diagram to: {save_path}")

if __name__ == "__main__":
    main()
