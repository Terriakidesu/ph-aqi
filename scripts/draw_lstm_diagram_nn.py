import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_names):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)

    # Node positions
    node_coords = []
    
    # Draw Nodes
    for n, (layer_size, name) in enumerate(zip(layer_sizes, layer_names)):
        layer_top = v_spacing * (layer_size - 1) / 2. + (top + bottom) / 2.
        layer_coords = []
        for m in range(layer_size):
            x = n * h_spacing + left
            y = layer_top - m * v_spacing
            
            # Use different colors for different layers
            if "Input" in name:
                color = '#e8f5e9'
                edge = '#2e7d32'
            elif "LSTM" in name:
                color = '#e3f2fd'
                edge = '#1565c0'
            else:
                color = '#fce4ec'
                edge = '#c2185b'
                
            # Increase circle size for better visibility
            circle = plt.Circle((x, y), v_spacing/3.5,
                              color=color, ec=edge, zorder=4, lw=1.5)
            ax.add_artist(circle)
            layer_coords.append((x, y))
            
            # Add recurrent loop for LSTM nodes
            if "LSTM" in name:
                loop = patches.FancyArrowPatch((x, y + v_spacing/3.5), (x + v_spacing/3.5, y),
                                              connectionstyle="arc3,rad=1.5",
                                              arrowstyle="->", color='#555555',
                                              mutation_scale=12, zorder=5)
                ax.add_patch(loop)

        node_coords.append(layer_coords)
        
        # Add Layer Labels (Increased font size)
        ax.text(n * h_spacing + left, bottom - v_spacing * 1.5, name, 
                ha='center', va='top', fontsize=12, fontweight='bold', color='#222222')

    # Draw Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for coord_a in node_coords[n]:
            for coord_b in node_coords[n+1]:
                line = plt.Line2D([coord_a[0], coord_b[0]], [coord_a[1], coord_b[1]], 
                                  c='#b0bec5', lw=1.2, zorder=1, alpha=0.8)
                ax.add_artist(line)

def create_network_diagram(output_path):
    # Standard IEEE double column width is ~7.16 inches. Using 7.5x4 for a nice wide aspect ratio.
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor='white')
    ax.axis('off')
    
    # Represent the network architectually
    layer_sizes = [5, 7, 5, 1]
    layer_names = [
        "Input Features\n(Pollutants & Lags)", 
        "LSTM Layer 1\n(128 Units)", 
        "LSTM Layer 2\n(64 Units)", 
        "Dense Output\n(AQI)"
    ]
    
    draw_neural_net(ax, 0.1, 0.9, 0.25, 0.95, layer_sizes, layer_names)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_network_diagram("D:/dev/python/ph-aqi/figs/model_comparison/lstm_architecture.png")
