import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_lstm_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.axis('off')

    # Define colors
    box_color = '#e3f2fd'
    edge_color = '#1565c0'
    text_color = '#0d47a1'
    arrow_color = '#333333'

    # Box dimensions
    width, height = 1.2, 0.8
    y_center = 0.5
    
    # Layer definitions
    layers = [
        {"name": "Input Layer\n(Pollutants & Temporal)", "x": 1.0},
        {"name": "LSTM Layer 1\n(128 Units)", "x": 3.0},
        {"name": "Dropout\n(p = 0.05)", "x": 5.0},
        {"name": "LSTM Layer 2\n(64 Units)", "x": 7.0},
        {"name": "Dropout\n(p = 0.05)", "x": 9.0},
        {"name": "Dense Layer\n(Output: AQI)", "x": 11.0},
    ]

    for layer in layers:
        x = layer["x"]
        
        # Draw box
        rect = patches.FancyBboxPatch(
            (x - width/2, y_center - height/2), width, height,
            boxstyle="round,pad=0.1",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=1.5,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(x, y_center, layer["name"], 
                ha='center', va='center', 
                fontsize=10, fontweight='bold', color=text_color, zorder=3)

    # Draw arrows between boxes
    for i in range(len(layers) - 1):
        x_start = layers[i]["x"] + width/2 + 0.1
        x_end = layers[i+1]["x"] - width/2 - 0.1
        
        ax.annotate('', 
                    xy=(x_end, y_center), 
                    xytext=(x_start, y_center),
                    arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2),
                    zorder=1)

    # Set limits
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    create_lstm_diagram("D:/dev/python/ph-aqi/figs/model_comparison/lstm_architecture.png")
