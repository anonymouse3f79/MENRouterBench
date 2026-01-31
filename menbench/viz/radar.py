import numpy as np
import matplotlib.pyplot as plt
from menbench.utils import logger


def draw_radar_chart(model_stats:list[dict], save_path="radar_comparison.png", max_latency=30.0):
    """
    Draw multimodel comprehension radar chart.
    :param model_stats:
        [[
            acc_dir_yaw, acc_dir_lon, acc_dir_lat, macro_dir, 
            acc_num_yaw, acc_num_lon, acc_num_lat, macro_num, overall
        ], ...]
    :param save_path: The way to save png.
    :param max_latency
    """
    
    labels = [
        'Yaw-Dir', 'Lon-Dir', 'Lat-Dir', 
        'Yaw-Num', 'Lon-Num', 'Lat-Num',
        'Overall Score', 'Speed (1/Lat)'
    ]
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for stats in model_stats:
        speed_score = max(0, 1 - (stats["scores_all"][9] / max_latency))
        
        values = [
            stats["scores_all"][0], stats["scores_all"][1], stats["scores_all"][2],
            stats["scores_all"][4], stats["scores_all"][5], stats["scores_all"][6],
            stats["scores_all"][8], speed_score
        ]
        values += values[:1] 

        ax.plot(angles, values, linewidth=2, label=stats['model'])
        ax.fill(angles, values, alpha=0.25)

    ax.set_theta_offset(np.pi / 2) 
    ax.set_theta_direction(-1) 
    

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)

    ax.set_ylim(0, 1)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"])

    plt.title("Multi-Model Task 1 Performance Comparison", y=1.08)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    logger.info(f"[*] Radar chart saved to {save_path}")

if __name__ == "__main__":
    data = model_comparison_data = [
    {
        'model': 'GPT-4o',
        'scores_all': [0.85, 0.95, 0.88,0, 0.72, 0.90, 0.80,0, 0.85, 0.8]
    },
]
    draw_radar_chart(model_stats=data)