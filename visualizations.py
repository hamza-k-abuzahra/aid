import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.animation as animation

from algorithms import compute_MST
from report import normalize_metrics


def animate_visit(sim, frames=1000, interval=200, visiting_leader_id=0, save_as=None):
    """Animate the simulation with visualization style from visualize_visit()."""
    
    # Static data
    initial_positions = {uav.id: uav.initial_pos for uav in sim.uavs}
    paths = {uav.id: uav.path for uav in sim.uavs}
    
    leader = sim.uavs[visiting_leader_id]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, sim.map_size)
    ax.set_ylim(0, sim.map_size)

    # Legend elements
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Sensor (visited)',
              markerfacecolor='tomato', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Sensor (unvisited)',
              markerfacecolor='gold', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='P', color='w', label='Current Leader',
              markerfacecolor='limegreen', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='X', color='w', label='Other Leaders',
              markerfacecolor='crimson', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', label='Assistants',
              markerfacecolor='navy', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='UAVs with Info',
              markerfacecolor='blue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='v', color='w', label='Other UAVs',
              markerfacecolor='royalblue', markersize=10, markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.31, 1.0))

    # Containers for dynamic plots
    scatter_elems = []
    text_elems = []
    halo_elems = []
    path_lines = []

    def update(frame):
        # Clear old dynamic artists
        for elem in scatter_elems + text_elems + halo_elems + path_lines:
            elem.remove()
        scatter_elems.clear()
        text_elems.clear()
        halo_elems.clear()
        path_lines.clear()

        # Plot sensors
        for uav in sim.uavs:
            color = 'tomato' if leader.id in uav.sensor_info else 'gold'
            marker = 's' if leader.id in uav.sensor_info else 'o'
            scatter_elems.append(
                ax.scatter(
                    *initial_positions[uav.id], c=color, marker=marker,
                    s=100, zorder=1, linewidth=1, edgecolor='black'
                )
            )

        # Plot UAVs
        for uav in sim.uavs:
            pos = uav.curr_pos
            if uav.id == leader.id:
                color, marker = 'limegreen', 'P'
            elif uav.is_leader:
                color, marker = 'crimson', 'X'
            elif uav in leader.assistants:
                color, marker = 'navy', 'D'
            elif leader.id in uav.info:
                color, marker = 'blue', '^'
            else:
                color, marker = 'royalblue', 'v'

            scatter_elems.append(
                ax.scatter(
                    *pos, c=color, marker=marker, s=50,
                    zorder=2, edgecolor='black', linewidth=2
                )
            )

            text_elems.append(
                ax.text(
                    pos[0], pos[1] + 4, str(uav.id + 1),
                    color='black', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'),
                    zorder=3
                )
            )

            halo_elems.append(
                ax.scatter(
                    *pos, c=color, marker='o', s=550, alpha=0.3,
                    zorder=2, linewidth=0
                )
            )

        # Plot paths for leader and assistants
        for uav in sim.uavs:
            if uav.path and (uav in leader.assistants or uav.id == visiting_leader_id):
                path_coords = [uav.curr_pos] + [initial_positions[p] for p in uav.path]
                path_coords = np.array(path_coords)
                line, = ax.plot(
                    path_coords[:, 0], path_coords[:, 1],
                    linestyle='dashed' if uav.is_leader else 'solid',
                    alpha=0.5, linewidth=1.5
                )
                path_lines.append(line)

        # Advance simulation
        if sim.step(): return []
        ax.set_title(f"UAV Simulation - Step {frame}")

        return scatter_elems + text_elems + halo_elems + path_lines

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=False, repeat=False)

    # Save if requested
    if save_as:
        if save_as.endswith(".gif"):
            ani.save(save_as, writer="pillow", fps=1000//interval)
        elif save_as.endswith(".mp4"):
            ani.save(save_as, writer="ffmpeg", fps=1000//interval)
        else:
            raise ValueError("Unsupported format. Use .gif or .mp4")

    return ani


# 2. TIMESTEPS
def visualize_visit(sim, visiting_id):

    # Precompute data that doesn't change per leader
    initial_positions = {uav.id: uav.initial_pos for uav in sim.uavs}
    current_positions = {uav.id: uav.curr_pos for uav in sim.uavs}
    paths = {uav.id: uav.path for uav in sim.uavs}
    
    leader = sim.uavs[visiting_id] 
    if not leader.is_leader: leader = leader.leader
    # for leader in [u for u in sim.uavs if u.is_leader]:
    # Prepare plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, sim.map_size)
    ax.set_ylim(0, sim.map_size)
    # ax.set_title(f"System Status at Step {sim.steps}")
    
    # Create directory for this leader
    leader_dir = os.path.join(sim.folder_name, f"leader_{leader.id}")
    # os.makedirs(leader_dir, exist_ok=True)
    
    # Plot initial positions (sensors)
    for uav in sim.uavs:
        color = 'tomato' if leader.id in uav.sensor_info else 'gold'
        marker = 's' if leader.id in uav.sensor_info else 'o'
        ax.scatter(
            *initial_positions[uav.id], 
            c=color,
            marker=marker,
            s=100,
            zorder=1,
            linewidth=1,
            edgecolor='black'  # Outline for better visibility
        )

    # Plot current positions
    uav_colors = []
    uav_markers = []
    uav_positions = []
    uav_markers = {
        'leader': ('limegreen', 'P'),
        'other_leaders': ('crimson', 'X'),
        'assistants': ('navy', 'D'),
        'has_info': ('blue', '^'),
        'others': ('royalblue', 'v') 
    }


    for uav in sim.uavs:
        pos = current_positions[uav.id]

        uav_positions.append(current_positions[uav.id])
        if uav.id == leader.id:
            color, marker = uav_markers['leader']
        elif uav.is_leader:
            color, marker = uav_markers['other_leaders']
        elif uav in leader.assistants:
            color, marker = uav_markers['assistants']
        elif leader.id in uav.info:
            color, marker = uav_markers['has_info']
        else:
            color, marker = uav_markers['others']

        ax.scatter(
            *pos, 
            c=color,
            marker=marker,
            s=50,
            zorder=2,
            edgecolor='black',
            linewidth=2
        )

        ax.text(
            pos[0], pos[1] + 4, str(uav.id + 1),
            color='black',
            fontsize=12,
            ha='center',
            va='center',
            bbox=dict(
                facecolor='white',
                alpha=0.7,
                edgecolor='none',
                boxstyle='round,pad=0.2'
            ),
            zorder=3
        )
        
        # Halo effect (larger, transparent)
        ax.scatter(
            *pos,
            c=color,
            marker='o',
            s=550,
            alpha=0.3,
            zorder=2,
            linewidth=0
        )

    # ax.scatter(
    #     *zip(*uav_positions), 
    #     c=uav_colors, 
    #     marker=uav_markers,
    #     s=50, 
    #     zorder=2
    # )

    # ax.scatter(
    #     *zip(*uav_positions), 
    #     s=550,
    #     c=uav_colors, 
    #     alpha=0.3,
    #     zorder=2
    # )
    
    # Plot paths (only for current leader and its assistants)
    for uav in sim.uavs:
        if paths[uav.id] and (uav in leader.assistants or uav.id == leader.id):
            path_coords = [current_positions[uav.id]] + [initial_positions[p] for p in paths[uav.id]]
            path_coords = np.array(path_coords)
            ax.plot(path_coords[:, 0], path_coords[:, 1], linestyle='dashed' if uav.is_leader else 'solid', alpha=0.5, linewidth=1.5, label=f'Path {uav.id}')
    

    legend_elements = [
        # Sensors
        Line2D([0], [0], marker='s', color='w', label='Sensor (visited)',
              markerfacecolor='tomato', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='o', color='w', label='Sensor (unvisited)',
              markerfacecolor='gold', markersize=10, markeredgecolor='black'),
        # UAVs
        Line2D([0], [0], marker='P', color='w', label='Current Leader',
              markerfacecolor='limegreen', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='X', color='w', label='Other Leaders',
              markerfacecolor='crimson', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w', label='Assistants',
              markerfacecolor='navy', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='^', color='w', label='UAVs with Info',
              markerfacecolor='blue', markersize=10, markeredgecolor='black'),
        Line2D([0], [0], marker='v', color='w', label='Other UAVs',
              markerfacecolor='royalblue', markersize=10, markeredgecolor='black'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.31, 1.0))
    
    # Save figure
    filename = f"step_{sim.steps}_vl_({visiting_id})_leader_{leader.id}.png"
    plt.savefig(os.path.join(leader_dir, filename), dpi=100, bbox_inches='tight')
    plt.close(fig)


def plot_agent_performance(uavs, folder_name):
        # Prepare data
    agent_labels = []
    distances = []
    assistants = []
    comm_steps = []
    colors = []
    
    for uav in uavs:
        agent_labels.append(f"Leader {uav.id}" if uav.is_leader else f"Agent {uav.id}")
        distances.append(uav.total_distance_travelled)
        assistants.append(len(uav.assistants))
        comm_steps.append(uav.total_comm_visits)
        colors.append('red' if uav.is_leader else 'blue')

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Main bars for distance
    bars = ax.bar(agent_labels, distances, color=colors, alpha=0.7)
    
    # Add annotations above bars
    max_dist = max(distances) if distances else 0
    for bar, count, comm_step_count in zip(bars, assistants, comm_steps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_dist*0.02,
               f'{count} Asst ({comm_step_count})' if count > 0 else f'No Asst ({comm_step_count})',
               ha='center', va='bottom', fontsize=8)

    # Customize plot
    ax.set_xlabel("Agent")
    ax.set_ylabel("Distance Traveled")
    ax.set_title(f"Agent Performance (Total UAVs: {len(uavs)})\nDistance Traveled and Assistant Assignments")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Create legend
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Leaders'),
        Patch(facecolor='blue', alpha=0.7, label='Agents'),
        Patch(facecolor='white', label='Text: Assistant Count (Comm Visits)')
    ]
    ax.legend(handles=legend_elements)
    
    # Save plot
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, "agent_performance_plot.png")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_initial_configuration(uavs, folder_name, map_size):
    # Extract positions and colors
    positions = np.array([uav.initial_pos for uav in uavs])
    colors = ['red' if uav.is_leader else 'blue' for uav in uavs]
    
    # Compute MST
    mst = compute_MST(positions)
    
    # Find longest edge
    max_i, max_j = np.unravel_index(mst.argmax(), mst.shape)
    longest_edge_length = mst[max_i, max_j]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot all agents
    for i, (pos, color) in enumerate(zip(positions, colors)):
        ax.scatter(*pos, color=color, s=100, 
                  label=f'Leader {i}' if color == 'red' else f'Agent {i}')

    # Plot MST edges
    for i in range(mst.shape[0]):
        for j in range(i+1, mst.shape[1]):  # Only upper triangle to avoid duplicates
            if mst[i,j] > 0:
                ax.plot([positions[i,0], positions[j,0]],
                       [positions[i,1], positions[j,1]], 
                       'gray', linewidth=2, alpha=0.3)

    # Highlight longest edge
    ax.plot([positions[max_i,0], positions[max_j,0]],
           [positions[max_i,1], positions[max_j,1]], 
           'r-', linewidth=2, 
           label=f'Longest MST edge: {longest_edge_length:.2f}')

    # Add annotation
    midpoint = (positions[max_i] + positions[max_j])/2
    ax.annotate(f'{longest_edge_length:.2f}', 
               xy=midpoint, 
               xytext=(5,10),
               fontsize=20,
               textcoords='offset points',
               ha='center',
               color='red',
               weight='bold')

    # Configure plot
    # ax.set_xlim(0, map_size)
    # ax.set_ylim(0, map_size)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax.set_xlabel("X Position", fontsize=25)
    ax.set_ylabel("Y Position", fontsize=25)
    # ax.set_title("Initial Positions (Longest MST Edge Highlighted)")
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, "initial_pos_with_mst.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# single configuration, all metrics
def plot_single_result(results_dict: dict, grouping_field: str = 'n_agents', grouping_value: str = "10", x_field: str ="k", 
                       metric: str = 'avg_max_dist', comparison_metric: str = 'avg_leader_max_dist', secondary_metric: str = 'avg_assistants', 
                       normalized: bool = False, save_path: str = 'results') -> None: 
        
    if grouping_value not in results_dict:
        raise ValueError(f"No data for {grouping_value} {grouping_field}")
    
    results = results_dict[grouping_value]
    if not results:
        raise ValueError(f"No results available for {grouping_value} {grouping_field}")
    
    # Apply normalization if requested
    if normalized:
        results = [normalize_metrics(r) for r in results]
        metric = f"norm_{metric}"
        if comparison_metric:
            comparison_metric = f"norm_{comparison_metric}"

    # Extract and sort data
    x_values = np.array([r[x_field] for r in results])
    y_values = np.array([r[metric] for r in results])
    comp_values = np.array([r[comparison_metric] for r in results]) if comparison_metric else None
    sec_values = np.array([r[secondary_metric] for r in results]) if secondary_metric else None
    
    sort_idx = np.argsort(x_values)
    x_values = x_values[sort_idx]
    y_values = y_values[sort_idx]
    if comp_values is not None: comp_values = comp_values[sort_idx]
    if sec_values is not None: sec_values = sec_values[sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    max_y_value = max(y_values)

    if comp_values is not None: max_y_value = max(max(comp_values), max_y_value)

    plt.ylim([0, max_y_value + 30])

    
    # Plot primary metric
    ax.set_xlabel(x_field.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), color='blue', fontsize=12)
    primary_line = ax.plot(x_values, y_values, 'o-', linewidth=2, markersize=8,
                         color='blue', label=metric.replace('_', ' ').title())
    ax.tick_params(axis='y', labelcolor='blue')

    # Plot comparison metric
    if comparison_metric:
        comp_line = ax.plot(x_values, comp_values, 's--', linewidth=2, markersize=6,
                          color='green', alpha=0.7, 
                          label=comparison_metric.replace('_', ' ').title())

    # Plot secondary metric on twin axis
    if secondary_metric:
        ax2 = ax.twinx()
        ax2.set_ylabel(secondary_metric.replace('_', ' ').title(), 
                      color='red', fontsize=12)
        sec_line = ax2.plot(x_values, sec_values, '^:', linewidth=1, markersize=6,
                          color='red', alpha=0.7, label=f'{secondary_metric}')
        ax2.tick_params(axis='y', labelcolor='red')

    # Combine legends
    lines = primary_line
    if comparison_metric: lines += comp_line
    if secondary_metric: lines += sec_line
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')

    # Title and save
    config_str = f"{grouping_field}={grouping_value}"
    if 'longest_edge' in results[0] and grouping_field == "n_agents":
        is_sparse = 'Sparse' if np.mean([r["longest_edge"] for r in results]) > grouping_value else 'Dense'
        config_str += f" ({is_sparse})"
    
    plt.title(f"{metric.replace('_', ' ').title()} vs {x_field.replace('_', ' ').title()} ({config_str})", 
             fontsize=14)

    # Annotate extremes
    max_idx = np.argmax(y_values)
    min_idx = np.argmin(y_values)

    ax.annotate(f'Max: {y_values[max_idx]:.1f}', 
                (x_values[max_idx], y_values[max_idx]),
                textcoords="offset points", xytext=(0,15), ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
    ax.annotate(f'Min: {y_values[min_idx]:.1f}', 
                (x_values[min_idx], y_values[min_idx]),
                textcoords="offset points", xytext=(0,-20), ha='center',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.3))
    

    norm_suffix = '_normalized' if normalized else ''
    comp_suffix = f"_vs_{comparison_metric}" if comparison_metric else ""
    filename = f"results_{grouping_field}_{grouping_value}_{metric}{norm_suffix}_by_{x_field}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    

# single configuration, single metric
def plot_single_metric(results_dict: dict, metric: str, group_by: str = 'n_agents', x_field: str = 'k', style: str = 'line', save_path: str = 'results') -> None:

    # Prepare data
    groups = sorted([k for k in results_dict.keys() if isinstance(k, (int, float))])
    
    plt.figure(figsize=(12, 7))
    
    # Color map for different groups
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))
    
    for group, color in zip(groups, colors):
        if group not in results_dict:
            continue
            
        # Sort and extract data
        results = sorted(results_dict[group], key=lambda x: x[x_field])
        x_values = [r[x_field] for r in results]
        y_values = [r[metric] for r in results]
        
        # Plot based on selected style
        if style == 'line':
            plt.plot(x_values, y_values, 'o-', color=color, 
                    linewidth=2, markersize=8, label=f'{group} {group_by}')
        elif style == 'scatter':
            plt.scatter(x_values, y_values, color=color, 
                       s=100, alpha=0.7, label=f'{group} {group_by}')
        elif style == 'bar':
            plt.bar([x + group/len(groups) for x in range(len(x_values))], 
                   y_values, width=0.8/len(groups), color=color,
                   label=f'{group} {group_by}')
    
    # Formatting
    plt.title(f"{metric.replace('_', ' ').title()} by {x_field.replace('_', ' ').title()}", 
             fontsize=14)
    plt.xlabel(x_field.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title=group_by.replace('_', ' ').title())
    plt.ylim(bottom=0)
    
    # For bar plots, set x-ticks
    if style == 'bar':
        plt.xticks(range(len(x_values)), x_values)
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    filename = f"{metric}_by_{x_field}_grouped_by_{group_by}_{style}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


# across different configurations, one metric for evaulation
def plot_comparative_metrics(results_dict: dict, grouping_field: str = "n_agents", x_field: str = 'k', metric: str = 'avg_total_dist', normalized: bool = False, save_path: str = 'results') -> None:
    # Prepare data
    plot_data = {}
    group_values = sorted([k for k in results_dict.keys() if isinstance(k, (int, float))])
    
    for group_value in group_values:
        if group_value not in results_dict or not results_dict[group_value]:
            continue
            
        # Apply normalization if requested
        if normalized:
            results = [normalize_metrics(r) for r in results_dict[group_value]]
            metric_key = f"norm_{metric}"
        else:
            results = results_dict[group_value]
            metric_key = metric
        
        # Sort and extract data
        sorted_results = sorted(results, key=lambda x: x[x_field])
        x_values = [r[x_field] for r in sorted_results]
        y_values = [r[metric_key] for r in sorted_results]
        
        plot_data[group_value] = (x_values, y_values)
    
    # Create plot
    plt.figure(figsize=(12, 7))
    
    # Color map for different agent counts
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
    
    # Plot each agent count
    for (group_value, (x, y)), color in zip(plot_data.items(), colors):
        plt.plot(x, y, 'o-', color=color, linewidth=2, markersize=8,
                label=f'{group_value} {grouping_field}')
    
    # Formatting
    x_label = f"{x_field.replace('_', ' ').title()}"
    
    plt.title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    
    y_label = f"{metric.replace('_', ' ').title()}"
    if normalized:
        y_label += " (normalized)"
    plt.ylabel(y_label, fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.legend(title=f'{grouping_field}')
    plt.ylim(bottom=0)
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    norm_suffix = '_normalized' if normalized else ''
    filename = f"comparison_{metric}{norm_suffix}_by_{x_field}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparative_algorithms(names, results_dict: dict, comparison_field: str = "n_agents", comparison_value: str = "10", x_field: str = 'k', 
        metric: str = 'avg_total_dist', normalized: bool = False, include_std_dev: bool = False, save_path: str = 'results', bench_value: int = None) -> None:

    # Prepare data
    plot_data = {}
    algorithms = [k for k in results_dict.keys() if isinstance(k, (str))]
    
    
    markers = ['s', 'o', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd']


    for alg in algorithms:
        if alg not in names: 
            continue
        if alg not in results_dict or not results_dict[alg]:
            continue
        
        if normalized:
            results = [normalize_metrics(r) for r in results_dict[alg][comparison_value]]
            metric_key = f"norm_{metric}"
        else:
            results = results_dict[alg][comparison_value]
            metric_key = metric
        
    # for group_value in group_values:
        # if group_value not in results_dict or not results_dict[group_value]:
        #     continue
            
        # Apply normalization if requested
        # if normalized:
        #     results = [normalize_metrics(r) for r in results_dict[group_value]]
        #     metric_key = f"norm_{metric}"
        # else:
        #     results = results_dict[group_value]
        #     metric_key = metric
        
        # Sort and extract data
        sorted_results = sorted(results, key=lambda x: x[x_field])
        if comparison_field == "n_agents":
            x_values = [int(r[x_field]) / int(comparison_value) for r in sorted_results]
        else:
            x_values = [int(r[x_field]) for r in sorted_results] 
        y_values = [r[metric_key] for r in sorted_results]
        if include_std_dev:
            m = "_".join(metric.split('_')[1:])
            y_std = [r["stats"][f"std_dev_{m}"] for r in sorted_results]
        else: y_std = None
        plot_data[alg] = (x_values, y_values, y_std)
    

    # TSP benchmark
    if bench_value: 
        results = results_dict[alg][comparison_value]
        sorted_results = sorted(results, key=lambda x: x[x_field])

        # if comparison_field == "n_agents":
        bench_x_values = [int(r[x_field]) / int(comparison_value) for r in sorted_results]
        # else:
        bench_x_values_i = [int(r[x_field]) for r in sorted_results] 
        
        
        bench_y_values = [bench_value * i if metric == "avg_total_dist" else bench_value for i in bench_x_values_i ]
        plot_data["benchmark"] = (bench_x_values, bench_y_values, ())

    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Color map for different agent counts
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))



    # Plot each agent count
    # for (alg, (x, y)), color in zip(plot_data.items(), colors):
    for i, (alg, (x, y, y_std)) in enumerate(plot_data.items()):
        marker = markers[i % len(markers)]  # Cycle through markers
        x = np.array(x)
        y = np.array(y)
        y_std = np.array(y_std)
        
        if alg == "benchmark": 
            linestyle = "--"
        else: 
            linestyle = "-"

        plt.plot(x, y, marker=marker, color=colors[i], linestyle=linestyle, linewidth=1, markersize=12,
            label=names[alg], markeredgecolor='black', markeredgewidth=0.5)
        if include_std_dev:
            plt.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=colors[i],
                alpha=0.2
            )
        # plt.plot(x, y, 'o-', color=color, linewidth=2, markersize=8,
                # label=f'{alg}')
    
    # Formatting
    x_label = f"{x_field.split('_')[0]}"    
    if x_label == "k": 
        x_label = r"Leader Percentage ($\rho$)"


    
    # plt.title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=14)
    plt.xlabel(x_label, fontsize=18)
    # plt.rc("axes", labelsize=18)

    # for label in (plt.get_xticklabels() + plt.get_yticklabels()):
    #     label.set_fontsize(16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # y_label = f"{metric.replace('_', ' ').title()}"
    # y_label = "Distance Travelled"
    if "dist" in metric:
        y_label = "Distance Travelled"
    elif "comm" in metric:
        y_label = "Communication Count"
    else: 
        y_label = f"{metric.replace('_', ' ').title()}"

    if normalized:
        y_label += " (normalized)"
    plt.ylabel(y_label, fontsize=18)
    
    # plt.ylim(180, 370)

    # plt.ylim(0.85, 1.0)
    # plt.axhline(0, linestyle="--", linewidth=0.8)


    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    
    # Save plot
    os.makedirs(save_path, exist_ok=True)
    norm_suffix = '_normalized' if normalized else ''
    filename = f"comparison_{metric}{norm_suffix}_by_{x_field}_in_{comparison_value}_{comparison_field}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


# plot graphs
def plot_graph_comparative_algorithms(names, results_dict: dict, comparison_field: str = "n_agents", 
                              comparison_value: str = "10", x_field: str = 'k', 
                              metric: str = 'avg_total_dist', normalized: bool = False, 
                              save_path: str = 'results') -> None:
    # Prepare data - select only 5 values with 5% steps
    algorithms = [k for k in results_dict.keys() if isinstance(k, str)]

    all_x_values = sorted(list({r[x_field] for alg in algorithms 
                              for r in results_dict[alg][comparison_value]}))
    
    # Select 5 values with ~5% steps (handles non-perfect divisions)
    step = max(1, len(all_x_values) // 5)  # Ensure at least 1 step
    x_values = all_x_values[::step][:5]  # Take first 5 with step

    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Bar settings
    bar_width = 0.12  # Fixed width for 5 algorithms (adjust if needed)
    group_spacing = 0.05  # Space between groups of bars
    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']  # B&W patterns
    colors = plt.cm.tab10.colors  # Color-blind friendly
    
    for i, alg in enumerate(algorithms):
        if alg not in results_dict or not results_dict[alg]:
            continue
        
        # Get data for selected x_values only
        results = {r[x_field]: r[f"norm_{metric}" if normalized else metric] 
                  for r in results_dict[alg][comparison_value]}
        y_values = [results.get(x, 0) for x in x_values]  # 0 if missing
        
        # Position bars
        x_pos = [x * len(algorithms) * (bar_width + group_spacing) + i * bar_width
                 for x in range(len(x_values))]
        
        plt.bar(
            x_pos, y_values,
            width=bar_width,
            color=colors[i],
            edgecolor='black',
            hatch=patterns[i % len(patterns)],
            label=names[alg]
        )
    
    # Axis formatting
    # plt.xlabel(f"{x_field.replace('_', ' ').title()}", fontsize=12)
    plt.xlabel(f"{x_field.split('_')[0]}", fontsize=12)
    # y_label = f"{metric.replace('_', ' ').title()}{' (normalized)' if normalized else ''}"
    y_label = "Communication Count"
    plt.ylabel(y_label, fontsize=12)
    plt.ylim(0, 40)
  
    # X-axis ticks (center under grouped bars)
    plt.xticks(
        [x * (bar_width + group_spacing) * len(algorithms) + (len(algorithms)-1)*bar_width/2  for x in range(len(x_values))],
        # [x / comparison_value for x in x_values]  # Force percentage format
        x_values
    )
    # plt.xlim(-group_spacing, tick_positions[-1] + group_spacing)

    # Legend and grid
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(
        bbox_to_anchor=(0, 1),
        loc='upper left'
    )
    
    # Save
    os.makedirs(save_path, exist_ok=True)
    filename = f"bar_5step_{metric}_{comparison_value}{'_normalized' if normalized else ''}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


# plot graphs
def plot_graph_single_configuration(
    names,
    results_dict: dict,
    comparison_field: str = "n_agents",
    comparison_value: str = "10",
    x_field: str = "k",
    x_value=None,                     # <-- SINGLE configuration
    metric: str = "avg_max_dist",
    normalized: bool = False,
    save_path: str = "results"
) -> None:

    algorithms = [k for k in results_dict.keys() if isinstance(k, str)]

    if x_value is None:
        raise ValueError("x_value must be provided for a single-configuration plot")

    y_values = []
    labels = []

    for alg in algorithms:
        if alg not in results_dict or not results_dict[alg] or alg not in names.keys():
            continue

        # Find the result matching the chosen configuration
        matching = [
            r for r in results_dict[alg][int(comparison_value)]
            if r[x_field] == float(x_value) * int(comparison_value)
        ]

        if not matching:
            continue

        value = matching[0][f"norm_{metric}" if normalized else metric]
        y_values.append(value)
        labels.append(names[alg])

    # Plot
    plt.figure(figsize=(8, 5))

    patterns = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    colors = plt.cm.tab10.colors

    x_pos = range(len(y_values))

    best_idx = min(range(len(y_values)), key=lambda i: y_values[i])

    bars = []
 
    for i, y in enumerate(y_values):
        bar = plt.bar(
            i, y,
            # color=colors[i % len(colors)],
            # color=colors[0],
            color="white",
            edgecolor='black',
            linewidth=2.0 if i != best_idx else 8,  # <-- highlight
            hatch=patterns[i % len(patterns)]
        )
        bars.append(bar)

    # Inside your loop:
    for i, (bar, y, label) in enumerate(zip(bars, y_values, labels)):
        # Position text inside bar
        plt.text(
            bar[0].get_x() + bar[0].get_width()/2.,
            y / 2,  # Halfway up the bar
            label,
            ha='center', va='center',
            fontsize=24, fontweight='bold',
            # color='white' if y > max(y_values)*0.5 else 'black', # Contrast
            color="black",
            rotation=90,
        )
    # # Add value labels on top of bars
    # for i, (bar, y, label) in enumerate(zip(bars, y_values, labels)):
    #     plt.text(
    #         bar[0].get_x() + bar[0].get_width()/2.,  # x position (center of bar)
    #         y + (max(y_values) * 0.02),  # y position (slightly above bar)
    #         label,  # Format the number
    #         ha='center', va='bottom',
    #         fontsize=14, fontweight='bold'
    #     )

    # plt.xticks(x_pos, labels, rotation=20, fontsize=20)
    
    plt.yticks(fontsize=20)
    
    plt.ylabel("Distance Travelled", fontsize=20)
    # plt.ylim(0, max(y_values) * 1.15)   # include 0, scale nicely
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"bar_single_{metric}_{comparison_value}_{x_field}{x_value}{'_normalized' if normalized else ''}.png"
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()



def plot_win_table_heatmap(
    names,
    win_table,
    algorithms=None,
    normalize="win_rate",  # "win_rate" or "score"
    title=None,
    figsize=(10, 6),
    annotate=True,
):
    """
    normalize:
        - "win_rate": wins / (wins + ties + losses)
        - "score": (wins + 0.5 * ties) / total
    """

    # Sort configs for stable visualization
    config_labels = sorted(win_table.keys())

    if algorithms is None:
        algorithms = sorted(next(iter(win_table.values())).keys())

    data = np.zeros((len(config_labels), len(algorithms)))

    for i, cfg in enumerate(config_labels):
        for j, alg in enumerate(algorithms):
            stats = win_table[cfg][alg]
            wins = stats["wins"]
            ties = stats["ties"]
            losses = stats["losses"]
            total = wins + ties + losses

            if total == 0:
                value = np.nan
            else:
                if normalize == "win_rate":
                    value =(((wins - losses) / total) + 1) /2
                elif normalize == "score":
                    value = (wins + 0.5 * ties) / total
                else:
                    raise ValueError("Unknown normalization")

            data[i, j] = value

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", vmin=0, vmax=1)

    algorithms_display_name = [names[alg] for alg in algorithms]
    # Axes labels
    ax.set_xticks(np.arange(len(algorithms)))
    ax.set_xticklabels(algorithms_display_name, rotation=45, ha="right", fontsize=22)
    ax.set_yticks(np.arange(len(config_labels)))
    ax.set_yticklabels(config_labels, fontsize=22)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    # cbar.set_label("Win rate" if normalize == "win_rate" else "Score")

    # Cell annotations
    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(
                        j, i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=18,
                        color="white" if data[i, j] < 0.75 else "white"
                    )

    # if title:
        # ax.set_title(title)

    plt.tight_layout()
    plt.show()
