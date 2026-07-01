import numpy as np
import os, csv
from pathlib import Path
from collections import defaultdict
from utils import write_csv, walk_naturally_sorted

def average_reports(reports):
    if not reports:
        return {}
    
    n_agents = reports[0]['n_agents']

    # Initialize storage for sorted metrics
    sorted_metrics = {
        'distances': [[] for _ in range(n_agents)],
        'leader_flags': [[] for _ in range(n_agents)],
        'assistants': [[] for _ in range(n_agents)], 
        'comm_visits_count': [[] for _ in range(n_agents)]
    }
    
    # Process each run
    for run in reports:
        sorted_indices = np.argsort(run['distances'])
        
        for rank in range(n_agents):
            orig_idx = sorted_indices[rank]
            sorted_metrics['distances'][rank].append(run['distances'][orig_idx])
            sorted_metrics['leader_flags'][rank].append(run['is_leader_list'][orig_idx])
            sorted_metrics['assistants'][rank].append(run['assistants_count'][orig_idx])
            sorted_metrics['comm_visits_count'][rank].append(run['comm_visits_count'][orig_idx])

    
    aggregates = {
        'by_distance_rank': {
            'dist': {
                'avg': [np.mean(dist) for dist in sorted_metrics['distances']],
                'max': [np.max(dist) for dist in sorted_metrics['distances']],
                'min': [np.min(dist) for dist in sorted_metrics['distances']]
            },
            'leader_prob': [np.mean(flags) for flags in sorted_metrics['leader_flags']],
            'assistants': {
                'avg': [np.mean(a) for a in sorted_metrics['assistants']],
                'max': [np.max(a) for a in sorted_metrics['assistants']],
                'min': [np.min(a) for a in sorted_metrics['assistants']]
            },
            'comm_visits_count': {
                'avg': [np.mean(a) for a in sorted_metrics['comm_visits_count']],
                'max': [np.max(a) for a in sorted_metrics['comm_visits_count']],
                'min': [np.min(a) for a in sorted_metrics['comm_visits_count']]
            }
        }, 
        'avg_total_dist': np.mean([r['total_dist'] for r in reports]),
        'avg_avg_dist': np.mean([r['avg_dist'] for r in reports]),
        'avg_max_dist': np.mean([r['max_dist'] for r in reports]),

        'avg_total_comm': np.mean([r['total_comm'] for r in reports]),
        'avg_avg_comm': np.mean([r['avg_comm'] for r in reports]),
        'avg_max_comm': np.mean([r['max_comm'] for r in reports]),
        

        # leader stats
        'avg_leader_total_dist': np.mean([r['leader_total_dist'] for r in reports]),
        'avg_leader_avg_dist': np.mean([r['leader_avg_dist'] for r in reports]),
        'avg_leader_max_dist': np.mean([r['leader_max_dist'] for r in reports]),
        'avg_leader_total_comm': np.mean([r['leader_total_comm'] for r in reports]),
        'avg_leader_avg_comm': np.mean([r['leader_avg_comm'] for r in reports]),
        'avg_leader_max_comm': np.mean([r['leader_max_comm'] for r in reports]),

        'avg_assistants': np.mean([r['total_assistants'] for r in reports]),
        'avg_steps': np.mean([r['steps'] for r in reports]),
        
        'map_size': reports[0]['map_size'],
        'k': reports[0]['k'],
        'n_agents': reports[0]['n_agents'],
        'longest_edge': np.mean([r['longest_edge'] for r in reports])
    }
    return aggregates




def log_single_report(report: dict, folder_name: str):
    
    # Separate scalar and list values
    global_attrs = {k: v for k, v in report.items() if not isinstance(v, list)}
    per_agent_attrs = {k: v for k, v in report.items() if isinstance(v, list)}

    # Write global report (scalar values)
    write_csv(
        path=os.path.join(folder_name, "report.txt"),
        headers=global_attrs.keys(),
        rows=[global_attrs.values()],
    )
    
    # Write per-agent data if exists
    if per_agent_attrs:
        # Transpose the data to get per-agent rows
        per_agent_rows = zip(*per_agent_attrs.values())
        write_csv(
            path=os.path.join(folder_name, "per_agent.txt"),
            headers=per_agent_attrs.keys(),
            rows=per_agent_rows,
        )


def log_averaged_report(report: dict, folder_name: str):
    
    # Separate scalar and nested values
    global_attrs = {k: v for k, v in report.items() if not isinstance(v, dict)}
    nested_attrs = {k: v for k, v in report.items() if isinstance(v, dict)}
    
    # Write global report (scalar values)
    write_csv(
        path=os.path.join(folder_name, "avg_report.txt"),
        headers=global_attrs.keys(),
        rows=[global_attrs.values()],
    )
    
    # Process nested attributes if they exist
    if nested_attrs:
        # Flatten the nested structure into columns
        flat_headers = []
        flat_rows = []
        
        # Get the length from the first nested attribute's first sub-attribute
        sample_attr = next(iter(nested_attrs.values()))
        sample_subattr = next(iter(sample_attr.values()))
        num_entries = len(sample_subattr) if isinstance(sample_subattr, list) else 1
        
        for i in range(num_entries):
            row = []
            for attr_name, nested_dict in nested_attrs.items():
                for sub_attr, values in nested_dict.items():
                    if i == 0:  # Only add headers once
                        flat_headers.append(f"{sub_attr}_{attr_name}")
                    
                    # Get the value for this row
                    if isinstance(values, list):
                        row.append(str(values[i]))
                    else:
                        row.append(str(values))
            flat_rows.append(row)
        
        write_csv(
            path=os.path.join(folder_name, "avg_per_agent.txt"),
            headers=flat_headers,
            rows=flat_rows,
        )


def parse_single_report(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if file_path.__str__().endswith("report.txt"):
            for row in reader:                
                return {
                    'map_size': int(row['map_size']),
                    'n_agents': int(row['n_agents']),
                    'k': int(row['k']),
                    'total_dist': float(row['total_dist']),
                    'avg_dist': float(row['avg_dist']),
                    'max_dist': float(row['max_dist']),
                    'total_comm': float(row['total_comm']),
                    'max_comm': float(row['max_comm']),
                    'avg_comm': float(row['avg_comm']),
                    'leader_total_dist': float(row['leader_total_dist']),
                    'leader_max_dist': float(row['leader_max_dist']),
                    'leader_avg_dist': float(row['leader_avg_dist']),
                    'leader_total_comm': float(row['leader_total_comm']),
                    'leader_max_comm': float(row['leader_max_comm']),
                    'leader_avg_comm': float(row['leader_avg_comm']),
                    'total_assistants': int(row['total_assistants']),
                    'steps': int(row['steps']),
                    # 'assigning_frequency': int(row['assigning_frequency']),
                    # 'assistant_limit': int(row['assistant_limit']),
                    'longest_edge': float(row['longest_edge']),
                }

def load_single_results(experiment_folder: str, group_by: str = "n_agents", second_grouping: str = "k"):
    grouped_results = defaultdict(lambda: defaultdict(list))

    for root, dirs, files in walk_naturally_sorted(experiment_folder):  
        for file in files: 
            if file.endswith("report.txt"):
                file_path = Path(root) / file
                try: 
                    results = parse_single_report(file_path)
                    group_value = results[group_by] # v/ results["n_agents"]
                    if group_by != "n_agents": group_value /= results["n_agents"]
                    
                    second_group_value = results[second_grouping]
                    if second_group_value not in grouped_results[group_value].keys():
                        grouped_results[group_value][second_group_value] = []
                    grouped_results[group_value][second_group_value].append(results)

                except Exception as e:
                    print(f"Error parsing {file_path}: {str(e)}")
    return grouped_results

def get_std_dev_of_metric(runs: list, metric: str):
    values = [run[metric] for run in runs]
    return np.std(values)


def write_std_dev_report(experiment_folder: str, output_file: str, metrics: list, group_by: str = "n_agents", second_grouping: str = "k"):
    grouped_results = load_single_results(experiment_folder, group_by, second_grouping)

    for metric in metrics: 
        std_dev_data = []
        for group_value, second_group_dict in grouped_results.items():
            for second_group_value, runs in second_group_dict.items():
                std_dev = get_std_dev_of_metric(runs, metric)
                std_dev_data.append({
                    group_by: group_value,
                    second_grouping: second_group_value,
                    f'std_dev_{metric}': std_dev
                })
    
        # Write to CSV
        write_csv(
            path= f"{metric}_{output_file}",
            headers=[group_by, second_grouping, f'std_dev_{metric}'],
            rows=[[d[group_by], d[second_grouping], d[f'std_dev_{metric}']] for d in std_dev_data]
        )


def load_std_dev_csv(experiment_folder, group_by, second_grouping, metrics=["max_dist", "avg_dist", "total_dist", "max_comm"]):
    std_lookup = defaultdict(lambda: defaultdict(dict))
    for metric in metrics:
        experiment_name = experiment_folder.split("/")[-1]
        
        csv_path = f"{metric}_{experiment_name}_std_dev_report.txt"
        try: 
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    group_by_value = int(row[group_by])
                    second_grouping_value = int(row[second_grouping])
                    std_lookup[group_by_value][second_grouping_value][f"std_dev_{metric}"] = float(
                        row[f"std_dev_{metric}"]
                    )
        except FileNotFoundError:
            print(f"{csv_path} not found.") 

    # print(std_lookup)
    return std_lookup

def get_max_assistants(file_path): 
    with open(file_path, newline='') as csvfile: 
        reader = csv.DictReader(csvfile)
        return max([r["assistants_count"] for r in reader])
        

def parse_single_report(file_path):     
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if file_path.__str__().endswith("report.txt"):
            path_e = file_path.__str__().split("\\")[:-1]
            path_e.append("per_agent.txt")
            path_e = "\\".join(path_e)
            max_assistants = get_max_assistants(path_e)
            for row in reader:
                return {
                    'map_size': float(row['map_size']),
                    'n_agents': int(row['n_agents']),
                    'k': int(row['k']),
                    'total_dist': float(row['total_dist']),
                    'avg_dist': float(row['avg_dist']),
                    'max_dist': float(row['max_dist']),
                    # 'assistants': float(row['assistants']),
                    'steps': float(row['steps']),

                    # added later - first 2 experiments don't have this information in their report and need to use load_per_agent_results()
                    'total_comm': float(row['total_comm']),
                    'avg_comm': float(row['avg_comm']),
                    'max_comm': float(row['max_comm']),
                    'leader_total_dist': float(row['leader_total_dist']),
                    'leader_avg_dist': float(row['leader_avg_dist']),
                    'leader_max_dist': float(row['leader_max_dist']),
                    'leader_total_comm': float(row['leader_total_comm']),
                    'leader_avg_comm': float(row['leader_avg_comm']),
                    'leader_max_comm': float(row['leader_max_comm']),
                    'longest_edge': float(row['longest_edge']),
                    'assistant_limmt': float(row['assistant_limit']),
                    'assigning_frequency': float(row['assigning_frequency']),
                    'total_assistants': float(row['total_assistants']),
                    'max_assistants': max_assistants,
                }

def parse_report_file(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if file_path.__str__().endswith("avg_report.txt"):
            for row in reader:
                return {
                    'map_size': float(row['map_size']),
                    'n_agents': int(row['n_agents']),
                    'k': int(row['k']),
                    'avg_total_dist': float(row['avg_total_dist']),
                    'avg_avg_dist': float(row['avg_avg_dist']),
                    'avg_max_dist': float(row['avg_max_dist']),
                    'avg_assistants': float(row['avg_assistants']),
                    'avg_steps': float(row['avg_steps']),

                    # added later - first 2 experiments don't have this information in their report and need to use load_per_agent_results()
                    'avg_total_comm': float(row['avg_total_comm']),
                    'avg_avg_comm': float(row['avg_avg_comm']),
                    'avg_max_comm': float(row['avg_max_comm']),
                    'avg_leader_total_dist': float(row['avg_leader_total_dist']),
                    'avg_leader_avg_dist': float(row['avg_leader_avg_dist']),
                    'avg_leader_max_dist': float(row['avg_leader_max_dist']),
                    'avg_leader_total_comm': float(row['avg_leader_total_comm']),
                    'avg_leader_avg_comm': float(row['avg_leader_avg_comm']),
                    'avg_leader_max_comm': float(row['avg_leader_max_comm']),
                    'longest_edge': float(row['longest_edge']),
                }


def load_results(experiment_folder: str, group_by: str = "n_agents", metrics=["max_dist", "avg_dist", "total_dist", "max_comm"]):
    grouped_results = defaultdict(list)
    second_grouping = "k" if group_by != "k" else "n_agents"
    std_lookup = load_std_dev_csv(experiment_folder, group_by, second_grouping, metrics)
    for root, dirs, files in walk_naturally_sorted(experiment_folder):  

        for file in files: 
            if file.endswith("avg_report.txt"):
                file_path = Path(root) / file
                try: 
                    results = parse_report_file(file_path)

                    results["stats"] = std_lookup[results[group_by]][results[second_grouping]] #?? 
                    group_value = results[group_by] # v/ results["n_agents"]
                    if group_by != "n_agents": group_value /= results["n_agents"]
                    grouped_results[group_value].append(results)

                except Exception as e:
                    print(f"Error parsing {file_path}: {str(e)}")
    return grouped_results


def load_multiple_results(experiment_folders: list, group_by: str ="n_agents", metrics=["max_dist", "avg_dist", "total_dist", "max_comm"]):
    results = dict()
    for experiment_folder in experiment_folders:
        experiment_title = experiment_folder.split("_")[1]
        results[experiment_title] = load_results(experiment_folder, group_by, metrics)

    return results


def classify_density(longest_edge, n):
    return "Dense" if (longest_edge * 2) < n  else "Sparse"

def classify_leader_ratio(n_agents, k, threshold=0.5):
    ratio = k / n_agents
    return "High" if ratio >= threshold else "Low"

def compute_win_table(experiment_results, metric, algorithms=None):
    if not algorithms:
        algorithms = [k for k in experiment_results.keys() if isinstance(k, (str)) and k != "base-case"]
    win_table = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "ties": 0, "losses": 0})
    )
    # Group results by unique configuration (excluding method)
    config_groups = defaultdict(list)


    fields = experiment_results[algorithms[0]].keys()
    for alg in algorithms: 
        for k in fields:
            for res in experiment_results[alg][k]:
                res["method"] = alg
                key = (res["n_agents"], res["k"])
                config_groups[key].append(res)

    lens_dict = defaultdict(int)
    seen_configs = []
    for (n_agents, k), group in config_groups.items():
        leader_class = classify_leader_ratio(n_agents, k)
        # for res in group:
        res = group[0]
        longest_edge = res["longest_edge"]
        density = classify_density(longest_edge, n_agents)

        config_label = f"{density} + {leader_class}"
        if (n_agents, k) not in seen_configs: 
            lens_dict[config_label] += 1
            seen_configs.append((n_agents, k))
            
        best_value = min(r[metric] for r in group)

        winner_methods = [
            r["method"] for r in group
            if abs(r[metric] - best_value) < 1e-6
        ]

        winner_set = set(winner_methods)

        if len(winner_set) == 1:
            # single winner
            winner = next(iter(winner_set))
            win_table[config_label][winner]["wins"] += 1
        else:
            # tie among multiple methods
            for method in winner_set:
                win_table[config_label][method]["ties"] += 1

        # losses for everyone else
        for alg in algorithms:
            if alg not in winner_set:
                win_table[config_label][alg]["losses"] += 1
        
        # winner_count = 0
        # winner_methods = []
        # for r in group:
        #     if abs(r[metric] - best_value) < 1e-6:  # tie tolerance
        #         winner_count += 1
        #         winner_methods.append(r["method"])
            
        # for winner_method in winner_methods:
        #     win_table[config_label][winner_method] += 1 / winner_count

    print(lens_dict)
    return win_table


##########################################################################################################
###############                                 NOT USED                                  ################
##########################################################################################################
# def load_per_agent_results(experiment_folder):
    # for root, dirs, files in walk_naturally_sorted(experiment_folder):  
    #     for file in files: 
    #         if file.endswith("avg_per_leader.txt"):
    #             file_path = Path(root) / file
    #             try: 
    #                 with open(file_path, newline='') as csvfile:
    #                     reader = csv.DictReader(csvfile)
    #                     if file_path.__str__().endswith("avg_per_leader.txt"):
    #                         file_paths = file_path.__str__().split("\\")
                            
    #                         n_agents = int(file_paths[-3].split("_")[-1])
    #                         k = int(file_paths[-2].split("_")[-1])
    #                         row = reader.__next__()
    #                         leader_stats = {
    #                             "leader_avg_max_dist" : float(row["leader_avg_max_dist"]),
    #                             "leader_avg_total_dist" : float(row["leader_avg_total_dist"]),
    #                             "leader_avg_avg_dist" : float(row["leader_avg_avg_dist"])
    #                             }
    #                         result_dict = [r for r in results_by_agents[n_agents] if r['k'] == k][0]

    #                         result_dict["leader_avg_max_dist"] = leader_stats["leader_avg_max_dist"]
    #                         result_dict["leader_avg_total_dist"] = leader_stats["leader_avg_total_dist"]
    #                         result_dict["leader_avg_avg_dist"] = leader_stats["leader_avg_avg_dist"]

    #             except Exception as e:
    #                 print(f"Error parsing {file_path}: {str(e)}")
        

def normalize_metrics(run):
    """Add all three normalization approaches to results"""
    run['norm_avg_total_dist'] = run['avg_total_dist'] / (run['map_size'] **2)
    run['norm_avg_avg_dist'] = run['avg_avg_dist'] / (run['map_size'] **2)
    run['norm_avg_max_dist'] = run['avg_max_dist'] / (run['map_size'] **2)
    # run['norm_area'] = run['total_dist'] / (run['map_size'] ** 2)
    # run['norm_sqrt'] = run['total_dist'] / (run['map_size'] ** 0.5)
    return run


