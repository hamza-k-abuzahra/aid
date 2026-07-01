import os, datetime
from info_dissemination import InfoDiss
from report import log_single_report, log_averaged_report, average_reports
from visualizations import plot_initial_configuration, plot_agent_performance
from rl_agents import RLAgent
from utils import parse_dumas_extended

def experiment_run(experiment_name, assigning_strategy, agent: RLAgent):
    seeds = [121212, 131313, 141414, 151515, 161616, 171717, 181818, 191919, 202020, 111111]
            #  101010, 303030, 404040, 505050, 606060, 707070, 808080, 909090, 112233, 223344, 
            #  334455, 445566, 556677, 667788, 778899, 889911, 111222, 222333, 3333444, 4444555]

    # git_branch = git.Repo(os.getcwd()).active_branch.name
        
    experiment_folder_name = "experiment_" + experiment_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(experiment_folder_name, exist_ok=True)
    os.chdir(experiment_folder_name)
    

    ## CHANGING n_ledaer, FIXING map_size & n_agents 
    # n_leaders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # map_size = 100
    # n_agents = 20
    # for k in n_leaders:    
    #     run_name = f"k_{k}"
    #     os.makedirs(run_name, exist_ok=True)
    #     run_id = str(k)
    #     # different seed to test on different configurations
    #     for seed in seeds:
    #         sim_i = InfoDiss(k=k, n_agents=n_agents, map_size=map_size, seed=seed, run_id=run_id)
    #         sim_i.run_simulation()
    #         sim_i.log_report()
    #         sim_i.plot_settings()
    #         os.chdir("..")

    ## CHANGING map_size, n_agents
    # n_agents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # for n_agent in n_agents:
    #     run_name = f"agent_{n_agent}"
    #     os.makedirs(run_name, exist_ok=True)
    #     os.chdir(run_name)
    #     map_sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #     n_leader = 5
    #     for map_size in map_sizes:    
    #         run_name = f"map_{map_size}"
    #         os.makedirs(run_name, exist_ok=True)
    #         # different seed to test on different configurations
    #         reports = []
    #         for seed in seeds:
    #             sim_i = InfoDiss(k=n_leader, n_agents=n_agent, map_size=map_size, seed=seed, run_id=run_name)
    #             sim_i.run_simulation()
                    
    #             report_i = sim_i.evaluate()
    #             log_single_report(report_i, sim_i.folder_name)
    #             sim_i.plot_settings()
    #             reports.append(report_i)
    #             os.chdir("..")
    #         final_report = average_reports(reports)
    #         log_averaged_report(final_report, run_name)
            
    #     os.chdir("..")

    ## Set map_size and vary n_agents and k
    map_size = 100
    
    n_agents = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   
    # n_agents = [100]
    for n in n_agents:
        run_name = f"agent_{n}"
        os.makedirs(run_name, exist_ok=True)
        os.chdir(run_name)
        # n_leaders = [i for i in range(1, (step * 10) + 1, step)]
        n_leaders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # n_leaders = [4, 5, 6, 7, 8, 9, 10]
        for k in n_leaders:
            k = int(k*n/10)
            run_name = f"k_{k}"
            os.makedirs(run_name, exist_ok=True)
            # different seed to test on different configurations
            reports = []
            InfoDiss.i = 0
            for seed in seeds:

                sim_i = InfoDiss(k=k, n_agents=n, map_size=map_size,assigning_strategy=assigning_strategy, seed=seed, run_id=run_name, agent=agent)
                
                # sim_i.base_case()
                sim_i.run_simulation()
                
                report_i = sim_i.evaluate()
                log_single_report(report_i, sim_i.folder_name)
                plot_initial_configuration(sim_i.uavs, sim_i.folder_name, map_size)
                plot_agent_performance(sim_i.uavs, sim_i.folder_name)
                reports.append(report_i)
                os.chdir("..")
            final_report = average_reports(reports)
            log_averaged_report(final_report, run_name)
            
        os.chdir("..")


def bench_experiment(experiment_name, data_path, assigning_strategy, agent: RLAgent, num_instances: int = 10, n: int = 29):
    
    seeds = [121212, 131313, 141414, 151515, 161616, 171717, 181818, 191919, 202020, 111111]
    # instances = parse_tsplib_file(data_path, num_instances)
    # coords, total_dist = parse_dumas_extended(data_path)
   
    # 15 / 291
    # instances = [{
    #     "id": "15", 
    #     "coords": 
    #     [
    #   [ -0.0000000400893815       , 0.0000000358808126],
    #   [-28.8732862244731230       ,-0.0000008724121069],
    #   [-79.2915791686897506      , 21.4033307581457670],
    #   [-14.6577381710829471      , 43.3895496964974043],
    #   [-64.7472605264735108      ,-21.8981713360336698],
    #   [-29.0584693142401171      , 43.2167287683090606],
    #   [-72.0785319657452987       ,-0.1815834632498404],
    #   [-36.0366489745023770      , 21.6135482886620949],
    #   [-50.4808382862985496       ,-7.3744722432402208],
    #   [-50.5859026832315024      , 21.5881966132975371],
    #   [ -0.1358203773809326      , 28.7292896751977480],
    #   [-65.0865638413727368      , 36.0624693073746769],
    #   [-21.4983260706612533       ,-7.3194159498090388],
    #   [-57.5687244704708050      , 43.2505562436354225],
    #   [-43.0700258454450875      ,-14.5548396888330487],
    #     ],
    #     "bench_total_dist": 291
    # }]

    # 42 / 699
    # instances = [{
    #     "id": "42", 
    #     "coords": 
    #     [
    #         [170.0,  85.0],
    #         [166.0,  88.0],
    #         [133.0,  73.0],
    #         [140.0,  70.0],
    #         [142.0,  55.0],
    #         [126.0,  53.0],
    #         [125.0,  60.0],
    #         [119.0,  68.0],
    #         [117.0,  74.0],
    #         [99.0,  83.0],
    #         [73.0,  79.0],
    #         [72.0,  91.0],
    #         [37.0,  94.0],
    #         [6.0, 106.0],
    #         [3.0,  97.0],
    #         [21.0,  82.0],
    #         [33.0,  67.0],
    #         [4.0,  66.0],
    #         [3.0,  42.0],
    #         [27.0,  33.0],
    #         [52.0,  41.0],
    #         [57.0,  59.0],
    #         [58.0,  66.0],
    #         [88.0,  65.0],
    #         [99.0,  67.0],
    #         [95.0,  55.0],
    #         [89.0,  55.0],
    #         [83.0,  38.0],
    #         [85.0,  25.0],
    #         [104.0,  35.0],
    #         [112.0,  37.0],
    #         [112.0,  24.0],
    #         [113.0,  13.0],
    #         [125.0,  30.0],
    #         [135.0,  32.0],
    #         [147.0,  18.0],
    #         [147.5,  36.0],
    #         [154.5,  45.0],
    #         [157.0,  54.0],
    #         [158.0,  61.0],
    #         [172.0,  82.0],
    #         [174.0,  87.0],
    #     ],
    #     "bench_total_dist": 699
    # }]

    # 26 / 938
    # instances = [{
    #     "id": "26", 
    #     "coords": 
    #     [
    #         [  1.84969232, 115.68564972],
    #         [-71.82443118 , 77.4917583 ],
    #         [-54.05714947 , 41.34482334],
    #         [-93.80734163 , 29.09695885],
    #         [-89.44415254 , 18.65687811],
    #         [-97.87614099 , 19.10028839],
    #         [-76.68970366 ,-13.60226952],
    #         [-67.98908964 ,-38.38531671],
    #         [-42.77892607 ,-11.43309725],
    #         [-28.52982415  , 5.44298818],
    #         [ -8.41583589  , 6.55856427],
    #         [  5.42587498 , 17.60009637],
    #         [ -5.79210036 , 17.21348357],
    #         [-27.50596729 , 25.49128414],
    #         [-18.1665917  , 37.55051203],
    #         [-14.07592608 ,-35.06433176],
    #         [ 29.5620625  ,-40.88633715],
    #         [ 31.75004841 ,-63.02215107],
    #         [ -0.22170933 ,-56.77898034],
    #         [  3.64580482 ,-69.46094235],
    #         [ 52.78827963 ,-22.3197795 ],
    #         [ 79.04085741 ,-20.71456361],
    #         [129.35791041 ,-18.5313509 ],
    #         [180.61389192 ,-12.85993848],
    #         [103.26216211 , 39.5011235 ],
    #         [ 79.87830546 ,-47.67535012],
    #     ],
    #     "bench_total_dist": 937
    # }]

    # 17 / 2085
    # instances = [{
    #     "id": "17", 
    #     "coords": 
    #     [
    #         [-109.90785539,  161.85764393],
    #         [ 282.51691877, -352.29784303],
    #         [ 115.695011,     18.96063804],
    #         [-118.14202464,   91.40662593],
    #         [ 144.11655775, -161.76226034],
    #         [   4.26432175,   63.50793269],
    #         [ -52.29519527,  111.4991939 ],
    #         [ -17.64147513,   95.06792431],
    #         [-199.28608877,  -81.35564106],
    #         [ 322.48354945, -100.40019573],
    #         [ 155.64486967,  -88.65376498],
    #         [-281.50430508,  -90.26926133],
    #         [ -79.18858738,   81.38486683],
    #         [  78.2020816,    79.22745993],
    #         [ 134.81506873,   51.97563371],
    #         [-354.83857812,   46.23270872],
    #         [ -24.93426895,   73.61833848],
    #     ],
    #     "bench_total_dist": 2085
    # }]

    # 5 / 19
    instances = [{
        "id": "5", 
        "coords": 
        [
            [-1.1182935,   1.4807061 ],
            [ 1.63490376, -1.16217042],
            [ 2.02656047,  3.10338116],
            [-3.04682178,  1.0789136 ],
            [ 0.50365105, -4.50083043],
        ],
        "bench_total_dist": 19
    }]

    experiment_folder_name = "experiment_" + experiment_name + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(experiment_folder_name, exist_ok=True)
    os.chdir(experiment_folder_name)
    
    ## Set map_size and vary n_agents and k
    map_size = 200
    
    
    for instance in instances:
        id = instance["id"]
        positions = instance["coords"]
        bench_total_dist = instance["bench_total_dist"]
    
        run_name = f"agent_{id}"
        os.makedirs(run_name, exist_ok=True)
        os.chdir(run_name)
        with open('bench_total_dist.txt', 'w') as file:
            file.write(str(bench_total_dist))
        # n_leaders = range(1, n, 3)
        n_leaders = [1]
        for k in n_leaders:
            # k = int(k*n/10)
            run_name = f"k_{k}"
            os.makedirs(run_name, exist_ok=True)
            # different seed to test on different configurations
            reports = []
            InfoDiss.i = 0
            for seed in seeds:

                sim_i = InfoDiss(positions=positions, k=k, n_agents=n, map_size=map_size,assigning_strategy=assigning_strategy, seed=seed, run_id=run_name, agent=agent)
                
                # sim_i.base_case()
                sim_i.run_simulation()
                
                report_i = sim_i.evaluate()
                log_single_report(report_i, sim_i.folder_name)
                plot_initial_configuration(sim_i.uavs, sim_i.folder_name, map_size)
                return
                plot_agent_performance(sim_i.uavs, sim_i.folder_name)
                reports.append(report_i)
                os.chdir("..")
            final_report = average_reports(reports)
            log_averaged_report(final_report, run_name)
            
        os.chdir("..")
