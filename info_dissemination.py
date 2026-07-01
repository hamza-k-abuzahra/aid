import numpy as np
import random, os, datetime, math

from agent import Agent, Leader, IntelligentLeader
from rl_agents import RLAgent
from algorithms import nearest_neighbor_tsp, longest_edge_MST, compute_MST, minimum_enclosing_circle, kmeans_mtsp
from utils import *
from visualizations import visualize_visit

class InfoDiss:
    i = 0

    # initialize locations of agents, assign the leaders, set initial tsp for leader agents
    def __init__(self, n_agents=20, k=3, map_size=100, assigning_strategy="normal", visualize=False, seed=121212, run_id=None, agent: RLAgent = None, train=False, positions: list = None, leader_uavs: list =  None): # map size is set to be very large to simulate unbounded environment
        Agent.agents_initial_pos = []
        self.k = k
        self.n_agents = n_agents
        self.map_size = map_size
        self.steps = 0
        self.visualize = visualize
        random.seed(seed)

        self.run_id = run_id
        # for evaluation purposes
        self.total_assistants = 0

        if leader_uavs is None:
            self.leader_uavs = random.sample(range(n_agents), k)  
        else: 
            self.leader_uavs = leader_uavs

        # initialize agent objects 
        if positions is None: 
            positions = [np.array([random.uniform(0, map_size), random.uniform(0, map_size)]) for _ in range(n_agents)]
        mst = compute_MST(np.array(positions))
        self.longest_edge = longest_edge_MST(mst)
        self.is_dense = self.longest_edge < self.n_agents * Agent.comm_range

        # strategy defining variables
        # choices=["normal-case", "limited", "toggle", "adaptive-limit", "adaptive-freq", "random", "no-assistant", "rl"],
        limit = n_agents
        freq = 1
        rand = False
        enable_assistants = True
        if assigning_strategy == "limited": 
            limit = self.limit_assistant()
        elif assigning_strategy == "toggle": 
            freq = self.toggle_assignment()
        elif assigning_strategy == "adaptive-limit": 
            limit = self.adaptive_limit()
        elif assigning_strategy == "adaptive-freq": 
            freq = self.adaptive_toggle()
        elif assigning_strategy == "random": 
            rand = True

        self.assistant_limit = limit
        self.assigning_frequency = freq
        self.training = train

        self.uavs = [Agent(i, positions[i], k, self, is_leader=i in self.leader_uavs)
                            if i not in self.leader_uavs else 
                    Leader(i, positions[i], k, self, assigning_strategy, limit, freq, rand)  
                            if assigning_strategy != "rl" else 
                    IntelligentLeader(i, positions[i], k, self, agent)
                            for i in range(n_agents)]
        
        for uav in self.uavs:
            Agent.agents_initial_pos.append(uav.initial_pos)

        if train: self.visualize = False
        else: self._create_folder()

    def limit_assistant(self):
        limit = math.floor((self.n_agents - self.k) / self.k)
        return limit 
    
    def adaptive_limit(self):
        edge_clamped = max(10, min(100, self.longest_edge))
        density_score = (edge_clamped - 10) / (100 - 10)  # sparse → 1.0
        non_leaders = self.n_agents - self.k
        base_limit = non_leaders / self.k
        scaled_limit = base_limit * (0.5 + 0.5 * density_score)

        limit = max(1, min(round(scaled_limit), base_limit))
        return limit

    def toggle_assignment(self):
        freq = 2
        return freq 
    
    def adaptive_toggle(self):
        leader_ratio = self.k / self.n_agents
        high_leader_ratio = leader_ratio >= 0.5
        # # Apply your logic
        # if high_leader_ratio: 
        #     freq = 3
        # else: 
        #     if self.is_dense: 
        #         freq = 2
        #     else: 
        #         freq = 1

        if not high_leader_ratio:
            freq = 1
        elif self.is_dense:
            freq = 2
        else: 
            freq = 3

        return freq

    def _create_folder(self):
        # folder creation to save frames for visualization purposes
        if self.run_id:
            os.chdir(self.run_id)
            self.folder_name = f"simulation_run_{InfoDiss.i}"
            InfoDiss.i += 1
        else:
            self.folder_name = "simulation_run_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.folder_name, exist_ok=True)
        os.chdir(self.folder_name)
        for leader in self.leader_uavs:
            try:
                os.makedirs(f"leader_{leader}")
            except Exception as e:
                print(os.getcwd())
                print(f"Error parsing leader_{leader}: {str(e)}")
        os.chdir("..")


    def _initialize_leader_paths(self, center_indx=None):
        if center_indx is not None: paths = {uav_id: [center_indx] for uav_id in self.leader_uavs}
        else: paths = {uav_id: self._tsp(uav_id, list(set(range(len(self.uavs))) - {uav_id})) for uav_id in self.leader_uavs}

        for uav_id, path in paths.items():
            self.uavs[uav_id].path = path
            self.uavs[uav_id].n_agents = self.n_agents
            # if self.visualize: visualize_visit(self, uav_id)
        
    # initial tsp 
    def _tsp(self, start, nodes):
        positions = [uav.initial_pos for uav in self.uavs]
        return nearest_neighbor_tsp(start, nodes, positions)
        # return genetic_algorithm_tsp(start, nodes, positions)


    def step(self):
        for uav in self.uavs:
            uav.step(self.uavs, self.visualize)

        self.steps += 1               
        return all([len(uav.info) >= self.k for uav in self.uavs])


    def run_simulation(self):
        self._initialize_leader_paths()

        while True:
            self.step()
            if all([len(uav.info) >= self.k for uav in self.uavs]):
                break
 
    def is_complete(self):
        return all([len(uav.info) >= self.k for uav in self.uavs])
       
    def _step_leaders(self, leaders):
        for uav in leaders:
            if uav.path:
                uav.move()
                if uav.near_point(self.center): 
                    uav.path.pop(0)
                    if self.visualize:
                        visualize_visit(self, uav.id)

        self.steps += 1

 
    def get_global_obs(self):
        leaders = [uav for uav in self.uavs if uav.is_leader]

        ### Initial global state
        global_obs = []
        for ag in leaders: 
            obs = [
                ag._len_path(),                       # remaining waypoints
                ag._path_dist(),                      # remaining path dist
                ag._number_of_assistants(),           # number of assistants
                ag.L,                                 # fair limit
                ag._dist_to_next(),                   # next waypoint dist
                ag._dist_to_second_next(),            # 2nd next waypoint dist
                self.k,                               # number of leaders in scenario
                # ag.total_comm_visits,                 # communication count
                ag.waypoints,                         # initial load N0
            ]
            global_obs.append(obs)

        return np.array(global_obs, dtype=np.float32)

    def base_case(self):
        leaders = [uav for uav in self.uavs if uav.is_leader]
        targets = [uav for uav in self.uavs if not uav.is_leader]
        # get initial positions of only leaders
        if self.k == 1: 
            center = leaders[0].curr_pos
        else: 
            leaders_pos = np.array([uav.initial_pos for uav in self.uavs if uav.is_leader], dtype=np.float32)
            center, radius = minimum_enclosing_circle(leaders_pos)
        
        self.center = center
        Agent.agents_initial_pos.append(np.array(center))


        # add the center to all leaders path
        self._initialize_leader_paths(self.n_agents)

        # loop until all leaders reach the center point
        while True:
            # step only leaders
            self._step_leaders(leaders)
            if all([len(uav.path) == 0 for uav in leaders]):
                break
        
        # exchanging information amongs leaders
        for leader in leaders: 
            for other_leader in leaders:
                if leader.can_communicate(other_leader):
                    leader.inform(other_leader)

        # mtsp for the rest of the agents
        routes = kmeans_mtsp(leaders, targets, Agent.agents_initial_pos)
        for leader, path in zip(leaders, routes):
            leader.path = path

        while True:
            self.step()
            if all([len(uav.info) == self.k for uav in self.uavs]):
                break


    def evaluate(self):
        # overall stats
        report = dict()
        # per agent lists
        distances = list()
        leader_distances = list()
        is_leader_list = list()
        assistants_count = list()
        communication_steps = list()
        leader_comm_steps = list()
        
        for uav in self.uavs:
            distances.append(uav.total_distance_travelled)
            if uav.is_leader: 
                leader_distances.append(uav.total_distance_travelled)
                leader_comm_steps.append(uav.total_comm_visits)

            is_leader_list.append(uav.is_leader)
            assistants_count.append(len(uav.assistants))
            communication_steps.append(uav.total_comm_visits)

        self.total_assistants = sum(assistants_count)
        
        total_dist = sum(distances)
        max_dist = max(distances)
        avg_dist = total_dist / self.n_agents

        total_comm = sum(communication_steps)
        max_comm = max(communication_steps)
        avg_comm = total_comm / self.n_agents

        # leader only stats
        leader_total_dist = sum(leader_distances)
        leader_max_dist = max(leader_distances)
        leader_avg_dist = leader_total_dist / self.k

        leader_total_comm = sum(leader_comm_steps)
        leader_max_comm = max(leader_comm_steps)
        leader_avg_comm = total_comm / self.k

        report["total_dist"] = total_dist
        report["max_dist"] = max_dist
        report["avg_dist"] = avg_dist
        report["total_comm"] = total_comm
        report["max_comm"] = max_comm
        report["avg_comm"] = avg_comm

        report["leader_total_dist"] = leader_total_dist
        report["leader_max_dist"] = leader_max_dist
        report["leader_avg_dist"] = leader_avg_dist
        report["leader_total_comm"] = leader_total_comm
        report["leader_max_comm"] = leader_max_comm
        report["leader_avg_comm"] = leader_avg_comm
        
        report["assistant_limit"] = self.assistant_limit
        report["assigning_frequency"] = self.assigning_frequency

        report["total_assistants"] = self.total_assistants
        report["map_size"] = self.map_size
        report["k"] = self.k
        report["n_agents"] = self.n_agents  
        report["steps"] = self.steps
        report["longest_edge"] = self.longest_edge
        # report["knn_density"] = self.density_nn
        # report["voronoi_density"] = self.density_v

        # per agent list stats
        report["distances"] = distances
        report["is_leader_list"] = is_leader_list
        report["assistants_count"] = assistants_count
        report["comm_visits_count"] = communication_steps


        return report


