from sklearn.cluster import KMeans
import numpy as np
from sympy import false
from utils import *
import random, math
from algorithms import nearest_neighbor_mtsp
from rl_agents import *
from visualizations import visualize_visit
from ctde import CTDE_PPO

class Agent: 
    speed = 2.0 # unit/s
    comm_range = 0.5 # radius
    agents_initial_pos = []

    def __init__(self, id, pos, k, sim, is_leader=False):
        self.id = id
        self.k = k
        self.curr_pos = np.array(pos)
        self.initial_pos = np.array(pos)
        self.has_info = is_leader
        self.is_assistant = False
        self.is_leader = is_leader
        self.leader = None
        self.assistants = [] # moved here to make plotting logic easier... check plot_agent_performance
        self.path = []
        self.sim = sim
    
        self.info = set() if not is_leader else set([self.id])
        self.sensor_info = set() if not is_leader else set([self.id])
        
        self.total_distance_travelled = 0
        self.total_comm_visits = 0 # number of past visits

    def inform_sensor(self, other_agent):
        other_agent.sensor_info.update(self.info)
        self.info.update(other_agent.sensor_info)
        self.total_comm_visits += 1

    def inform(self, other_agent):
        other_agent.has_info = True
        other_agent.info.update(self.info)
        self.info.update(other_agent.info)

    def visit_sensor(self):
        self.info.update(self.sensor_info)


    # moves one time step by updating the position of the agent according to a fixed speed
    def move(self):
        if self.path:
            next_target = Agent.agents_initial_pos[self.path[0]]
            direction = next_target - self.curr_pos
            distance = euclidean_dist(next_target, self.curr_pos)
            if distance <= self.speed:
                self.curr_pos = next_target.copy()
            else:
                self.curr_pos += (direction / distance) * self.speed
            
            # save the distance travelled
            self.total_distance_travelled += min(distance, self.speed)


    # in case the communication range is to be taken into consideration
    def can_communicate(self, other_agent):
        return euclidean_dist(self.curr_pos, other_agent.curr_pos) < (self.comm_range)
            
    def near_sensor(self, other_agent):
        return euclidean_dist(self.curr_pos, other_agent.initial_pos) < (self.comm_range)
    
    def near_point(self, point):
        return euclidean_dist(self.curr_pos, point) < (self.comm_range / 2)
    
    def interact(self, other_uav): 
        return NotImplementedError
        
    def step(self, uavs, visualize=False): 
        if self.has_info and self.path:
            self.move()
        
            for other_uav in uavs:
                if self.near_sensor(other_uav):
                    visiting_id = None
                    if self.path and other_uav.id == self.path[0]:
                        self.path.pop(0)
                        visiting_id = self.id
                    if self.id == other_uav.id:
                        self.visit_sensor()
                    else: 
                        self.inform_sensor(other_uav)
                        if self.can_communicate(other_uav):
                            self.inform(other_uav)
                            if isinstance(self, Leader) and not isinstance(other_uav, Leader) \
                                    and not other_uav.is_assistant and len(self.path) >= 2: 
                                self.interact(other_uav)
                    if visualize and visiting_id is not None:
                        visualize_visit(self.sim, visiting_id)
            # if not leader_id: 
            #     leader_id = self.id
        if self.near_sensor(self): 
            self.visit_sensor()

class Leader(Agent): 
    STRATEGIES = {
        "normal-case": "normal_assign",
        "limited": "limit_assign",
        "toggle": "toggle_assign",
        "adaptive-limit": "limit_assign",
        "adaptive-freq": "toggle_assign",
        "random": "random_assign",
        "no-assistant": "no_assign", 
        "base-case": "no_assign"
    }

    def __init__(self, id, pos, k, sim, assigning_strategy="normal", assistants_limit=10, assigning_freq=1, rand=False):
        super().__init__(id, pos, k, sim, True)

        # startegy control variables
        self.assistants_limit = assistants_limit # limit assistants
        self.should_assign = True # toggle assistants
        self.toggle_count = 0
        self.assigning_frequency = assigning_freq
        self.rand = rand # random assignment

        if assigning_strategy != "rl":
            self.assigning_strategy = getattr(self, self.STRATEGIES[assigning_strategy])


    def toggle_should_assign(self):
        self.toggle_count += 1
        if self.toggle_count % self.assigning_frequency == 0:
            self.toggle_count = 0
            self.should_assign = True
        else: 
            self.should_assign = False


    # only assigns the visited agent if it is a leader
    def normal_assign(self, other_agent):
        self.assistants.append(other_agent)
        if not self.update_path():
            self.assistants.pop()
            return False
        else: 
            other_agent.is_assistant = True
            other_agent.leader = self
            return True
    

    
    def toggle_assign(self, other_agent):
        if self.should_assign:
            self.assistants.append(other_agent)
            if not self.update_path():
                self.assistants.pop()
            else: 
                other_agent.is_assistant = True
                other_agent.leader = self
        self.toggle_should_assign()


    def limit_assign(self, other_agent):
        if len(self.assistants) < self.assistants_limit: 
            self.assistants.append(other_agent)
            if not self.update_path():
                self.assistants.pop()
            else: 
                other_agent.is_assistant = True
                other_agent.leader = self
                    

    def random_assign(self, other_agent):
        if random.random() > 0.5:
            self.assistants.append(other_agent)
            if not self.update_path():
                self.assistants.pop()
            else: 
                other_agent.is_assistant = True
                other_agent.leader = self

    def no_assign(self, other_agent): 
        pass

    def interact(self, other_uav):
        # if self.is_assign_enabled(other_uav):
        self.assigning_strategy(other_uav)
    
    # multi-tsp implementation
    def update_path(self):
        if len(self.path) == 2: 
            self.assistants[-1].path = [self.path[0], self.assistants[-1].id]
            self.path = [self.id]
            return True
        
        if len(self.path) <= 2 or not self.assistants:
            return False
        
        waypoints = np.array([Agent.agents_initial_pos[p] for p in self.path[:-1]])

        # Kmeans
        num_clusters = 2  # Leader + new assistant
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(waypoints)

        clusters = {i: [] for i in range(num_clusters)}
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(self.path[i])

        self.path = nearest_neighbor_mtsp(self.curr_pos, clusters[0], Agent.agents_initial_pos)
        # self.path = clusters[0] 
        self.path.append(self.id)
        self.assistants[-1].path = nearest_neighbor_mtsp(self.curr_pos, clusters[1], Agent.agents_initial_pos)
        # self.assistants[-1].path = clusters[1]
        self.assistants[-1].path.append(self.assistants[-1].id) 

        return True
        # Genetaic Algorithm 
        # best_route = genetic_tsp(waypoints)
        # best_path = [self.path[i] for i in best_route] 
        
        # self.assistants[-1].path = best_path[len(best_path)//2:]
        # self.assistants[-1].path.append(self.assistants[-1].id) 

        # self.path = best_path[:len(best_path)//2]
        # self.path.append(self.id)

class IntelligentLeader(Leader): 
    agents = {
        "PPO": PPO
    }

    def __init__(self, id, pos, k, sim, agent: RLAgent):
        super().__init__(id, pos, k, sim, "rl")
        self.rl_agent = agent
        self.waypoints = self.sim.n_agents
        self.fair_limit = max(1, math.floor((self.waypoints - self.k) / self.k))
        self.L = max(1, min(self.fair_limit, math.ceil(math.log2(self.waypoints)) - 1))
        self.target_split_ration = 1 / (1 + self.L)
        self.last_snapshot = dict(
            remaining_waypoints = self.waypoints,
            remaining_path=self._path_dist(),
            # assistants=0
        )
        self.training = self.sim.training
        self.episode_reward = 0
        self.reward_history = []
        

    def _path_dist(self):
        total_dist = 0
        curr = self.curr_pos
        for i, waypoint in enumerate(self.path):
            next = Agent.agents_initial_pos[waypoint]
            total_dist += euclidean_dist(curr, next)
            curr = next

        return total_dist
    
    def _len_path(self):
        return len(self.path)
    
    def _dist_to_inital_pos(self):
        return euclidean_dist(self.curr_pos, self.initial_pos)

    def _number_of_assistants(self):
        return len(self.assistants)

    def _dist_to_next(self):
        if not self.path: return 0

        next = self.path[0]
        return euclidean_dist(self.curr_pos, Agent.agents_initial_pos[next])

    def _dist_to_second_next(self):
        if not self.path or len(self.path) <= 1: return 0

        next = self.path[0]
        second_next = self.path[1]

        return euclidean_dist(Agent.agents_initial_pos[next], Agent.agents_initial_pos[second_next])

    # # FIRST STATE REPRESENTATION 
    # def get_observation(self):
    #     return ( 
    #         self.total_distance_travelled, 
    #         self._path_dist(),
    #         self._len_path(),
    #         self._dist_to_inital_pos(),
    #         self.k,
    #         self.n_agents,
    #         self._number_of_assistants(),
    #         self.total_comm_visits,
    #         self._dist_to_next(),
    #     )
    
    # # SECOND STATE REPRESENTATION
    def get_observation(self):
        return ( 
            self.total_distance_travelled, 
            self._path_dist() / self.waypoints,
            self._len_path(),
            self._len_path() / self.waypoints,
            (self.waypoints -  self._len_path())/ self.waypoints, 
            
            self.k,
            self.waypoints,
            self.k / self.waypoints,
            self.waypoints / 100,
            
            self._number_of_assistants() / self.L, # normalized
            self._number_of_assistants(),
            
            self._path_dist(),
            self._dist_to_inital_pos(),
            self._dist_to_next(),
            self._dist_to_second_next(),
            # self.total_comm_visits,
            
        )

    def get_action(self, state, determenistic=False):
        return self.rl_agent.policy(state, determenistic=determenistic)
    
    def take_assignment_snapshot(self):
        return dict(
            remaining_waypoints=self._len_path(),
            remaining_path=self._path_dist(),
            assistants=self._number_of_assistants()
        )
    
    def compute_delta_reward(self, pre, post):
        R_assigned = max(0, pre["remaining_waypoints"] - post["remaining_waypoints"])
        assigned_ratio = R_assigned / max(1, self.waypoints)

        share_penalty = abs(assigned_ratio - self.target_split_ration) / self.target_split_ration
        share_reward = 1.0 - share_penalty

        over_penalty = max(0, post["assistants"] - self.fair_limit) * 0.5

        reward = (
            2 * share_reward
            - over_penalty
        )
        self.episode_reward += reward
        self.reward_history.append((share_penalty, over_penalty))
        # print(f"[DEBUG] Leader {self.id} time: {self.sim.steps} Assignment Details: R_assigned: {R_assigned}, remaining_waypoints_pre: {pre['remaining_waypoints']}, remaining_waypoints_post: {post['remaining_waypoints']}")
        # print(f"[REWARD] Leader {self.id} assigned_ratio: {assigned_ratio:.3f}, target: {self.target_split_ration:.3f}, share_reward: {share_reward:.3f}, over_penalty: {over_penalty:.3f}, total_reward: {reward:.3f}")

        # reward *= 10
        # reward = max(-0.5, min(0.5, reward))
        return reward

    def not_assign_reward(self):
        R_assigned = self._len_path() // 2
        assigned_ratio = R_assigned / max(1, self.waypoints)
        share_reward = abs(assigned_ratio - self.target_split_ration) / self.target_split_ration

        # over_reward = max(0, self._number_of_assistants() - self.fair_limit) * 0.5
        
        reward = ( share_reward
            # + 0.5 * over_reward
        )
        # reward = 0
        
        over_reward = 0
        
        self.episode_reward += reward
        self.reward_history.append((share_reward, over_reward))
        
        return reward

    def interact(self, other_uav):
        state = self.get_observation()

        global_state = None
        full_state = state
        if isinstance(self.rl_agent, CTDE_PPO):
            global_state = self.sim.get_global_obs()
            full_state = (state, global_state)
        
        pre = self.take_assignment_snapshot()
        
        assigned = False
        # print("---- Leader", self.id, "deciding assignment ----")
        action = self.get_action(full_state, (not self.training))
        if self.rl_agent.is_assign(action):
            assigned = self.normal_assign(other_uav)
            
        
        # print("assigned:", assigned)

        # action, log_prob, v = action
        post = self.take_assignment_snapshot()
    
        if assigned:        
            reward = self.compute_delta_reward(pre, post) 
            # action = (action, log_prob, v)
        else: 
            # a0 = torch.zeros_like(action)  # 0
            # recompute log_prob for the corrected action
            # print("---- Leader", self.id, "could not assign ----")
            # with torch.no_grad():
            #     # state_t = torch.tensor(full_state, dtype=torch.float32, device=self.rl_agent.device)
            #     _, logprob0, _ = self.get_action(full_state, (not self.training))
            #     action = (a0.detach(), logprob0.detach(), v.detach())
            
            reward = self.not_assign_reward()
            
        # done = (
        #     self._len_path() <= 2
        #     or self.sim.is_complete()
        #     # or self._number_of_assistants() >= self.L # ? keep this condition
        # )

        done = False # ? 

        # if done: print(f"[DONE] Leader {self.id} at step {self.sim.steps}, assistants={a_p}, remaining={self._len_path()}")

        next_state = self.get_observation()
        full_state = next_state
        if isinstance(self.rl_agent, CTDE_PPO):
            next_global_state = self.sim.get_global_obs()
            full_state = (next_state, next_global_state)

        with torch.no_grad():
            _, _, next_value = self.rl_agent.policy(full_state)
        
        if global_state is not None:
            self.rl_agent.observe(state, global_state, action, reward, done, next_value)
        else: 
            self.rl_agent.observe(state, action, reward, done, next_value)
            
    