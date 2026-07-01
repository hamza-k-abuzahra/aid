import argparse, torch
from info_dissemination import InfoDiss
from experiments import experiment_run, bench_experiment
from visualizations import animate_visit, plot_agent_performance, plot_initial_configuration, plot_comparative_metrics, plot_single_result, plot_comparative_algorithms, plot_graph_comparative_algorithms
from report import log_single_report
from rl_agents import PPO
from ctde import CTDE_PPO

def run_simulation_with_args(args):
    if args.mode == "custom": 
        if args.assigning_strategy == "rl": 
            # Define environment and RL agent
            state_dim = 15   # based on your get_observation()
            action_dim = 2  # assign or not-assign
            
            agent = PPO(state_dim, action_dim, device="cpu")
            if args.rl_agent: 
                agent.load(args.rl_agent)
        else: 
            agent = None
        
        sim = InfoDiss(
            k=args.k,
            n_agents=args.n_agents,
            map_size=args.map_size,
            seed=args.seed,
            visualize=args.visualize,
            assigning_strategy=args.assigning_strategy,
            agent=agent,
            positions=args.positions,
            leader_uavs=args.leader_uavs
        )
        plot_initial_configuration(sim.uavs, sim.folder_name, sim.map_size)

        # sim._initialize_leader_paths()

        # sim.base_case()
        sim.run_simulation()
        # animate_visit(sim, visiting_leader_id=sim.leader_uavs[1], save_as="test2.gif")

        # plot_agent_performance(sim.uavs, sim.folder_name)
        print("done")
        print("producing report...")
        report_i = sim.evaluate()
        log_single_report(report_i, sim.folder_name)

    elif args.mode == "experiment":
        if args.experiment_name is None:
            raise ValueError("experiment_name is required if mode is 'experiment'")
        if args.assigning_strategy == "rl": 
            state_dim = 15   # based on your get_observation()
            global_dim = 10
            action_dim = 2  # assign or not-assign
            
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # agent = PPO(state_dim, device=device)
            agent = CTDE_PPO(state_dim, global_dim, device=device)
            if args.rl_agent: 
                agent.load(args.rl_agent)
        else: 
            agent = None
    
        experiment_run(args.experiment_name, args.assigning_strategy, agent)
    elif args.mode == "bench": 
        if args.experiment_name is None:
            raise ValueError("experiment_name is required if mode is 'experiment'")
        if args.assigning_strategy == "rl": 
            state_dim = 15   # based on your get_observation()
            global_dim = 10
            action_dim = 2  # assign or not-assign
            
            device = "cuda" if torch.cuda.is_available() else "cpu"

            agent = PPO(state_dim, device=device)
            # agent = CTDE_PPO(state_dim, global_dim, device=device)
            if args.rl_agent: 
                agent.load(args.rl_agent)
        else: 
            agent = None
    
        bench_experiment(args.experiment_name, args.data_path, args.assigning_strategy, agent, n=args.n_agents)
    else:
        if args.assigning_strategy == "rl": 
            # Define environment and RL agent
            state_dim = 15   # based on your get_observation()
            action_dim = 2  # assign or not-assign
            
            agent = PPO(state_dim, action_dim, device="cpu")
            if args.rl_agent: 
                agent.load(args.rl_agent)
        else: 
            agent = None
        
        sim = InfoDiss(
            k=args.k,
            n_agents=args.n_agents,
            map_size=args.map_size,
            seed=args.seed,
            visualize=args.visualize,
            assigning_strategy=args.assigning_strategy,
            agent=agent, 
            positions=args.positions,
            leader_uavs=args.leader_uavs   
        )
        plot_initial_configuration(sim.uavs, sim.folder_name, sim.map_size)

        # sim._initialize_leader_paths()

        # sim.base_case()
        sim.run_simulation()
        # animate_visit(sim, visiting_leader_id=sim.leader_uavs[1], save_as="test2.gif")

        # plot_agent_performance(sim.uavs, sim.folder_name)
        print("done")
        print("producing report...")
        report_i = sim.evaluate()
        log_single_report(report_i, sim.folder_name)


def main():
    # parser = argparse.ArgumentParser(description="Run Information Dissemination Simulation")

    # parser.add_argument("--assigning_strategy", type=str, required=True, 
    #                     choices=["base-case", "normal-case", "limited", "toggle", "adaptive-limit", "adaptive-freq", "random", "no-assistant", "rl"],
    #                     help="Assistant assignment strategy")
    # parser.add_argument("--mode", type=str, default="run", choices=["run", "experiment"], required=True,
    #                     help="Mode: 'run' for one-time simulation, 'experiment' for batch runs")
    # parser.add_argument("--n_agents", type=int, help="Number of agents in the simulation")
    # parser.add_argument("--k", type=int, help="Number of leaders in the simulation")
    # parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    
    # parser.add_argument("--rl_agent", type=str, default=None, 
    #                     help="RL agent to use if assigning_strategy == 'rl' (e.g., 'ppo', 'dqn')", 
    #                     choices=["PPO"])
    
    # parser.add_argument("--experiment_name", type=str, default=None, 
    #                     help="Name of the experiment (required if mode=experiment)")

    # parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    # parser.add_argument("--map_size", type=int, default=100, help="Size of the simulation map (default: 100)")

    # args = parser.parse_args()


    # PROGRAMATICALLY RUN
    # # Single run
    # args = argparse.Namespace(
    #     assigning_strategy="normal-case",
    #     mode="run",
    #     n_agents=8,
    #     k=2,
    #     seed=141414,
    #     rl_agent=None,
    #     experiment_name=None,
    #     visualize=True, 
    #     map_size=100
    # )

    # # Experiment
    # args = argparse.Namespace(
    #     assigning_strategy="adaptive-freq",
    #     mode="experiment",
    #     n_agents=20,
    #     k=None,
    #     seed=None,
    #     # rl_agent="test93.pt",
    #     rl_agent=None,
    #     experiment_name="adaptive-freq_paper",
    #     visualize=False, 
    #     map_size=100,
    #     data_path=None,
    # )

    # # benchmark
    # args = argparse.Namespace(
    #     assigning_strategy="no-assistant",
    #     mode="bench",
    #     n_agents=5,
    #     k=None,
    #     seed=None,
    #     rl_agent=None,
    #     # rl_agent="models/specialized/test31.pt",
    #     # rl_agent="models/curriculum learning/test61.pt",
    #     experiment_name="no_assistant_bench",
    #     visualize=True, 
    #     map_size=100,
    #     data_path="dumas/n20w120.001.txt",
    # )
    
    # run with specific initial positions
    args = argparse.Namespace(
        assigning_strategy="normal-case",
        mode="custom",
        n_agents=3,
        k=1,
        seed=None,
        rl_agent=None,
        # rl_agent="models/specialized/test31.pt",
        # rl_agent="models/curriculum learning/test61.pt",
        experiment_name="gazebo_5",
        visualize=False, 
        map_size=100,
        positions=[
            [ 49.78147354, -20.18210742],
            [-44.0748303 , -14.32444946],
            [41.112335 , 12.8400851],
        ], 
        leader_uavs=[1]
    )
    run_simulation_with_args(args)

if __name__ == "__main__":
    main()
