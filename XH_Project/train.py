from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from PPO import ppo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--layers", "-l", type=int, default=2)
    parser.add_argument("--pi_lr", "-pi", type=float, default=3e-4)
    parser.add_argument("--v_lr", "-v", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.97)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_name", type=str, default="model/model10.pt")
    parser.add_argument(
        "--scenario",
        default="simple_tag.py",
        help="Path of the scenario Python script.",
    )
    args = parser.parse_args()

    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        shared_viewer=False,
    )

    agent = ppo(
        lambda: env,
        hid=args.hid,
        layers=args.layers,
        seed=args.seed,
        lam=args.lam,
        steps_per_epoch=args.steps,
        pi_lr=args.pi_lr,
        v_lr=args.v_lr,
    )
    agent.train(epochs=args.epochs, save_name=args.save_name)
