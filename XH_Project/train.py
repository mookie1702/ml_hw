from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios
from PPO import ppo

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # 隐藏层大小
    parser.add_argument("--hid", type=int, default=64)
    # 神经网络层数
    parser.add_argument("--layers", type=int, default=2)
    # 策略网络的学习率
    parser.add_argument("--pi_lr", type=float, default=3e-4)
    # 价值网络的学习率
    parser.add_argument("--v_lr", type=float, default=1e-3)
    # PPO 算法中的 lambda 参数
    parser.add_argument("--lam", type=float, default=0.97)
    # 折扣因子大小
    parser.add_argument("--gamma", type=float, default=0.99)
    # 随机数种子
    parser.add_argument("--seed", type=int, default=0)
    # 每个训练回合中的步数
    parser.add_argument("--steps", type=int, default=4000)
    # 训练回合数
    parser.add_argument("--epochs", type=int, default=50)
    # 模型保存的路径和文件名
    parser.add_argument("--save_name", type=str, default="model/model10.pt")
    args = parser.parse_args()

    # 创建并初始化环境
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    # TODO: observation 函数
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        done_callback=scenario.is_done,
        shared_viewer=False,
    )

    # 设置 PPO 算法参数
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
