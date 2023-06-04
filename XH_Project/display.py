import time
import os.path as osp
import numpy as np

import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

render_rate = 10


def load_policy(fpath):
    """
    函数作用: 加载使用 Spinning Up Logger 保存的 Policy 模型, 并返回动作获取函数.
    """
    print("Loading policy model from %s!\n" % fpath)
    net = torch.load(fpath, map_location="cpu")

    # 生成智能体动作的函数
    @torch.no_grad()
    def get_action(obs_n):
        logits = net(torch.Tensor(obs_n))
        pi = Categorical(logits=logits)
        action_n = pi.sample().numpy()
        return action_n

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    """
    函数作用: 运行一个策略来与环境进行交互.

    函数参数:
        env: 环境对象, 用于与策略进行交互.
        get_action: 生成动作的函数.
        max_ep_len: 每个回合的最大步数.
        num_episodes: 要运行的回合数目.
        render: 是否需要渲染可视化的环境.
    """
    # 初始化 状态观测量, 回合回报, 回合步数, 回合数
    obs_n, ep_ret, ep_len, episode = env.reset()[1], 0, 0, 0
    while episode < num_episodes:
        # 渲染环境
        if render:
            env.render()
            time.sleep(1 / render_rate)

        action_n = get_action(obs_n)
        obs_n, rew_n, done_n, _ = env.step(action_n, 0)
        obs_n, rew_n = obs_n[1], rew_n[1]
        ep_ret += np.sum(rew_n)
        ep_len += 1
        done = done_n
        # 检测回合是否结束
        if done or (ep_len == max_ep_len):
            print(
                "Episode %d \t EpRet %.3f \t EpLen %d \t Result %s"
                % (episode, ep_ret, ep_len, done)
            )
            obs_n, ep_ret, ep_len = env.reset()[1], 0, 0
            episode += 1


def make_env(benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # 创建并初始化环境
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    # 解析参数
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        done_callback=scenario.is_done,
        shared_viewer=True,
    )
    return env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # 模型文件路径
    parser.add_argument("--fpath", type=str, default="model/model.pt")
    # 每回合最大步数
    parser.add_argument("--len", type=int, default=400)
    # 回合数
    parser.add_argument("--episodes", type=int, default=10)
    # 是否禁用渲染
    parser.add_argument("--norender", action="store_true")
    # 是否离散化动作空间
    parser.add_argument("--discrete_action_space", default=False, action="store_true")
    # 是否运行最终模型
    parser.add_argument("--final", default=False, action="store_true")
    args = parser.parse_args()

    env = make_env(args)
    get_action = load_policy(args.fpath)

    run_policy(env, get_action, args.len, args.episodes, not args.norender)
