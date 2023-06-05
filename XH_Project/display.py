import time
import numpy as np

import torch
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

render_rate = 100
"""画面渲染帧率 (帧/每秒)"""


def action_one_hot(num):
    """
    函数作用: 对动作进行 one-hot 编码.
    """
    if num == 0:
        # action: noop
        return [1, 0, 0, 0, 0]
    elif num == 1:
        # action: move right
        return [0, 1, 0, 0, 0]
    elif num == 2:
        # action: move left
        return [0, 0, 1, 0, 0]
    elif num == 3:
        # action: move up
        return [0, 0, 0, 1, 0]
    elif num == 4:
        # action: move down
        return [0, 0, 0, 0, 1]
    else:
        return [0, 0, 0, 0, 0]


def load_policy(fpath):
    """
    函数作用: 加载保存的 Policy 模型, 并返回动作获取函数.
    """
    print("Loading policy model from %s\n" % fpath)
    net = torch.load(fpath, map_location="cpu")

    # 生成 agent 动作的函数
    @torch.no_grad()
    def get_action(obs_n):
        logits = net(torch.Tensor(obs_n))
        pi = Categorical(logits=logits)
        action = pi.sample().numpy()
        return action

    return get_action


def run_policy(
    env, get_action, max_ep_len=None, num_episodes=100, render=True, agent_type="good"
):
    """
    函数作用: 运行一个 agent 策略来与环境进行交互.

    :param env: 环境对象, 用于与 agent 进行交互.
    :param get_action: 生成动作的函数.
    :param max_ep_len: 每个回合的最大步数.
    :param num_episodes: 要运行的回合数目.
    :param render: 是否需要渲染可视化.
    :param agent_type: agent 的类型, good 为逃逸者
    """
    agent_id = 1 if agent_type == "good" else 0
    # 初始化 状态观测量, 回合回报, 回合步数, 回合数
    obs_n, ep_ret, ep_len, episode = env.reset()[agent_id], 0, 0, 0
    while episode < num_episodes:
        # 渲染环境
        if render:
            env.render()
            time.sleep(1 / render_rate)
        # 获取 agent 动作
        action = get_action(obs_n)
        action_n = action_one_hot(action)
        # agent 与环境交互
        obs_n, rew_n, done_n, _ = env.step(action_n)
        # 获取 agent 的状态与 reward
        obs_n, rew_n = obs_n[agent_id], rew_n[agent_id]
        ep_ret += np.sum(rew_n)
        ep_len += 1

        # 检测回合是否结束
        if True in done_n or (ep_len == max_ep_len):
            if True == done_n[0]:
                result = "Fail!"
            else:
                result = "Succeed!"
            print(
                "Episode %d \t EpRet %.3f \t EpLen %d \t Result %s"
                % (episode + 1, ep_ret, ep_len, result)
            )
            obs_n, ep_ret, ep_len = env.reset()[agent_id], 0, 0
            episode += 1


def make_env():
    """
    函数作用: 创建游戏环境.
    """
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
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
    parser.add_argument("--fpath", type=str, default="models/model.pt")
    # 每回合最大步数
    parser.add_argument("--len", type=int, default=400)
    # 回合数
    parser.add_argument("--episodes", type=int, default=10)
    # 是否禁用渲染
    parser.add_argument("--norender", action="store_true")
    args = parser.parse_args()

    env = make_env()
    get_action = load_policy(args.fpath)

    run_policy(env, get_action, args.len, args.episodes, not args.norender)
