import datetime
import numpy as np
import time

import torch

from dqn.DQN.agent import DQN
from multiagent.environment import MultiAgentEnv
import multiagent.scenarios as scenarios

# 逃逸者 id
agent_id = 1
render_rate = 30
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time


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


class DQNConfig:
    def __init__(self) -> None:
        self.algo = "DQN"
        self.env = "simple_tag.py"
        self.seed = 15
        self.ShowImage = True  # render image
        self.load_model = True  # load model
        self.train = False
        self.model_path = "models/"  # path to save models
        self.capacity = int(2e5)  # replay buffer size
        self.batch_size = 256  # minibatch size
        self.gamma = 0.99  # discount factor
        self.tau = 1e-3  # for soft update of target parameters
        self.lr = 1e-4  # learning rate
        self.update_every = 1  # Learn every UPDATE_EVERY time steps.
        self.train_eps = 10000
        self.train_steps = 2000
        self.eval_eps = 1
        self.eval_steps = 2000
        self.eps_start = 1.0
        self.eps_decay = 0.995
        self.eps_end = 0.01
        self.hidden_layers = [256, 64]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.frames = 1


def env_agent_config(cfg: DQNConfig):
    """
    Create env and agent
    ------------------------
    Input: configuration
    Output: env, agent
    """
    # 创建并初始化环境
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(
        world,
        scenario.reset_world,
        scenario.reward,
        scenario.observation,
        info_callback=None,
        done_callback=scenario.is_done,
        shared_viewer=False,
    )
    state_dim = env.observation_space[agent_id].shape[0]
    action_dim = env.action_space[agent_id].n

    print(f"action dim: {action_dim}, state dim: {state_dim}")
    agent = DQN(state_dim, action_dim, cfg)
    print(agent.qnetwork_local)
    return env, agent


def train(cfg: DQNConfig, env, agent):
    """
    Train model, and save agent and result during training.
    --------------------------------
    Input: configuration, env, agent
    Output: reward and ma_reward
    """
    print("Start to train !")
    print(f"Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}")
    rewards = []
    # moveing average reward
    ma_rewards = []

    eps = cfg.eps_start
    for i_ep in range(cfg.train_eps):
        observation = env.reset()
        state = observation[agent_id]
        ep_reward = 0
        i_step = 0

        while True:
            # for i_step in range(cfg.train_steps):
            action = agent.act(state, eps)
            action_n = action_one_hot(action)
            next_observation, reward, done_n, info = env.step(action_n)
            next_state = next_observation[agent_id]
            reward = reward[agent_id]

            terminated = done_n[0]
            if terminated is None:
                terminated = False

            truncated = done_n[1]
            if truncated is None:
                truncated = False

            agent.step(state, action, reward, next_state, terminated or truncated)
            state = next_state
            ep_reward += reward
            i_step += 1

            # print(f"episode: {i_ep}, step: {i_step}", end="\r")
            # print(f"action: {action}, real_action: {action_in}")
            # print(f"state: {state}", end="\n")
            print(
                f"Episode:{i_ep+1}/{cfg.train_eps}, eps: {eps}, step {i_step}, action: {action} Reward:{ep_reward:.3f}",
                end="\r",
            )
            if terminated or truncated or (i_step >= cfg.train_steps):
                break

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        eps = max(cfg.eps_end, cfg.eps_decay * eps)
        if i_ep % 20 == 0:
            print("\nsave model")
            agent.save(cfg.model_path)
            np.savetxt(cfg.model_path + "reward_{}.txt".format(curr_time), rewards)
            np.savetxt(
                cfg.model_path + "ma_reward_{}.txt".format(curr_time), ma_rewards
            )
            if i_ep > 1500:
                tmp_rewards, tmp_ma_rewards = eval(cfg, env, agent)
                if tmp_rewards[-1] >= 0.0:
                    while True:
                        tmp_rewards, tmp_ma_rewards = eval(
                            cfg, env, agent, ShowImage=True
                        )

    print("Complete training！")
    return rewards, ma_rewards


def eval(cfg: DQNConfig, env, agent, ShowImage=False):
    """
    Evaluate Model
    """
    print("Start to eval !")
    print(f"Env: {cfg.env}, Algorithm: {cfg.algo}, Device: {cfg.device}")
    rewards = []
    ma_rewards = []

    for i_ep in range(cfg.eval_eps):
        observation = env.reset()
        state = observation[agent_id]
        ep_reward = 0
        i_step = 0
        while True:
            if ShowImage:
                env.render()
                time.sleep(1 / render_rate)
            action = agent.act(state)
            action_n = action_one_hot(action)
            next_observation, reward, done_n, info = env.step(action_n)
            state = next_observation[agent_id]
            reward = reward[agent_id]

            terminated = done_n[0]
            if terminated is None:
                terminated = False

            truncated = done_n[1]
            if truncated is None:
                truncated = False

            ep_reward += reward
            i_step += 1

            print(
                f"Episode:{i_ep+1}/{cfg.eval_eps}, action: {action}, step {i_step} Reward:{ep_reward:.3f}",
                end="\r",
            )
            if terminated or truncated or (i_step >= cfg.eval_steps):
                break

        print(f"\nEpisode:{i_ep+1}/{cfg.train_eps}, Reward:{ep_reward:.3f}")
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    np.savetxt(cfg.model_path + "reward_eval.txt", rewards)
    print("Complete evaling！")
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = DQNConfig()
    env, agent = env_agent_config(cfg)

    if cfg.train:
        # train
        if cfg.load_model:
            print(">>>>>>>>>>load model<<<<<<<<<<<<<<<")
            agent.load(path=cfg.model_path)
        rewards, ma_rewards = train(cfg, env, agent)
    else:
        # eval
        print(">>>>>>>>>>load model<<<<<<<<<<<<<<<")
        agent.load(path=cfg.model_path)
        rewards, ma_rewards = eval(cfg, env, agent, ShowImage=cfg.ShowImage)
