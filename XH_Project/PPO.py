import numpy as np
from gym.spaces import Discrete

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

import policy_gradient


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


class GAE_buffer:
    """
    A buffer for storing trajectories experienced by a policy gradient agent
    interacting with the environment, and using Generalized Advantage Estimation
    (GAE-Lambda) for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, obs_dim, act_dim, gamma, lam):
        """
        :param size: 缓冲区大小, 即数据点的数量
        :param obs_dim: 状态空间的维度
        :param act_dim: 动作空间的维度
        :param gamma: 折扣因子,用于计算奖励的折扣累积
        :param lam: GAE-lambda, lam=1 means REINFORCE and lam=0 means A2C, typically 0.9~0.99
        """
        # 存储状态的数组
        self.obs_buf = np.zeros(policy_gradient.combined_shape(size, obs_dim))
        # 存储动作的数组
        self.act_buf = np.zeros(policy_gradient.combined_shape(size, act_dim))
        # 存储 reward 的数组
        self.rew_buf = np.zeros((size,))
        # 存储 value 的数组
        self.val_buf = np.zeros((size,))
        # 存储折扣累积奖励的数组
        self.rtg_buf = np.zeros((size,))
        # 存储 advantage 的数组
        self.adv_buf = np.zeros((size,))

        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val):
        """
        Function: Store one transition into buffer.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, val):
        """
        Compute reward-to-go whenever the following cases appear
         1. agent dies, which means the return following is zero.
         2. it reaches the max_episode_len or the trajectory being cut off at time T,
            then you should provided an estimate V(S_T) using critic to compensate for
            the rewards beyond time T.

        :param val: the value estimated by critic for the final state

        return: None
        """
        # 获取在 GAE_buffer 中的当前轨迹
        path_slice = slice(self.path_start_idx, self.ptr)
        # 获取 reward 数组
        rews = np.append(self.rew_buf[path_slice], val)
        # 获取 value 数组
        vals = np.append(self.val_buf[path_slice], val)
        # GAE 算法
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        self.adv_buf[path_slice] = policy_gradient.discount_cumsum(
            deltas, self.gamma * self.lam
        )
        self.rtg_buf[path_slice] = policy_gradient.discount_cumsum(rews, self.gamma)[
            :-1
        ]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size, "You must fulfill buffer before getting data!"
        self.path_start_idx, self.ptr = 0, 0

        adv_mu, adv_std = policy_gradient.statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mu) / adv_std
        data = dict(
            obs=self.obs_buf, act=self.act_buf, rtg=self.rtg_buf, adv=self.adv_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class ppo:
    def __init__(
        self,
        env_fn,
        hid=256,
        layers=2,
        gamma=0.99,
        lam=0.97,
        agent_type="good",
        seed=0,
        steps_per_epoch=4000,
        pi_lr=1e-2,
        v_lr=1e-3,
        clip_ratio=0.2,
    ):
        super(ppo, self).__init__()
        # agent 0 为追击者, agent 1 为逃逸者
        self.agent_id = 1 if agent_type == "good" else 0
        # 初始化环境
        self.env = env_fn()

        # TODO: 不清楚这里为什么 self.discrete = False, 正常应该是 True.
        # self.discrete = isinstance(self.env.action_space[self.agent_id], Discrete)
        self.discrete = True
        self.obs_dim = self.env.observation_space[self.agent_id].shape[0]
        self.act_dim = (
            self.env.action_space[self.agent_id].n
            if self.discrete
            else self.env.action_space[self.agent_id].shape[0]
        )

        # random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ppo clip ratio
        self.clip_ratio = clip_ratio

        # create actor-critic network
        self.mlp_sizes = [self.obs_dim] + [hid] * layers
        self.log_std = torch.nn.Parameter(
            -0.5 * torch.ones(self.act_dim, dtype=torch.float32)
        )
        self.pi = policy_gradient.mlp(
            sizes=self.mlp_sizes + [self.act_dim], activation=nn.Tanh
        )
        self.v = policy_gradient.mlp(sizes=self.mlp_sizes + [1], activation=nn.Tanh)

        # Count variables
        var_counts = tuple(
            policy_gradient.count_vars(module) for module in [self.pi, self.v]
        )
        print("Number of parameters: \t pi: %d, \t v: %d\n" % var_counts)

        # 创建一个 GAE_buffer
        # Discrete action in buf is of shape (N, )
        self.steps_per_epoch = steps_per_epoch
        self.buf = GAE_buffer(
            steps_per_epoch,
            self.obs_dim,
            self.env.action_space[self.agent_id].shape,
            gamma,
            lam,
        )

        # 设置 optimizer 为 Adam 优化算法.
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.v_optimizer = Adam(self.v.parameters(), lr=v_lr)

    def act(self, obs):
        """
        Function: Used for collecting trajectories or testing, which doesn't require tracking grads.
        """
        with torch.no_grad():
            logits = self.pi(obs)
            if self.discrete:
                pi = Categorical(logits=logits)
            else:
                pi = Normal(loc=logits, scale=self.log_std.exp())
            act = pi.sample()
        return act.numpy()

    def log_prob(self, obs, act):
        """
        Compute log prob for the given batch observations and actions.
        :param obs: Assume shape (N, D_0)
        :param act: Assume shape (N, D_a)
        :return: log_prob of shape (N,)
        """
        act = act.squeeze(dim=-1)  # critical for discrete actions!
        logits = self.pi(obs)
        if self.discrete:
            pi = Categorical(logits=logits)
            return pi.log_prob(act)
        else:
            pi = Normal(loc=logits, scale=self.log_std.exp())
            return pi.log_prob(act).sum(dim=-1)

    def train(
        self,
        epochs=50,
        train_v_iters=80,
        train_pi_iters=80,
        max_ep_len=100,
        target_kl=0.01,
        save_name="model.pt",
    ):
        # 记录训练过程中每个回合的 return 值
        log = open("./models/result.txt", "w")
        # ret_stat 记录每个回合的 return 值, len_stat 存储回合长度.
        ret_stat, len_stat = [], []
        # 初始化环境, 并初始化 状态量, 每回合 return 值和回合长度.
        o, ep_ret, ep_len = self.env.reset()[self.agent_id], 0, 0
        for e in range(epochs):
            for t in range(self.steps_per_epoch):
                # 将观测值 o 转为 tensor
                o_torch = torch.as_tensor(o, dtype=torch.float32)
                # 由 o 的 tensor 获取 actor 和 critic 网络的输出
                a = self.act(o_torch)
                a_n = action_one_hot(a)
                v = self.v(o_torch).detach().numpy()
                # 进行游戏
                next_o, r, d, _ = self.env.step(a_n)

                next_o, r, d = next_o[self.agent_id], r[self.agent_id], any(d)
                ep_ret += r
                ep_len += 1
                self.buf.store(o, a, r, v)
                o = next_o

                # 判断回合是否结束
                timeout = ep_len == max_ep_len
                terminal = d or timeout
                epoch_ended = t == self.steps_per_epoch - 1

                if terminal or epoch_ended:
                    if epoch_ended and not terminal:
                        print(
                            "Warning: trajectory cut off by epoch at %d steps."
                            % ep_len,
                            flush=True,
                        )
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if timeout or epoch_ended:
                        v = (
                            self.v(torch.as_tensor(o, dtype=torch.float32))
                            .detach()
                            .numpy()
                        )
                    else:
                        v = 0.0
                    if terminal:
                        ret_stat.append(ep_ret)
                        len_stat.append(ep_len)
                    self.buf.finish_path(v)
                    # 重置环境
                    o, ep_ret, ep_len = self.env.reset()[self.agent_id], 0, 0

            # 获取本回合轨迹的训练数据
            data = self.buf.get()
            loss_pi, kl = self.update_pi(data, train_pi_iters, target_kl)
            loss_v = self.update_v(data, train_v_iters)
            print(
                "epoch: %3d \t loss of pi: %.3f \t loss of v: %.3f \t kl: %.3f \t return: %.3f \t ep_len: %.3f\n"
                % (e, loss_pi, loss_v, kl, np.mean(ret_stat), np.mean(len_stat))
            )
            log.write(str(np.mean(ret_stat)) + "\n")

            torch.save(self.pi, save_name)
            ret_stat, len_stat = [], []
        log.close()

    def update_pi(self, data, iter, target_kl):
        obs, act, adv = data["obs"], data["act"], data["adv"]
        logp_old = self.log_prob(obs, act).detach()
        for i in range(iter):
            self.pi_optimizer.zero_grad()
            logp = self.log_prob(obs, act)
            appro_kl = (logp_old - logp).mean().item()
            if appro_kl > 1.5 * target_kl:
                print("PPO stops at iter %d" % i)
                break

            ratio = (logp - logp_old).exp()
            adv_clip = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            )
            loss_pi = -torch.min(ratio * adv, adv_clip).mean()
            loss_pi.backward()
            self.pi_optimizer.step()
        return loss_pi.item(), appro_kl

    def update_v(self, data, iter):
        obs, rtg = data["obs"], data["rtg"]
        for i in range(iter):
            self.v_optimizer.zero_grad()
            v = self.v(obs).squeeze()
            loss_v = ((v - rtg) ** 2).mean()
            loss_v.backward()
            self.v_optimizer.step()
        return loss_v.item()
