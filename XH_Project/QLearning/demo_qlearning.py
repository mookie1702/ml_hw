import random
import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

# 设置算法的超参数
num_episodes = 100000
learning_rate = 0.1
discount_factor = 0.99

epsilon = 1
epsilon_discount = 0.999


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


# 定义Q-learning算法的主函数
def q_learning(env, num_episodes, learning_rate, discount_factor, epsilon):
    # 创建Q-table，初始化为0
    num_states_rows = 50
    num_states_cols = 50
    num_actions = 4
    Q = np.zeros((num_states_rows, num_states_cols, num_actions))

    # 迭代每个episode
    for episode in range(num_episodes):
        state = env.reset()[1][0:2]
        epsilon *= epsilon_discount

        if epsilon < 0.01:
            epsilon = 0.0

        while True:
            x = int((state[0] + 1.00) * 25)
            y = int((state[1] + 1.00) * 25)
            state[0] = x
            state[1] = y
            # 使用epsilon-greedy策略选择动作
            if np.random.rand() < epsilon:
                action = random.randint(1, 4)
            else:
                action = np.argmax(Q[x, y])

            # 执行动作，得到新的状态和奖励
            action_n = action_one_hot(action)
            next_state, reward, done, _ = env.step(action_n)
            next_state = next_state[1][0:2]
            reward = reward[1]

            next_x = int((next_state[0] + 1.00) * 25)
            next_y = int((next_state[1] + 1.00) * 25)

            # 使用Q-learning更新Q值
            Q[x, y, action - 1] += learning_rate * (
                reward
                + discount_factor * np.max(Q[next_x, next_y])
                - Q[x, y, action - 1]
            )
            state = next_state
            if True in done:
                if True == done[0]:
                    result = "Fail!"
                else:
                    result = "Succeed!"
                print(episode, result, epsilon)
                break

    return Q


if __name__ == "__main__":
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
        shared_viewer=True,
    )

    # 设置追捕智能体与逃逸智能体的位置
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))

    Q = q_learning(env, num_episodes, learning_rate, discount_factor, epsilon)
