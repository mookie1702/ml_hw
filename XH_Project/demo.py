import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

if __name__ == "__main__":
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
    # 设置追捕智能体与逃逸智能体的位置
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))
    # 读取游戏图片
    image = env.render("rgb_array")

    step = 0
    total_step = 0
    max_step = 400
    while True:
        # 动作空间: [noop, move right, move left, move up, move down]
        act_n = np.array([0, 0, 1, 0, 1])
        next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)

        # 读取游戏图片
        image = env.render("rgb_array")[0]
        # print(f"shape: {np.shape(image)}")

        step += 1

        if True in done_n or step > max_step:
            break
