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

    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))
    # 使用这种方法读取图片
    image = env.render("rgb_array")

    step = 0
    total_step = 0
    max_step = 400
    # file = open('./course/label.txt', 'w')
    while True:
        # 动作空间:[noop, move right, move left, move up, move down]
        act_n = np.array([0, 0, 1, 0, 1])
        next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
        # print(f"vel: {np.linalg.norm(next_obs_n[0][-4:-2])}, vel2: {np.linalg.norm(next_obs_n[0][-2:])}")
        # 使用这种方法读取图片
        image = env.render("rgb_array")[0]

        # print(f"shape: {np.shape(image)}")
        step += 1
        # if step % 50:
        #     print(image)
        #     plt.imshow(image)
        #     plt.show()
        #     time.sleep(0.0167) # 60 fps
        #     plt.close()

        if True in done_n or step > max_step:
            break
