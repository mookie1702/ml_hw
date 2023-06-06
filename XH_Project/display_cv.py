import time
import numpy as np

import torch
from torch.distributions.categorical import Categorical

render_rate = 100
"""画面渲染帧率 (帧/每秒)"""

old_red = [0.0, 0.0]
"""追击者初始位置"""
old_green = [0.7, 0.7]
"""逃逸者初始位置"""


def observation_cv(env):
    """
    函数作用: 由数字图像处理获取观测量.
    """

    def findObject(img):
        conts, heriachy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        location = []
        for cnt in conts:
            x1, y1, w, h = cv.boundingRect(cnt)
            ty = int(y1 + h / 2)
            tx = int(x1 + h / 2)
            location.append([tx, ty])
        return location

    def getCoordinate(loc):
        loc[0] = (loc[0] - 400) / 400
        loc[1] = (400 - loc[1]) / 400
        return loc

    import cv2 as cv

    global old_red
    global old_green

    obs_ = []
    # 获取当前游戏画面图像 img_bgr
    image = env.render(mode="rgb_array")
    img_bgr = np.array(image)[0]
    img_bgr = img_bgr[:, :, ::-1]

    # 获取 HSV 图像 img_hsv
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    # 追击者图像
    l_blue = np.array([[0, 43, 46]])
    h_blue = np.array([10, 255, 255])
    red_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # 逃逸者图像
    l_blue = np.array([[35, 43, 46]])
    h_blue = np.array([77, 255, 255])
    green_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # 检查点图像
    l_blue = np.array([[125, 43, 46]])
    h_blue = np.array([155, 255, 255])
    check_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # 障碍物图像
    l_blue = np.array([[0, 0, 46]])
    h_blue = np.array([180, 43, 200])
    landmark_mask = cv.inRange(img_hsv, l_blue, h_blue)

    # 获取追击者位置
    red_loc = getCoordinate(findObject(red_mask)[0])
    # 获取逃逸者位置
    green_loc = getCoordinate(findObject(green_mask)[0])
    # 获取检查点位置
    check_loc = getCoordinate(findObject(check_mask)[0])
    # 获取路障位置
    landmark_loc = findObject(landmark_mask)
    landmark1_loc = getCoordinate(landmark_loc[2])
    landmark2_loc = getCoordinate(landmark_loc[1])
    landmark3_loc = getCoordinate(landmark_loc[0])

    other_pos = [red_loc[0] - green_loc[0], red_loc[1] - green_loc[1]]
    check_pos = [green_loc[0] - check_loc[0], green_loc[0] - check_loc[1]]
    entity1_pos = [landmark1_loc[0] - green_loc[0], landmark1_loc[1] - green_loc[1]]
    entity2_pos = [landmark2_loc[0] - green_loc[0], landmark2_loc[1] - green_loc[1]]
    entity3_pos = [landmark3_loc[0] - green_loc[0], landmark3_loc[1] - green_loc[1]]
    p_vel = [green_loc[0] - old_green[0], green_loc[1] - old_green[1]]
    other_vel = [red_loc[0] - old_red[0], red_loc[1] - old_red[1]]
    old_red = red_loc
    old_green = green_loc
    obs_[0:1] = green_loc
    obs_[2:3] = other_pos
    obs_[4:5] = check_pos
    obs_[6:7] = entity1_pos
    obs_[8:9] = entity2_pos
    obs_[10:11] = entity3_pos
    obs_[12:13] = p_vel
    obs_[14:15] = other_vel
    return np.array(obs_)


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
        obs_n, rew_n = observation_cv(env), rew_n[agent_id]
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
            env.reset()
            obs_n, ep_ret, ep_len = observation_cv(env), 0, 0
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
    parser.add_argument("--fpath", type=str, default="models/model_cv.pt")
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
