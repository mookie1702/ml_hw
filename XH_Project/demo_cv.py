import numpy as np

import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv

old_red = [0.0, 0.0]
old_green = [0.7, 0.7]
obs_ = []


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

    # 获取当前游戏画面图像 img_bgr
    image = env.render(mode="rgb_array")
    img_bgr = np.array(image)[0]
    img_bgr = img_bgr[:, :, ::-1]
    # cv.imwrite("img_bgr.jpg", img_bgr)

    # 获取灰度图像 img_gray
    # img_gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    # cv.imwrite("img_gray.jpg", img_gray)

    # 获取 HSV 图像 img_hsv
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    # cv.imwrite("img_hsv.jpg", img_hsv)

    # 追击者图像
    l_blue = np.array([[0, 43, 46]])
    h_blue = np.array([10, 255, 255])
    red_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # cv.imshow("red_mask", red_mask)
    # cv.waitKey(0)

    # 逃逸者图像
    l_blue = np.array([[35, 43, 46]])
    h_blue = np.array([77, 255, 255])
    green_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # cv.imshow("green_mask", green_mask)
    # cv.waitKey(0)

    # 检查点图像
    l_blue = np.array([[125, 43, 46]])
    h_blue = np.array([155, 255, 255])
    check_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # cv.imshow("check_mask", check_mask)
    # cv.waitKey(0)

    # 障碍物图像
    l_blue = np.array([[0, 0, 46]])
    h_blue = np.array([180, 43, 200])
    landmark_mask = cv.inRange(img_hsv, l_blue, h_blue)
    # cv.imshow("landmark_mask", landmark_mask)
    # cv.waitKey(0)

    # 获取追击者位置
    red_loc = getCoordinate(findObject(red_mask)[0])
    # print(f"追击者位置: {red_loc}")

    # 获取逃逸者位置
    green_loc = getCoordinate(findObject(green_mask)[0])
    # print(f"逃逸者位置: {green_loc}")

    # 获取检查点位置
    check_loc = getCoordinate(findObject(check_mask)[0])
    # print(f"检查点位置: {check_loc}")

    # 获取路障位置
    landmark_loc = findObject(landmark_mask)
    landmark1_loc = getCoordinate(landmark_loc[2])
    landmark2_loc = getCoordinate(landmark_loc[1])
    landmark3_loc = getCoordinate(landmark_loc[0])
    # print(f"障碍物1位置: {landmark1_loc}")
    # print(f"障碍物2位置: {landmark2_loc}")
    # print(f"障碍物3位置: {landmark3_loc}")

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
    return obs_


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

    # 动作空间: [noop, move right, move left, move up, move down]
    act_n = np.array([0, 0, 1, 0, 1])
    next_obs_n, reward_n, done_n, _ = env.step(act_n)

    print(next_obs_n[1])
    obs_ = observation_cv(env)
    print(obs_)
