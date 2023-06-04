import numpy as np

from multiagent.core import World, Agent, Landmark, Border, Check
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        """
        函数作用: 创建虚拟世界对象, 并设置世界的属性和初始条件
        """
        world = World()
        # 设置世界的维度
        world.dim_c = 2
        # 设置智能体的数目
        num_good_agents = 1
        num_adversaries = 1
        num_agents = num_adversaries + num_good_agents
        # 设置障碍物的数目
        num_landmarks = 3
        # 设置检查点的数目
        num_check = 1
        # 设置边界的长度: (20 * 4, 4 条边，每边 20 个 border)
        num_borders = 80
        # 添加智能体
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent %d" % i
            agent.collide = True
            # 设置静默性: 智能体之间是否可以交流
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.025 if agent.adversary else 0.025
            # 设置加速度
            agent.accel = 0.85 if agent.adversary else 1.0
            # 设置最大速度
            agent.max_speed = 0.28 if agent.adversary else 0.25
        # 添加障碍物
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.boundary = False
        # 添加检查点
        world.check = [Check() for i in range(num_check)]
        for i, check in enumerate(world.check):
            check.name = "checkpoint %d" % i
            check.collide = False
            check.movable = False
            check.size = 0.05
            check.boundary = False
            check.shape = [
                [-0.025, -0.025],
                [0.025, -0.025],
                [0.025, 0.025],
                [-0.025, 0.025],
            ]
        # 添加边界
        world.borders = [Border() for i in range(num_borders)]
        for i, border in enumerate(world.borders):
            border.name = "border %d" % i
            border.collide = True
            border.movable = False
            border.size = 0.15
            border.boundary = True
            # 设置边界厚度
            border.shape = [[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]]
        # 重置世界
        self.reset_world(world)
        return world

    def reset_world(self, world, agent_pos=None):
        # 设置智能体颜色
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
        # 设置障碍物颜色
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # 设置边界颜色
        for i, border in enumerate(world.borders):
            border.color = np.array([0.8, 0.4, 0.4])
        # 设置终点颜色
        for i, check in enumerate(world.check):
            check.color = np.array([0.8, 0.6, 0.8])
        # 设置智能体初始位置状态 ([x,y]=[0,0]是在视野中心)
        for agent in world.agents:
            if agent_pos is None:
                agent.state.p_pos = (
                    np.array([0.0, 0.0]) if agent.adversary else np.array([0.5, 0.5])
                )
            else:
                agent.state.p_pos = agent_pos[0] if agent.adversary else agent_pos[1]
            # 设置智能体的初始速度
            agent.state.p_vel = np.zeros(world.dim_p)
            # 设置智能体初始交流状态
            agent.state.c = np.zeros(world.dim_c)
        # 设置障碍物的位置和速度
        pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]]
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = pos[i]
                landmark.state.p_vel = np.zeros(world.dim_p)
        # 设置检查点的位置和速度
        world.check[0].state.p_pos = [-0.5, -0.5]
        world.check[0].state.p_vel = np.zeros(world.dim_p)
        # 每条边20个 border, 计算大概位置, 依次为每条边的 border 生成位置坐标
        pos = []
        # 底部边界
        x = -0.95
        y = -1.0
        for count in range(20):
            pos.append([x, y])
            x += 0.1
        # 右边边界
        x = 1.0
        y = -0.95
        for count in range(20):
            pos.append([x, y])
            y += 0.1
        # 上面边界
        x = 0.95
        y = 1.0
        for count in range(20):
            pos.append([x, y])
            x -= 0.1
        # 左边边界
        x = -1.0
        y = 0.95
        for count in range(20):
            pos.append([x, y])
            y -= 0.1
        # 设置边界位置
        for i, border in enumerate(world.borders):
            border.state.p_pos = np.asarray(pos[i])
            border.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        """
        Function: returns data for benchmarking purposes.
        """
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        """
        函数作用: 检测智能体之间是否发生碰撞.
        """
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def good_agents(self, world):
        """
        Function: return all agents that are not adversaries.
        """
        return [agent for agent in world.agents if not agent.adversary]

    def adversaries(self, world):
        """
        Function: return all adversarial agents.
        """
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        """
        函数作用: 计算 agent 的 reward.
        """
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        return main_reward

    def is_done(self, agent, world):
        """
        函数作用: 检测游戏是否结束.
        """
        # 追捕智能体的结束条件
        if agent.adversary:
            good_agent = self.good_agents(world)[0]
            # 检测追捕智能体是否追上逃逸智能体
            if self.is_collision(good_agent, agent):
                return True
        # 逃逸智能体的结束条件
        if not agent.adversary:
            # 检测逃逸智能体是否与障碍物碰撞
            for i, landmark in enumerate(world.landmarks):
                delta_dis = agent.state.p_pos - landmark.state.p_pos
                dist = np.sqrt(np.sum(np.square(delta_dis)))
                # dist_min = agent.size + landmark.size
                dist_min = 0.075
                if dist <= dist_min:
                    return True
            # 检测逃逸智能体是否到达检查点
            dist = np.sqrt(
                np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos))
            )
            if dist < agent.size + world.check[0].size:
                return True

    def agent_reward(self, agent, world):
        """
        函数作用: 返回逃逸智能体的奖励
        """
        rew = 0
        adversaries = self.adversaries(world)
        # 设置碰撞惩罚
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                if self.is_collision(landmark, agent):
                    rew -= 10
        for i, border in enumerate(world.borders):
            if self.is_collision(border, agent):
                rew -= 10
        # 设置距离惩罚: 距离 check 点越远，惩罚越大
        dist = np.sqrt(
            np.sum(np.square(agent.state.p_pos - world.check[0].state.p_pos))
        )
        rew -= 0.5 * dist
        # 设置完成任务的奖励
        if dist < agent.size + world.check[0].size:
            rew += 12
        return rew

    def adversary_reward(self, agent, world):
        """
        函数作用: 返回追捕智能体的奖励.
        """
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            for adv in adversaries:
                rew -= 0.1 * min(
                    [
                        np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos)))
                        for a in agents
                    ]
                )
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        """
        函数作用: 返回游戏的观测数值.
        """
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        check_pos = []
        check_pos.append(agent.state.p_pos - world.check[0].state.p_pos)
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            #     other_vel.append(other.state.p_vel)
            other_vel.append(other.state.p_vel)
            dists = np.sqrt(np.sum(np.square(agent.state.p_pos - other_pos)))
        # return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + dists)   # 知道自己的位置、速度和与其他agent的距离
        return np.concatenate(
            [agent.state.p_pos]
            + other_pos
            + check_pos
            + entity_pos
            + [agent.state.p_vel]
            + other_vel
        )
