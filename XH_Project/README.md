# 环境配置

质点小球追逃环境

> 参考 [博客](https://blog.csdn.net/kysguqfxfr/article/details/100070584?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-3.channel_param)

## 安装

系统：ubuntu （windows下有些依赖可能会有问题）

依赖:

- gymnasium 
- pyglet

> 运行 `demo.py` 检查环境，缺什么包下载什么包.

## 环境介绍

> 以`demo.py`为例子。

首先创建并初始化环境：

```py
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer = True)
    
    env.reset()
    image = env.render("rgb_array")  # 使用这种方法读取图片
```

动作为五维向量，范围分别是`[0,1]`, 控制的是加速度方向和大小

```py
	# [noop, move right, move left, move up, move down]
        act_n = np.array([0,1,0,0,1])
```

状态为`(800,800,3)`大小的rgb图片，可以用于目标检测得到位置状态或者直接端到端训练智能体。整个画⾯的实际尺度为 [-1,1] * [-1,1] , 屏幕正中⼼为原点。画⾯宽⾼为800*800像素

```py
    image = env.render("rgb_array")  # 使用这种方法读取图片
```

另外，除了图像数据，也直接获取状态，获取方法与一般的环境一样

```py
obs_n = env.reset()
.
.
.
next_obs_n, reward_n, done_n, _ = env.step(act_n, 1)
```

在 simple_tag.py ⽂件中的环境类中的函数`observation()`有直接获取位置等信息的代码

但是这种⽅式是不符合⽼师所提出的要求的，⽼师是想让同学们⽤图像处理等知识获取状态信息。

障碍物位置`[-0.35, 0.35], [0.35, 0.35], [0, -0.35]`, 半径0.05.

重置函数, 第一个位置是追击者，第二个位置是控制的智能体。

```py
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]))
```

撞倒障碍物（不包括边界）和被追击者追上就会结束

```py
if True in done_n:
            break
```

## 环境文件

最重要的环境⽂件是 core.py , simple_tag.py , environment.py 这三个⽂件。这三个⽂件的⼤致调⽤关系是：core被simple_tag 调⽤，simple_tag被environment调⽤。
- core⽂件中主要声明了环境中出现的各个实体：agent，border，landmark，check等。
- simple_tag⽂件主要设置实体的参数（初始位置、加速度等），reward设置等。
- environment⽂件是强化学习算法中的经典环境接口（step,reset, reward等）。你可以按照给定任务重写 environment.py ⽂件中的 self._set_action 函数。

## 要求

能不被追击者追上并到达打卡点（[-0.5,-0.5]）（（正方形位置）
