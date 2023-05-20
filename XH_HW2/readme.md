# RL in Gym

## 环境

- pytorch
- gymnasium

> see [here](https://github.com/Farama-Foundation/Gymnasium) for detail.

1. run `pip install gymnasium` to install the base Gymnasium library.
2. run `pip install gymnasium[atari]` and `pip install gymnasium[accept-rom-license]` to install Atari environment. (see [here](https://gymnasium.farama.org/environments/atari/) for detail)

## [SpaceInvaders](https://gymnasium.farama.org/environments/atari/space_invaders/)

> 所有信息可在官网查询.

- 可选环境：[官网](https://gymnasium.farama.org/environments/atari/space_invaders/#variants)
- 状态：`rgb` 或者 `ram`
- 动作：`Discrete(6)`

## Demo

可运行`demo.py`文件查看`demo`效果

- Algorithm: DQN
- Env: ALE/SpaceInvaders-ram-v5
- Score：375

## 要求

- 在一个episode获得分数等于或者超过`demo`, 即375分, 不限制使用各种算法.
- 提交算法代码
- 提交训练过程的`reward_*.txt`和`ma_reward_*.txt`文件。
- 提交最后的测试分数的`reward_eval.txt`文件。

> 提交文件可参考`.\agents\DQN\`和`.\saved_models\SpaceInvaders\`文件夹内的内容。

> 提交到邮箱: 13025291859@163.com
