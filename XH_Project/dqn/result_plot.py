import numpy as np
import matplotlib.pyplot as plt

# ma_reward = np.loadtxt("./saved_models/ma_reward_20230517-150629.txt")
# reward = np.loadtxt("./saved_models/reward_20230517-150629.txt")
ma_reward = np.loadtxt("./models/ma_reward_20230607-182909.txt")
reward = np.loadtxt("./models/reward_20230607-182909.txt")

plt.plot(ma_reward, label="ma_reward")
plt.plot(reward, label="reward")
plt.legend()
plt.grid()
plt.show()
