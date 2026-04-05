import numpy as np
data = np.load("D:/dytj/RL-PVES - all/RL-PVES - all/data/images/train_rewards_TD3.npy", allow_pickle=True)
np.set_printoptions(threshold=np.inf)
print(data.shape)
print(data)