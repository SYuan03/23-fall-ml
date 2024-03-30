import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

# 注册新的 CartPole 环境
gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='MyModel:CustomCartPoleEnv',
    max_episode_steps=600,  # 可以根据需要修改
)

# 创建新的环境实例
env = gym.make('CustomCartPole-v1', render_mode="human")


# 输入是状态，输出是每个动作的Q值
class DQN(nn.Module):

    def __init__(self, n_observations_, n_actions_):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations_, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions_)

    # forward函数定义了前向传播的运算
    # 返回值是每个动作的Q值
    # x可能包含多个样本，每个样本是一个状态，返回的是每个样本对应两个动作的Q值
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# 定义模型参数
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# 创建模型实例并加载训练好的参数
model = DQN(input_size, output_size)
model.load_state_dict(torch.load("./model/policy_net.pth"))

# 设置模型为评估模式
model.eval()

# 运行模型一次并显示动画
state, info = env.reset()
done = False
while not done:
    # 使用模型选择动作
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = model(state).max(1)[1].view(1, 1)
        print(action)
        # print(type(action))
    state, reward, terminated, truncated, _ = env.step(action.item())

    if terminated:
        print("terminated")

    if truncated:
        print("truncated")
    # done是终止或者被截断的标志
    done = terminated or truncated

    if terminated:
        state = None
    env.render()

# 关闭环境显示
env.close()
