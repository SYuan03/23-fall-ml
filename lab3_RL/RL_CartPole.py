import os

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 注册新的 CartPole 环境
gym.envs.register(
    id='CustomCartPole-v1',
    entry_point='MyModel:CustomCartPoleEnv',
    max_episode_steps=600,  # 可以根据需要修改
)

# 创建新的环境实例
env = gym.make('CustomCartPole-v1', render_mode="rgb_array")

# Transition是一个命名元组，用于表示一个转换，包含四个属性：state, action, next_state, reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# ReplayMemory是一个有限长度的存储器，用于存储Agent的经验
# 也叫作经验回放池（Experience Replay Pool）
class ReplayMemory(object):

    def __init__(self, capacity):
        # deque是一个双端队列，可以从头尾两端添加和删除元素
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        # *args是一个可变参数，可以接受任意多个参数
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # 选取batch_size个转换
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # 自定义实现len函数
        return len(self.memory)


# Q Network是一个简单的全连接神经网络
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


# IPython环境是指在Jupyter Notebook或者Jupyter QtConsole中运行Python代码
# 如果是IPython环境，那么is_ipython为True，调用display模块中的display函数显示动画
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# 启用交互模式，可以在动画进行中更新图像
plt.ion()

# 如果有GPU，使用GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# BATCH_SIZE是一个批次的大小，即每次从ReplayMemory中随机选取多少个转换进行训练
BATCH_SIZE = 128
# GAMMA是折扣因子，用于计算折扣回报
# 折扣回报是指未来的奖励的折扣累加和，越靠近1，越重视未来的奖励
GAMMA = 0.99
# EPS_START是起始的探索率， EPS_END是最终的探索率， EPS_DECAY是探索率的衰减率
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
# TAU是目标网络的更新率，LR是优化器的学习率
TAU = 0.005
LR = 1e-4

# n_actions是动作的数量，其实在CartPole的例子中就是2
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# 打印state 和 info
print("state:")
print(state)
print("info:")
print(info)
print("n_observations:")
print(n_observations)

# policy_net是当前的Q网络，target_net是目标Q网络
# 前者用于选择动作，后者用于计算目标Q值
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

# 一开始target_net初始化为policy_net的参数
target_net.load_state_dict(policy_net.state_dict())

# 优化器使用AdamW，学习率为LR，amsgrad=True表示使用amsgrad算法
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
# 经验回放池的容量为10000
memory = ReplayMemory(10000)

# steps_done是一个计数器，用于记录Agent总共与环境交互了多少次
steps_done = 0


# select_action函数用于根据当前状态选择动作
# 依据epsilon-greedy策略选择动作，即以epsilon的概率随机选择动作，以1-epsilon的概率选择Q值最大的动作
def select_action(state_):
    global steps_done

    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    # 以如果sample大于eps_threshold，那么以当前Q网络选择动作
    if sample > eps_threshold:
        # t.no_grad()表示不需要计算梯度，因为我们不需要对Q网络进行训练，只是用于选择动作
        with torch.no_grad():
            # 由于我们的输入是一个批次的状态，所以返回的是每个状态对应的两个动作的Q值
            # 因此我们要用t.max(1)[1]来获取状态对应的两个动作中Q值最大的那个动作的索引
            # view(1, 1)是将其变形为一个行向量
            return policy_net(state_).max(1)[1].view(1, 1)
    else:
        # 否则随机选择动作，调用env.action_space.sample()函数
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# new
# 用来显示CartPole的状态
def plot_cartpole_state(screen_):
    plt.figure(2)
    plt.clf()
    plt.imshow(screen_, interpolation='none')
    plt.title('CartPole State')
    plt.axis('off')
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


# training loop
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # 将一个批次的转换解压为一个批次的状态，一个批次的动作，一个批次的下一个状态，一个批次的奖励
    batch = Transition(*zip(*transitions))

    # 首先计算一个批次的非终止状态的掩码，即next_state不为None就为True，否则为False
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)
    # 提取了这个批次中所有next_state不为None的转换，然后将其拼接成一个tensor
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 将当前状态输入policy_net，得到每个状态对应的两个动作的Q值
    # 再使用gather函数，将action_batch中的动作对应的Q值提取出来
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 计算下一个状态的Q值，首先初始化为0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 预期的值是下一个状态的Q值乘以折扣因子再加上奖励 r + gamma * max_a' Q(s', a')
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 计算损失，使用smooth_l1_loss函数，即Huber Loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 优化模型，首先将梯度置为0，然后进行反向传播，最后进行梯度裁剪
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # 使用优化器来更新模型参数，使得梯度下降
    optimizer.step()


if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 600

for i_episode in range(num_episodes):
    # 初始化环境并获取初始状态
    # 按照官方文档的说法可以设置初始状态的生成范围
    state, info = env.reset()
    # unsqueeze(0) 将原始状态数据变成了一个形状为 (1, ...) 的张量，其中 1 表示批次大小
    # 便于后续的模型输入
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    # 这是一种无限循环的写法，直到任务终止才会退出循环
    for t in count():
        # 根据状态选择动作
        action = select_action(state)
        # step函数执行动作，返回新的状态、奖励、是否终止、是否被截断等信息
        print(action)
        print(type(action))
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        # done是终止或者被截断的标志
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 存储状态转换信息到经验回放缓冲区
        memory.push(state, action, next_state, reward)

        # 切换到下一个状态
        state = next_state

        # 执行一次模型优化（policy网络的训练）
        # 取样一个批次的转换，然后进行模型优化
        optimize_model()

        # new
        # 获取当前环境的屏幕状态
        screen = env.render()
        plot_cartpole_state(screen)

        # 软更新目标网络的权重
        # 这里的软更新是指将目标网络的权重向策略网络的权重靠近一点
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
# 结束后关闭环境
env.close()

# 保存模型
print("Saving model...")
# 创建文件夹名为model
if not os.path.exists('./model'):
    os.mkdir('./model')
# 保存模型参数
torch.save(policy_net.state_dict(), './model/policy_net.pth')
torch.save(target_net.state_dict(), './model/target_net.pth')
print("Model saved!")
