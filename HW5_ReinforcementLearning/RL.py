import copy

import gym


class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow

        self.P = self.createP()

    def createP(self):
        ##
        P = [[[] for i in range(4)] for i in range(self.nrow * self.ncol)]
        # 对应action_meaning
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:
                        # 一个大小为四的数组，四个意义分别代表
                        # 转移概率， 下一步的状态点(当前condition下为不动，即本省)， 反馈值， 是否结束
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 对于某一个策略，下一步的移动目标为next_state
                    # change 为 （x, y）
                    # min 与 max 用于限定下一个状态不出当前边界
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置是悬崖或者重点
                    if next_y == (self.nrow - 1) and next_x > 0:
                        # 设定为最下面一排中间一排是悬崖，现在已经到了最下面一排并且不在最一开始的地方
                        done = True
                        if next_x != self.ncol - 1:
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration:

    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        # 状态价值矩阵，每一个格子的四种策略都有值
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]
        self.theta = theta
        self.gamma = gamma

    def policy_improvement(self):
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            # 对于每个点，计算当前节点的四个策略合并的下一步的状态权重
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)

            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print("策略提升完成")
        return self.pi

    def policy_evaluation(self):
        cnt = 1
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("策略评估迭代" + str(cnt) + "轮后完成")

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)
            new_pi = self.policy_improvement()
            if old_pi == new_pi:
                break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print("%6.6s" % ("%.3f" % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print("****", end=' ')
            elif (i * agent.env.ncol + j) in end:
                print(" End", end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else '_'
                print(pi_str, end=' ')
        print()

"""env = CliffWalkingEnv()
action_meaning = ['↑', '↓', '←', '→']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print(list(range(37, 47)))
print_agent(agent, action_meaning, list(range(37, 47)), [47])
print(agent.pi)"""

