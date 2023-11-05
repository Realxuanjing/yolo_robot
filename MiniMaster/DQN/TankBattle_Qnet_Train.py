from DQN import TankBattle_QLearning as TBQ
import gym
from gym import spaces
import numpy as np
import pygame
import math
import time
import random
from DQN import TankBattle_QLearning as TBQ
import torch
from torch import nn
from torch import optim
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, namedtuple  # 队列类型
from tqdm import tqdm  # 绘制进度条用
import pygame

pygame.init()
font1 = pygame.font.SysFont('None', 15)  # 得分区域字体显示黑体24
score_color = (205, 193, 180)


def print_text(screen, font, x, y, text):
    imgText = font.render(text, True, score_color)
    screen.blit(imgText, (x, y))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        state, action, TBQ.reward, next_state, done = zip(*batch_data)
        return state, action, TBQ.reward, next_state, done

    def push(self, *args):
        # *args: 把传进来的所有参数都打包起来生成元组形式
        # self.push(1, 2, 3, 4, 5)
        # args = (1, 2, 3, 4, 5)
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)


class Qnet(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(Qnet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1 * 4, n_observations),######################################################################
            nn.ReLU(),
            nn.Linear(n_observations, n_observations),
            nn.ReLU(),
            nn.Linear(n_observations, n_observations),
            nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            nn.Linear(n_observations, n_actions)
        )

    def forward(self, state):
        return self.model(state)


class Agent(object):

    def __init__(self, observation_dim, action_dim, gamma, lr, epsilon, target_update):
        self.action_dim = action_dim
        self.q_net = Qnet(observation_dim, action_dim).to(device)
        self.target_q_net = Qnet(observation_dim, action_dim).to(device)
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

        self.optimizer = optim.Adam(params=self.q_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def take_action(self, state):
        if np.random.uniform(0, 1) < 1 - self.epsilon:
            state = torch.tensor(state, dtype=torch.float).to(device)
            action = torch.argmax(self.q_net(state)).item()
        else:
            action = np.random.choice(self.action_dim)
        return action

    def update(self, transition_dict):

        states = transition_dict.state
        actions = np.expand_dims(transition_dict.action, axis=-1)  # 扩充维度
        TBQ.rewards = np.expand_dims(transition_dict.reward, axis=-1)  # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)  # 扩充维度

        states = np.array(states, dtype=np.float32)
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = np.array(actions, dtype=np.int64)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        TBQ.rewards = np.array(TBQ.rewards, dtype=np.int64)
        TBQ.rewards = torch.tensor(TBQ.rewards, dtype=torch.int64).to(device)
        next_states = np.array(next_states, dtype=np.float32)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = np.array(dones, dtype=np.int64)
        dones = torch.tensor(dones, dtype=torch.int64).to(device)

        # update q_values
        # gather(1, acitons)意思是dim=1按行号索引， index=actions
        # actions=[[1, 2], [0, 1]] 意思是索引出[[第一行第2个元素， 第1行第3个元素],[第2行第1个元素， 第2行第2个元素]]
        # 相反，如果是这样
        # gather(0, acitons)意思是dim=0按列号索引， index=actions
        # actions=[[1, 2], [0, 1]] 意思是索引出[[第一列第2个元素， 第2列第3个元素],[第1列第1个元素， 第2列第2个元素]]
        # states.shape(64, 4) actions.shape(64, 1), 每一行是一个样本，所以这里用dim=1很合适
        # print(states.shape)
        predict_q_values = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            # max(1) 即 max(dim=1)在行向找最大值，这样的话shape(64, ), 所以再加一个view(-1, 1)扩增至(64, 1)
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
            q_targets = TBQ.rewards + self.gamma * max_next_q_values * (1 - dones)
        l = self.loss(predict_q_values, q_targets)

        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # copy model parameters
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1


def run_episode(env, agent, repalymemory, batch_size):
    state = env.reset()
    reward_total = 0
    while True:
        # state = state.flatten()
        action = agent.take_action(state)
        next_state, TBQ.reward, done, _ = env.step(action)
        # print(reward)
        repalymemory.push(state, action, TBQ.reward, next_state, done)
        reward_total += TBQ.reward
        if len(repalymemory) > batch_size:
            state_batch, action_batch, TBQ.reward_batch, next_state_batch, done_batch = repalymemory.sample(batch_size)
            T_data = Transition(state_batch, action_batch, TBQ.reward_batch, next_state_batch, done_batch)
            # print(T_data)
            agent.update(T_data)
        state = next_state
        if done:
            break
    return reward_total


def episode_evaluate(env, agent, render):
    reward_list = []
    for i in range(5):
        state = env.reset()
        reward_episode = 0
        while True:
            action = agent.take_action(state)
            next_state, TBQ.reward, done, _ = env.step(action)
            reward_episode += TBQ.reward
            state = next_state
            if done:
                break
            if render:
                env.render()
        reward_list.append(reward_episode)
    return np.mean(reward_list).item()


def take_action_with_model(state, q_net):
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.argmax(q_net(state)).item()
    return action


def test(env, agent, delay_time):
    state = env.reset()
    pygame.init()
    width, height = 600, 400
    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()
    loaded_q_net_player = Qnet(observation_n, action_n).to(device)
    loaded_q_net_player.load_state_dict(torch.load("train_920.pth"))

    totalRew = 0
    obs = env.reset()
    done = False
    reward_episode = 0
    # while True:
    #     action = agent.take_action(state)
    #     next_state, reward, done, _ = env.step(action)
    #     reward_episode += reward
    #     state = next_state
    #     if done:
    #         break
    #     env.render()
    #     time.sleep(delay_time)
    while not done:
        screen.fill((0, 0, 0))  # 清空屏幕

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # 渲染环境状态
        obs_rgb = env._get_observation()[0]  # 获取RGB图像状态
        obs_rgb_resized = pygame.surfarray.make_surface(obs_rgb)  # 旋转并转换为Pygame Surface
        player_position = env.player_position
        enemy_position = env.enemy_position

        player_x, player_y = player_position[0], player_position[1]
        enemy_x, enemy_y = enemy_position[0], enemy_position[1]
        pygame.draw.circle(obs_rgb_resized, (0, 255, 0), (player_x, player_y), 20)  # 绘制更大的绿色点

        pygame.draw.circle(obs_rgb_resized, (255, 0, 0), (enemy_x, enemy_y), 20)  # 绘制更大的红色点

        pygame.draw.line(obs_rgb_resized, (0, 255, 0), (player_x, player_y),
                         (player_x + TBQ.RayDir[0] * 1000, player_y + TBQ.RayDir[1] * 1000), 3)
        pygame.draw.line(obs_rgb_resized, (255, 0, 0), (enemy_x, enemy_y),
                         (enemy_x + TBQ.enemy_RayDir[0] * 1000, enemy_y + TBQ.enemy_RayDir[1] * 1000), 3)
        # print(RayDir)
        totalRew += TBQ.Rew
        screen.blit(obs_rgb_resized, (0, 0))
        print_text(screen, font1, 500, 10, f'{TBQ.life}')
        print_text(screen, font1, 500, 30, f'{totalRew}')
        print_text(screen, font1, 500, 50, f'{TBQ.player_life}')
        pygame.display.flip()
        clock.tick(10)  # 控制帧率

        action = take_action_with_model(state, loaded_q_net_player)
        state, _, done, _ = env.step(action)

    pygame.quit()


# # 初始化Pygame


if __name__ == "__main__":

    # print("prepare for RL")
    # env = gym.make("CartPole-v0")
    env = TBQ.TankBattleEnv()
    env_name = "test-v0"
    observation_n, action_n = 4, 7####################################################################################
    # print(observation_n)
    # print(observation_n, action_n)
    agent = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    agent.q_net.load_state_dict(torch.load("train_920.pth"))
    replaymemory = ReplayMemory(memory_size=10000)
    batch_size = 64

    # pygame.init()
    # width, height = 60, 40
    # screen = pygame.display.set_mode((width, height))
    # clock = pygame.time.Clock()

    num_episodes = 1000
    reward_list = []

    best_performance = float('-inf')
    best_q_net_state_dict = None
    # print("start to train model")
    # 显示10个进度条
    for i in range(5):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for episode in range(int(num_episodes / 10)):

                reward_episode = run_episode(env, agent, replaymemory, batch_size)

                if (episode + 1) % 10 == 0:
                    test_reward = episode_evaluate(env, agent, False)
                    reward_list.append(test_reward)

                    # 更新最佳性能和模型
                    if test_reward > best_performance:
                        best_performance = test_reward
                        best_q_net_state_dict = agent.q_net.state_dict()
                        # 保存效果最好的模型
                        torch.save(best_q_net_state_dict, "train_920.pth")

                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + episode + 1),
                        'return': '%.3f' % (test_reward)
                    })
                pbar.update(1)  # 更新进度条
    #torch.save(agent.q_net.state_dict(), "train_920.pth")


    episodes_list = list(range(len(reward_list)))
    plot_list = list(range(int(len(reward_list)/10)))
    print(plot_list)
    for i in range(len(plot_list)):
        for j in range(0, 9):
            plot_list[i] += reward_list[i * 10 + j]/10
    episodes_list = list(range(int(len(plot_list))))
    print(episodes_list, plot_list)
    plt.plot(episodes_list, plot_list)
    # plt.xlabel('Episodes')
    plt.xlabel('Episodes * 10')
    plt.ylabel('Returns')
    plt.title('Double DQN on {}'.format(env_name))
    plt.show()
    test(env, agent, 0.1)  # 最后用动画观看一下效果