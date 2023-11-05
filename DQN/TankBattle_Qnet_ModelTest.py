import TankBattle_QLearning as TBQ
import random
import torch
from torch import nn
from torch import optim
import numpy as np
from collections import deque, namedtuple  # 队列类型
from servoControl import PCA9685_test
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'Reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, memory_size):
        self.memory = deque([], maxlen=memory_size)

    def sample(self, batch_size):
        batch_data = random.sample(self.memory, batch_size)
        state, action, TBQ.Reward, next_state, done = zip(*batch_data)
        return state, action, TBQ.Reward, next_state, done

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
            nn.Linear(1 * 4, n_observations),
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
        TBQ.Rewards = np.expand_dims(transition_dict.TBQ.Reward, axis=-1)  # 扩充维度
        next_states = transition_dict.next_state
        dones = np.expand_dims(transition_dict.done, axis=-1)  # 扩充维度

        states = np.array(states, dtype=np.float32)
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = np.array(actions, dtype=np.int64)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        TBQ.Rewards = np.array(TBQ.Rewards, dtype=np.int64)
        TBQ.Rewards = torch.tensor(TBQ.Rewards, dtype=torch.int64).to(device)
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
            q_targets = TBQ.Rewards + self.gamma * max_next_q_values * (1 - dones)
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
    Reward_total = 0
    while True:
        # state = state.flatten()
        action = agent.take_action(state)
        next_state, TBQ.Reward, done, _ = env.step(action)
        # print(TBQ.Reward)
        repalymemory.push(state, action, TBQ.Reward, next_state, done)
        Reward_total += TBQ.Reward
        if len(repalymemory) > batch_size:
            state_batch, action_batch, TBQ.Reward_batch, next_state_batch, done_batch = repalymemory.sample(batch_size)
            T_data = Transition(state_batch, action_batch, TBQ.Reward_batch, next_state_batch, done_batch)
            # print(T_data)
            agent.update(T_data)
        state = next_state
        if done:
            break
    return Reward_total

def run_episode_enemy(env, agent, repalymemory, batch_size):
    state = env.reset()
    Reward_total = 0
    while True:
        # state = state.flatten()
        action_enemy = agent.take_action(state)
        next_state, _, done, _ = env.step(action_enemy)
        # print(TBQ.Reward)

        if len(repalymemory) > batch_size:
            state_batch, action_batch, TBQ.Reward_batch, next_state_batch, done_batch = repalymemory.sample(batch_size)
            T_data = Transition(state_batch, action_batch, TBQ.Reward_batch, next_state_batch, done_batch)
            # print(T_data)
            agent.update(T_data)
        state = next_state
        if done:
            break
    return Reward_total


def episode_evaluate(env, agent, render):
    Reward_list = []
    for i in range(5):
        state = env.reset()
        Reward_episode = 0
        while True:
            action = agent.take_action(state)
            next_state, TBQ.Reward, done, _ = env.step(action)
            Reward_episode += TBQ.Reward
            state = next_state
            if done:
                break
            if render:
                env.render()
        Reward_list.append(Reward_episode)
    return np.mean(Reward_list).item()


def take_action_with_model(state, q_net):
    state = torch.tensor(state, dtype=torch.float).to(device)
    action = torch.argmax(q_net(state)).item()
    return action


def test(env, agent, delay_time,agent_enemy):
    state = env.reset()
	
    loaded_q_net_player = Qnet(observation_n, action_n).to(device)
    loaded_q_net_player.load_state_dict(torch.load("train_920.pth"))
    # loaded_q_net_enemy = Qnet(observation_n, action_n).to(device)
    # loaded_q_net_enemy.load_state_dict(torch.load("train_901.pth"))


    totalRew = 0
    obs = env.reset()
    done = False
    Reward_episode = 0
    # while True:
    #     action = agent.take_action(state)
    #     next_state, TBQ.Reward, done, _ = env.step(action)
    #     Reward_episode += TBQ.Reward
    #     state = next_state
    #     if done:
    #         break
    #     env.render()
    #     time.sleep(delay_time)
    while not done:
        action = take_action_with_model(state, loaded_q_net_player)
        state, _, done, _ = env.step(action)



# # 初始化Pygame

if __name__ == "__main__":
    # print("prepare for RL")
    # env = gym.make("CartPole-v0")
    env = TBQ.TankBattleEnv()
    env_name = "test-v0"
    observation_n, action_n = 4, 7
    # print(observation_n)
    # print(observation_n, action_n)
    agent_player = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    agent_enemy = Agent(observation_n, action_n, gamma=0.98, lr=2e-3, epsilon=0.01, target_update=10)
    replaymemory = ReplayMemory(memory_size=10000)
    batch_size = 64

    # pygame.init()
    # width, height = 60, 40
    # screen = pygame.display.set_mode((width, height))
    # clock = pygame.time.Clock()

    num_episodes = 200
    Reward_list = []
    # print("start to train model")
    # 显示10个进度条


    state = env.reset()
    obs = env.reset()
    done = False
    PCA9685_test.stand()
    time.sleep(2)
    test(env, agent_player, 0.1, agent_enemy)  # 最后用动画观看一下效果

