import gym
from gym import spaces
import numpy as np
import math
import time
import random
import torch
from torch import nn
from torch import optim
import numpy as np

from collections import deque, namedtuple  # 队列类型

from servoControl import PCA9685_test

H = 400
W = 400
global player_reward
player_reward = 0
global enemy_reward
enemy_reward = 0
global last_life
last_life = 0
global Rew
Rew = 0
global StepCnt
StepCnt = 0
StepLenth1 = 10
StepLenth2 = 10
global RayDir
RayDir = [1, 0]
global enemy_RayDir
enemy_RayDir = [1, 0]
global life
life = 5
global player_life
player_life = 5
global PlayerDir
PlayerDir = [1, 0]
global react_action_player
global done_player
global cnt_player
done_player = 1
react_action_player = 0
cnt_player = 0
global react_action_enemy
global done_enemy
global cnt_enemy
done_enemy = 1
react_action_enemy = 0
cnt_enemy = 0


class TankBattleEnv(gym.Env):
    def __init__(self):
        super(TankBattleEnv, self).__init__()
        self.action_space = spaces.Discrete(7)  # 5个动作：0-不动，1-向前移动，2-向后移动，3-向左旋转，4-向右旋转 5-向右移动 6-向左移动
        self.observation_space = spaces.Box(low=0, high=255, shape=(W, H, 3), dtype=np.uint8)  # 400x400的RGB图像

        self.player_position = [200, 200]  # 初始玩家位置
        self.enemy_position = random.choice([[300, 100], [100, 100], [300, 300], [100, 300]])  # 初始敌人位置
        self.player_life = 10
        self.enemy_life = 10
        self.player_dir = [-1.0, 0]
        self.enemy_dir = [1, 0]
        self.enemy_is_appear = 0
        self.player_is_appear = 0
        self.target_dis = -1
        self.target_size = -1
        global last_life
        last_life = self.player_life

    def reset(self):
        self.player_position = [200, 200]  # 初始玩家位置
        #self.enemy_position =[100,100]
        self.enemy_position= random.choice([[300, 100], [100, 100], [300, 300], [100, 300]])  # 初始敌人位置
        self.player_life = 10
        self.enemy_life = 10
        self.player_dir = [-1.0, 0]
        self.enemy_dir = [1, 0]
        self.enemy_is_appear = 0
        self.player_is_appear = 0
        self.target_dis = -1
        self.target_size = -1
        global StepCnt
        StepCnt = 0
        global last_life
        last_life = self.player_life
        return self._get_observation()[1].flatten()

    def step(self, action):

        self._move_player(action)


        global player_life
        player_life = self.player_life
        global done
        done = False

        if self.player_life == 0:
            done = True

        return self._get_observation()[1].flatten(), player_reward, done, {}

    # 5个动作：0-不动，1-向前移动，2-向后移动，3-向左旋转，4-向右旋转
    def _move_player(self, action):
        global RayDir
        if action == 0:
            PCA9685_test.stand()
            print('stand')

        elif action == 1:
            PCA9685_test.forward()
            print('forward')
        elif action == 2:
            PCA9685_test.retreat()
            print('retreat')
        elif action == 3:
            PCA9685_test.turn_left()
            print('turn_left')
        elif action == 4:
            PCA9685_test.turn_right()
            print('turn_right')
        elif action == 5:
            PCA9685_test.move_left()
            time.sleep(0.3)
            PCA9685_test.move_left()
            print('move_left')
        elif action == 6:
            PCA9685_test.move_right()
            time.sleep(0.3)
            PCA9685_test.move_right()
            print('move_right')
        time.sleep(0.5)
        return 0



    def _get_observation(self):
        observation = np.zeros((W, H, 3), dtype=np.uint8)
        observation[self.player_position[0], self.player_position[1], :] = [0, 255, 0]  # 玩家用绿色表示
        observation[self.enemy_position[0], self.enemy_position[1], :] = [255, 0, 0]  # 敌人用红色表示

        # state_info = [self.player_position[0], self.player_position[1], self.enemy_position[0], self.enemy_position[1], self.enemy_life, self.player_life, self.player_dir[0], self.player_dir[1]]
        with open(r'../yolov3/runs/detect/ceshi/labels/0_ceshi.txt', 'r') as file:
            lines = file.readlines()

            if lines:

                last_line = lines[-1].strip()  # 获取最后一行并去除换行符
                last_line_arr = last_line.split(" ")
                print(f'Last line: {last_line_arr}', type(last_line_arr))
                self.enemy_is_appear = int(last_line_arr[0]) + 1
                if self.enemy_is_appear == 1:
                    self.target_dis = float(last_line_arr[1])-0.5
                    self.target_size = float(last_line_arr[3])*float(last_line_arr[4])
                else:
                    self.target_dis = -1
                    self.target_size = -1

            else:
                print('File is empty')  # 文件为空时的处理

        state_info = [self.enemy_is_appear, self.target_dis, self.target_size,
                      self.player_life]
        state_info = np.array(state_info)

        return [observation, state_info]
