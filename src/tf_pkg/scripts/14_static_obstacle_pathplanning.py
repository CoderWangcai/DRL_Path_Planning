#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
"""

# 生成500组任意的起点和终点
# 生成500组（10个一组）随机静态障碍
# 生成500个任意的转角

import random
import numpy as np
import math

# 随机初始化障碍的位置
def random_square(a):
    obstacle_robot_position = []
    x1, y1 = -a, -a
    x2, y2 = a, -a
    x3, y3 = -a, a
    x4, y4 = a, a

    # 初始化
    for i in range(num_obs):
        obstacle_robot_position.append([0.0, 0.0]) 

    random_list = np.random.random_sample(num_obs)
    for i in range(num_obs):
        if random_list[i] >= 0.5:
            rnd11 = np.random.random()
            rnd21 = np.random.random()
            rnd21 = np.sqrt(rnd21)
            obstacle_robot_position[i][0]=rnd21 * (rnd11 * x1 + (1 - rnd11) * x2) + (1 - rnd21) * x3
            obstacle_robot_position[i][1]=rnd21 * (rnd11 * y1 + (1 - rnd11) * y2) + (1 - rnd21) * y3
        else:
            rnd12 = np.random.random()
            rnd22 = np.random.random()
            rnd22 = np.sqrt(rnd22)
            obstacle_robot_position[i][0]=rnd22 * (rnd12 * x3 + (1 - rnd12) * x4) + (1 - rnd22) * x2
            obstacle_robot_position[i][1]=rnd22 * (rnd12 * y3 + (1 - rnd12) * y4) + (1 - rnd22) * y2
    return obstacle_robot_position

d        = 15.0  # 正方形边长的一半
d_min    = 20.0  # 起点和终点之间的最小距离
num_obs  = 10    # 障碍的数量
num_test = 500   # 测试的组数

# 保存起点和终点的列表
Start_point_x = []
Start_point_y = []
Goal_point_x  = []
Goal_point_y  = []

# 保存障碍
obs_pos_x     = []
obs_pos_y     = []
for i in range(num_test):
    obs_pos_x.append([])
    obs_pos_y.append([])
    for j in range(10):
        obs_pos_x[i].append(0)
        obs_pos_y[i].append(0)

# 随机种子
random.seed(1000)
np.random.seed(1000)

# 随机初始化起点和终点的位置
for i in range(num_test):
    while(True):
        randposition = 2 * d * np.random.random_sample((2, 2)) - d
        if math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2) > d_min:
            # print("start_x={},start_y={},goal_x={},goal_y={},d_sp={}".format(randposition[0][0],randposition[0][1],randposition[1][0],randposition[1][1],math.sqrt((randposition[0][0]-randposition[1][0])**2+(randposition[0][1]-randposition[1][1])**2)))
            Start_point_x.append(randposition[0][0])
            Start_point_y.append(randposition[0][1])
            Goal_point_x.append(randposition[1][0])
            Goal_point_y.append(randposition[1][1])
            break

start_point_x = np.array(Start_point_x)
start_point_y = np.array(Start_point_y)
goal_point_x = np.array(Goal_point_x)
goal_point_y = np.array(Goal_point_y)

np.savetxt('14_static_obstacle_start_point_x.txt', start_point_x, delimiter=',')
np.savetxt('14_static_obstacle_start_point_y.txt', start_point_y, delimiter=',')
np.savetxt('14_static_obstacle_goal_point_x.txt', goal_point_x, delimiter=',')
np.savetxt('14_static_obstacle_goal_point_y.txt', goal_point_y, delimiter=',')

# 随机初始化小车的转角
randangle = 2 * math.pi * np.random.random_sample((num_test,1)) - math.pi
np.savetxt('14_static_obstacle_randangle.txt', randangle, delimiter=',')

#获取障碍的随机初始位置
#障碍物为边长为1m的正方体
for m in range(num_test):
    while(True):
        OBS_pos = random_square(d)
        flag = True  # 用于判断起点和终点是否在障碍物的范围内
        for i in range(num_obs):
            # 如果起点在障碍物周围5.0m的范围内则需要重新生成障碍物
            if math.sqrt((Start_point_x[m]-OBS_pos[i][0])**2+(Start_point_y[m]-OBS_pos[i][1])**2) < 5.0:
                flag = False
            # 如果终点在障碍物周围5.0m的范围内则需要重新生成障碍物
            if math.sqrt((Goal_point_x[m]-OBS_pos[i][0])**2+(Goal_point_y[m]-OBS_pos[i][1])**2) < 5.0:
                flag = False
            # 如果两个障碍相隔太近则需要重新生成障碍物
            for j in range(i + 1, num_obs):
                if math.sqrt((OBS_pos[i][0]-OBS_pos[j][0])**2+(OBS_pos[i][1]-OBS_pos[j][1])**2) < 5.0:
                    flag = False
                    break
        if flag == True:
            break
    for i in range(num_obs):
        obs_pos_x[m][i]=OBS_pos[i][0]
        obs_pos_y[m][i]=OBS_pos[i][1]

OBS_POS_X=np.array(obs_pos_x)
OBS_POS_Y=np.array(obs_pos_y)
np.savetxt('14_static_obstacle_obs_pos_x.txt', OBS_POS_X, delimiter=',')
np.savetxt('14_static_obstacle_obs_pos_y.txt', OBS_POS_Y, delimiter=',')