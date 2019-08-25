#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
"""

# 生成10组起点和终点
# 生成10个任意的转角

import random
import numpy as np
import math

d     = 15.0  # 正方形边长的一半
d_min = 20.0  # 起点和终点之间的最小距离

# 随机种子
random.seed(1000)
np.random.seed(1000)

Start_point_x = []
Start_point_y = []
Goal_point_x  = []
Goal_point_y  = []

# 随机初始化起点和终点的位置
for i in range(10):
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

np.savetxt('12_start_point_x.txt', start_point_x, delimiter=',')
np.savetxt('12_start_point_y.txt', start_point_y, delimiter=',')
np.savetxt('12_goal_point_x.txt', goal_point_x, delimiter=',')
np.savetxt('12_goal_point_y.txt', goal_point_y, delimiter=',')

randangle = 2 * math.pi * np.random.random_sample((10,1)) - math.pi
np.savetxt('12_randangle.txt', randangle, delimiter=',')