#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
"""

# 画出路径规划的结果

import random
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl

num_test  = 30  # 测试的组数
gap       = 20   # 画出动态障碍的间隔
# 画出起点和终点，并在终点加上1.0m直径的圆
r = 1.0
for i in range(num_test):
    # 起点
    start_x   = []
    start_y   = []
    # 终点
    goal_x    = []
    goal_y    = []
    # 小车的起始转角
    randangle = []
    # 保存静态障碍
    obs_pos_x     = []
    obs_pos_y     = []
    # 保存动态障碍
    jackal0_x = []
    jackal0_y = []
    jackal1_x = []
    jackal1_y = []
    jackal2_x = []
    jackal2_y = []
    jackal3_x = []
    jackal3_y = []
    jackal4_x = []
    jackal4_y = []
    jackal5_x = []
    jackal5_y = []
    jackal6_x = []
    jackal6_y = []
    jackal7_x = []
    jackal7_y = []
    jackal8_x = []
    jackal8_y = []
    jackal9_x = []
    jackal9_y = []    
    # 读取jackal的数据
    JACKAL_x = []
    JACKAL_y = []
    
    plt.figure(figsize=(10, 10))
    # 读取起点的x
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__start_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Start_x = [float(s) for s in line.split()]
            start_x.append(Start_x[0])
    # 读取起点的y
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__start_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Start_y = [float(s) for s in line.split()]
            start_y.append(Start_y[0])
    # 读取终点的x
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__goal_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Goal_x = [float(s) for s in line.split()]
            goal_x.append(Goal_x[0])
    # 读取终点的y
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__goal_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Goal_y = [float(s) for s in line.split()]
            goal_y.append(Goal_y[0])
    # 画出起点和终点
    plt.plot(start_x, start_y, marker='o',color='lightgreen', markersize=10)
    plt.plot(goal_x, goal_y, marker='*',color='deeppink', markersize=10)
    a, b = (goal_x, goal_y)
    theta = np.arange(0, 2*np.pi, 0.01)
    x = a + r * np.cos(theta)
    y = b + r * np.sin(theta)
    plt.plot(x, y, label='line', color='dodgerblue', linewidth=2.0)
    
    # 读取动态障碍jackal0
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL0_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL0_x = [float(s) for s in line.split()]
            jackal0_x.append(JACKAL0_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL0_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL0_y = [float(s) for s in line.split()]
            jackal0_y.append(JACKAL0_y[0])
    # 画出动态障碍jackal0
    for j in range(len(jackal0_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal0_x[j], jackal0_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='coral', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal0_x[j]-0.25, jackal0_y[j]-0.25, str(j), size = 14, color='coral')
    # plt.plot(jackal0_x, jackal0_y, label='line', color='coral', linewidth=2.0)
    
    # 读取动态障碍jackal1
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL1_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL1_x = [float(s) for s in line.split()]
            jackal1_x.append(JACKAL1_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL1_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL1_y = [float(s) for s in line.split()]
            jackal1_y.append(JACKAL1_y[0])
    # 画出动态障碍jackal1
    for j in range(len(jackal1_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal1_x[j], jackal1_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='cornflowerblue', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal1_x[j]-0.25, jackal1_y[j]-0.25, str(j), size = 14, color='cornflowerblue')

    # 读取动态障碍jackal2
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL2_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL2_x = [float(s) for s in line.split()]
            jackal2_x.append(JACKAL2_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL2_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL2_y = [float(s) for s in line.split()]
            jackal2_y.append(JACKAL2_y[0])
    # 画出静态障碍
    for j in range(len(jackal2_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal2_x[j], jackal2_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='limegreen', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal2_x[j]-0.25, jackal2_y[j]-0.25, str(j), size = 14,color='limegreen')

    # 读取动态障碍jackal3
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL3_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL3_x = [float(s) for s in line.split()]
            jackal3_x.append(JACKAL3_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL3_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL3_y = [float(s) for s in line.split()]
            jackal3_y.append(JACKAL3_y[0])
    # 画出静态障碍
    for j in range(len(jackal3_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal3_x[j], jackal3_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='red', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal3_x[j]-0.25, jackal3_y[j]-0.25, str(j), size = 14, color='red')

    # 读取动态障碍jackal4
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL4_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL4_x = [float(s) for s in line.split()]
            jackal4_x.append(JACKAL4_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL4_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL4_y = [float(s) for s in line.split()]
            jackal4_y.append(JACKAL4_y[0])
    # 画出静态障碍
    for j in range(len(jackal4_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal4_x[j], jackal4_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='orchid', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal4_x[j]-0.25, jackal4_y[j]-0.25, str(j), size = 14,color='orchid')

    # 读取动态障碍jackal5
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL5_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL5_x = [float(s) for s in line.split()]
            jackal5_x.append(JACKAL5_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL5_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL5_y = [float(s) for s in line.split()]
            jackal5_y.append(JACKAL5_y[0])
    # 画出障碍
    for j in range(len(jackal5_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal5_x[j], jackal5_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='olivedrab', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal5_x[j]-0.25, jackal5_y[j]-0.25, str(j), size = 14,color='olivedrab')

    # 读取动态障碍jackal6
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL6_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL6_x = [float(s) for s in line.split()]
            jackal6_x.append(JACKAL6_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL6_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL6_y = [float(s) for s in line.split()]
            jackal6_y.append(JACKAL6_y[0])
    # 画出障碍
    for j in range(len(jackal6_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal6_x[j], jackal6_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='gray', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal6_x[j]-0.25, jackal6_y[j]-0.25, str(j), size = 14,color='gray')

    # 读取动态障碍jackal7
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL7_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL7_x = [float(s) for s in line.split()]
            jackal7_x.append(JACKAL7_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL7_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL7_y = [float(s) for s in line.split()]
            jackal7_y.append(JACKAL7_y[0])
    # 画出障碍
    for j in range(len(jackal7_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal7_x[j], jackal7_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='lightcoral', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal7_x[j]-0.25, jackal7_y[j]-0.25, str(j), size = 14,color='lightcoral')

    # 读取动态障碍jackal8
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL8_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL8_x = [float(s) for s in line.split()]
            jackal8_x.append(JACKAL8_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL8_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL8_y = [float(s) for s in line.split()]
            jackal8_y.append(JACKAL8_y[0])
    # 画出障碍
    for j in range(len(jackal8_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal8_x[j], jackal8_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='peru', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal8_x[j]-0.25, jackal8_y[j]-0.25, str(j), size = 14,color='peru')

    # 读取动态障碍jackal9
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL9_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL9_x = [float(s) for s in line.split()]
            jackal9_x.append(JACKAL9_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL9_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            JACKAL9_y = [float(s) for s in line.split()]
            jackal9_y.append(JACKAL9_y[0])
    # 画出障碍
    for j in range(len(jackal9_x)):
        if j%gap==0:
            # 障碍周围有1.5m的影响球半径
            a, b = (jackal9_x[j], jackal9_y[j])
            theta = np.arange(0, 2*np.pi, 0.01)
            x = a + 1.5 * np.cos(theta)
            y = b + 1.5 * np.sin(theta)
            plt.plot(x, y, label='line', color='purple', linewidth=2.0)
        if j%gap==0:
            plt.text(jackal9_x[j]-0.25, jackal9_y[j]-0.25, str(j), size = 14,color='purple')

    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL_PATH_x'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Jackal_x = [float(s) for s in line.split()]
            JACKAL_x.append(Jackal_x[0])
    filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test__JACKAL_PATH_y'+'_'+str(i+1)+'.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            Jackal_y = [float(s) for s in line.split()]
            JACKAL_y.append(Jackal_y[0])
    plt.plot(JACKAL_x, JACKAL_y, label='line', color='gold', linewidth=2.0)
    # 画出jackal
    for j in range(len(JACKAL_x)):
        if j%gap==0:
            plt.text(JACKAL_x[j]-0.25, JACKAL_y[j]-0.25, str(j), size = 14,color='k')
    # 设置坐标轴范围
    plt.xlim((-18, 18))
    plt.ylim((-18, 18))
    # 设置坐标轴刻度
    plt.xticks(np.arange(-18, 24, 6))
    plt.yticks(np.arange(-18, 24, 6))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.grid()

    ax = plt.gca() # gca = 'get current axis' 获取当前坐标
    ax.spines['bottom'].set_linewidth(5)
    ax.spines['left'].set_linewidth(5)
    ax.spines['top'].set_linewidth(5)
    ax.spines['right'].set_linewidth(5)
    
    # plt.show()
    # 保存图片
    plt.savefig(".../DRL_Path_Planning/src/tf_pkg/scripts/test_dynamic_obstacle_world_30m_results/image_add_sensor/10jackal/D3QN_PER_image_add_sensor_dynamic_10obstacle_world_30m_test_pathplanning_results_"+str(i+1)+".png")