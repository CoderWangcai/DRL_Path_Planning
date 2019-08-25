#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
"""

import rospy
from std_msgs.msg import String
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan

import math
import numpy as np
import time
from Models import *
from pathplaner import *
import matplotlib.pyplot as plt
import os
import shutil
import random
import tensorflow as tf
import datetime
import cv2
from cv_bridge import CvBridge, CvBridgeError


MAXENVSIZE  = 30.0  # 边长为30的正方形作为环境的大小
MAXLASERDIS = 10.0  # 雷达最大的探测距离

class envmodel():

    def __init__(self,):
        rospy.init_node('planningnode', anonymous=True)

        self.dict_name_id = {'jackal0':0,'jackal1':1,'jackal2':2,'jackal3':3,'jackal4':4,'jackal5':5,'jackal6':6,'jackal7':7,'jackal8':8,'jackal9':9,'jackal10':10}

        self.dict_id_name = {0:'jackal0',1:'jackal1',2:'jackal2',3:'jackal3',4:'jackal4',5:'jackal5',6:'jackal6',7:'jackal7',8:'jackal8',9:'jackal9',10:'jackal10'}

        # 动态障碍机器人的名字列表
        self.obs_robot_namelist = ['jackal0','jackal1','jackal2','jackal3','jackal4','jackal5','jackal6','jackal7','jackal8','jackal9']
        
        # agent机器人的名字
        self.agentrobot = 'jackal10'
        
        self.img_size = 80  # 图片resize后的大小

        self.dis      = 1.0  # 位置精度-->判断是否到达目标的距离

        self.bridge                = CvBridge()
        self.image_matrix          = []
        self.image_matrix_callback = []
        
        # 动态障碍的个数
        self.num_obs = 10

        # 储存动态障碍机器人每次的目标点 
        self.dynamic_obs_goal_x = np.zeros(self.num_obs)
        self.dynamic_obs_goal_y = np.zeros(self.num_obs)
        # 储存动态障碍机器人每次的起点
        self.dynamic_obs_start_pos = np.zeros([self.num_obs, 2])
        
        # 储存障碍机器人的控制指令
        self.pub_obs = []
        # 给障碍机器人发布控制指令
        for i in range(len(self.obs_robot_namelist)):
            pub_obs = rospy.Publisher('/' + self.obs_robot_namelist[i] + '/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
            self.pub_obs.append(pub_obs)
        
        # 发布控制指令给agent robot
        self.pub = rospy.Publisher('/' + self.agentrobot + '/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)

        # 接收agent robot的前视bumblebee相机消息
        self.subimage = rospy.Subscriber('/' + self.agentrobot +'/front/left/image_raw', Image, self.image_callback)

        # 接收agent robot的激光雷达信息
        self.subLaser = rospy.Subscriber('/' + self.agentrobot +'/front/scan', LaserScan, self.laser_states_callback)

        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        
        self.gazebo_model_states = ModelStates()
        
        # 人工势场法相关参数
        # 慢速
        # self.V = 0.2
        # 快速
        self.V = 0.4
        self.planer  = CPFpathplaner()
        # self.robs =  [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        self.robs =  [1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]
        # 初始化的障碍位置
        self.xobs  = np.zeros(9)
        self.yobs  = np.zeros(9)
        self.vxobs = np.zeros(9)
        self.vyobs = np.zeros(9)

        self.resetval()
    
    def resetval(self,):
        # agent机器人的相关参数
        self.robotstate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # x,y,v,w,yaw,vx,vy
        self.d          = 0.0                                  # 到目标的距离
        self.d_last     = 0.0                                  # 前一时刻到目标的距离
        self.v_last     = 0.0                                  # 前一时刻的速度
        self.w_last     = 0.0                                  # 前一时刻的角速度
        self.r          = 0.0                                  # 奖励
        self.cmd        = [0.0, 0.0]                           # agent robot的控制指令
        self.done_list  = False                                # episode是否结束的标志

        # 障碍机器人的相关参数
        self.obs_robot_state = []  # obs_robot_state--->x,y,v,w,yaw,vx,vy
        self.obs_d           = []  # 障碍机器人到各自目标的距离
        for _ in range(len(self.obs_robot_namelist)):
            self.obs_robot_state.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x,y,v,w,yaw,vx,vy
            self.obs_d.append(0.0)

    def sign(self, x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def gazebo_states_callback(self, data):
        self.gazebo_model_states = data
        # name: ['ground_plane', 'jackal1', 'jackal2', 'jackal0',...]
        for i in range(len(data.name)):
            # 障碍机器人
            if data.name[i] in self.obs_robot_namelist:
                # obs_robot_state--->x,y,v,w,yaw,vx,vy
                # data.name[i]='jackal0'
                self.obs_robot_state[self.dict_name_id[data.name[i]]][0] = data.pose[i].position.x
                self.obs_robot_state[self.dict_name_id[data.name[i]]][1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.obs_robot_state[self.dict_name_id[data.name[i]]][2] = v
                self.obs_robot_state[self.dict_name_id[data.name[i]]][3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.obs_robot_state[self.dict_name_id[data.name[i]]][4] = rpy[2]
                self.obs_robot_state[self.dict_name_id[data.name[i]]][5] = data.twist[i].linear.x
                self.obs_robot_state[self.dict_name_id[data.name[i]]][6] = data.twist[i].linear.y
            # agent机器人
            if data.name[i] == self.agentrobot:
                # robotstate--->x,y,v,w,yaw,vx,vy
                self.robotstate[0] = data.pose[i].position.x
                self.robotstate[1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.robotstate[2] = v
                self.robotstate[3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.robotstate[4] = rpy[2]
                self.robotstate[5] = data.twist[i].linear.x
                self.robotstate[6] = data.twist[i].linear.y
    
    def image_callback(self, data):
        try:
            self.image_matrix_callback = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)
    
    def laser_states_callback(self, data):
        self.laser = data

    def quaternion_from_euler(self, r, p, y):
        q = [0, 0, 0, 0]
        q[3] = math.cos(r / 2) * math.cos(p / 2) * math.cos(y / 2) + \
            math.sin(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[0] = math.sin(r / 2) * math.cos(p / 2) * math.cos(y / 2) - \
            math.cos(r / 2) * math.sin(p / 2) * math.sin(y / 2)
        q[1] = math.cos(r / 2) * math.sin(p / 2) * math.cos(y / 2) + \
            math.sin(r / 2) * math.cos(p / 2) * math.sin(y / 2)
        q[2] = math.cos(r / 2) * math.cos(p / 2) * math.sin(y / 2) - \
            math.sin(r / 2) * math.sin(p / 2) * math.cos(y / 2)
        return q

    def euler_from_quaternion(self, x, y, z, w):
        euler = [0, 0, 0]
        Epsilon = 0.0009765625
        Threshold = 0.5 - Epsilon
        TEST = w * y - x * z
        if TEST < -Threshold or TEST > Threshold:
            if TEST > 0:
                sign = 1
            elif TEST < 0:
                sign = -1
            euler[2] = -2 * sign * math.atan2(x, w)
            euler[1] = sign * (math.pi / 2.0)
            euler[0] = 0
        else:
            euler[0] = math.atan2(2 * (y * z + w * x),
                                  w * w - x * x - y * y + z * z)
            euler[1] = math.asin(-2 * (x * z - w * y))
            euler[2] = math.atan2(2 * (x * y + w * z),
                                  w * w + x * x - y * y - z * z)

        return euler

    # 获取agent机器人的回报值
    def getreward(self):
        reward = 0

        # 假如上一时刻到目标的距离<这一时刻到目标的距离就会有负的奖励
        if self.d_last < self.d:
            reward = reward - 0.1*(self.d-self.d_last)
        
        if self.d_last >= self.d:
            reward = reward + 0.1*(self.d_last - self.d)
        
        # 速度发生变化就会有负的奖励
        reward = reward - 0.01*(abs(self.w_last - self.cmd[1]) + abs(self.v_last - self.cmd[0])) 
        
        # 到达目标点有正的奖励
        if self.d < self.dis:        
            reward = reward + 20
            print("Get 20 points------Goal point!")
        
        # 与动态障碍发生碰撞
        # 动态障碍为边长0.8m的正方体
        for i in range(self.num_obs):
            if math.sqrt((self.robotstate[0]-self.obs_robot_state[i][0])**2 + (self.robotstate[1]-self.obs_robot_state[i][1])**2) < 1.0:
                reward = reward - 1
                print("Get -1 points------Dynamic Obstacle!")
                break

        return reward

    # 重置环境
    def reset_env(self, start=[0.0, 0.0], goal=[10.0, 10.0]):
        self.sp = start
        self.gp = goal
        # 初始点到目标点的距离
        self.d_sg = ((self.sp[0]-self.gp[0])**2 + (self.sp[1]-self.gp[1])**2)**0.5
        # 重新初始化各参数
        self.resetval()

        #获取障碍的随机初始位置
        while(True):
            flag = True
            obs_pos = self.random_square(MAXENVSIZE/2)
            
            # 给不同的障碍赋值
            # ----------------------------------------------------------------------------------------------------
            # 0-9是动态障碍机器人的初始位置
            for obs_robot_id in range(self.num_obs):
                self.dynamic_obs_start_pos[obs_robot_id][0] = obs_pos[obs_robot_id][0]
                self.dynamic_obs_start_pos[obs_robot_id][1] = obs_pos[obs_robot_id][1]
            # 10-19是动态障碍机器人的目标位置
            # print("obs_pos={}".format(obs_pos))
            for j in range(self.num_obs):
                self.dynamic_obs_goal_x[j] = obs_pos[self.num_obs+j][0]
                self.dynamic_obs_goal_y[j] = obs_pos[self.num_obs+j][1]
            # ----------------------------------------------------------------------------------------------------

            # 起点和终点不能在动态障碍机器人的起点5m范围内
            # ----------------------------------------------------------------------------------------------------
            for i in range(len(self.obs_robot_namelist)):
                if math.sqrt((self.sp[0]-self.dynamic_obs_start_pos[i][0])**2+(self.sp[1]-self.dynamic_obs_start_pos[i][1])**2) < 5.0:
                    flag = False
                    break
            # ----------------------------------------------------------------------------------------------------
            
            # 起点和终点不能在动态障碍机器人的终点4m范围内
            # 第1 2 3 4 5次训练保留
            # 第6次训练去掉
            # ----------------------------------------------------------------------------------------------------
            '''
            for i in range(len(self.obs_robot_namelist)):
                if math.sqrt((self.sp[0]-self.dynamic_obs_goal_x[i])**2+(self.sp[1]-self.dynamic_obs_goal_y[i])**2) < 4.0:
                    flag = False
                    break
                if math.sqrt((self.gp[0]-self.dynamic_obs_goal_x[i])**2+(self.gp[1]-self.dynamic_obs_goal_y[i])**2) < 4.0:
                    flag = False
                    break
            '''
            # ----------------------------------------------------------------------------------------------------

            # 动态障碍机器人起点之间的距离要大于n米
            # ----------------------------------------------------------------------------------------------------
            for i in range(len(self.obs_robot_namelist)):
                for j in range(len(self.obs_robot_namelist)):
                    if j!=i:
                        if math.sqrt((self.dynamic_obs_start_pos[i][0]-self.dynamic_obs_start_pos[j][0])**2+(self.dynamic_obs_start_pos[i][1]-self.dynamic_obs_start_pos[j][1])**2)<3.0:
                            flag = False
                            break
                if flag==False:
                    break
            # ----------------------------------------------------------------------------------------------------

            # 动态障碍机器人终点之间的距离要大于n米
            # ----------------------------------------------------------------------------------------------------
            for i in range(len(self.obs_robot_namelist)):
                for j in range(len(self.obs_robot_namelist)):
                    if j!=i:
                        if math.sqrt((self.dynamic_obs_goal_x[i]-self.dynamic_obs_goal_x[j])**2+(self.dynamic_obs_goal_y[i]-self.dynamic_obs_goal_y[j])**2)<3.0:
                            flag = False
                            break
                if flag==False:
                    break
            # ----------------------------------------------------------------------------------------------------

            # 满足所有的条件
            if flag==True:
                break

        rospy.wait_for_service('/gazebo/set_model_state')
        val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        # randomposition = 2 * self.dis * np.random.random_sample((1, 2)) - self.dis
        # agent robot生成一个随机的角度
        randangle = 2 * math.pi * np.random.random_sample(1) - math.pi
        # 根据model name对每个物体的位置初始化 
        state = ModelState()
        for i in range(len(self.gazebo_model_states.name)):
            # 放置动态障碍机器人的终点位置
            for j in range(self.num_obs):
                NAME_GOAL = 'obs_point_goal' + str(j)
                if self.gazebo_model_states.name[i] == NAME_GOAL:
                    state.reference_frame = 'world'
                    state.pose.position.z = 0.0
                    state.model_name = self.gazebo_model_states.name[i]
                    state.pose.position.x = self.dynamic_obs_goal_x[j]
                    state.pose.position.y = self.dynamic_obs_goal_y[j]    
                    val(state)
            # 放置动态障碍机器人
            for m in range(self.num_obs):
                NAME_JACKAL = 'jackal' + str(m)
                # print("NAME_JACKAL={}".format(NAME_JACKAL))
                if self.gazebo_model_states.name[i] == NAME_JACKAL:
                    state.reference_frame = 'world'
                    state.pose.position.z = 0.0
                    state.model_name = self.gazebo_model_states.name[i]
                    state.pose.position.x = self.dynamic_obs_start_pos[m][0]
                    state.pose.position.y = self.dynamic_obs_start_pos[m][1] 
                    val(state)
            # 放置agent机器人的终点位置
            if self.gazebo_model_states.name[i] == "point_goal":
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                state.pose.position.x = self.gp[0]
                state.pose.position.y = self.gp[1]
                val(state)
            # 放置agent机器人
            if self.gazebo_model_states.name[i] == self.agentrobot:
                state.reference_frame = 'world'
                state.pose.position.z = 0.0
                state.model_name = self.gazebo_model_states.name[i]
                rpy = [0.0, 0.0, randangle]
                q = self.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
                state.pose.orientation.x = q[0]
                state.pose.orientation.y = q[1]
                state.pose.orientation.z = q[2]
                state.pose.orientation.w = q[3]
                state.pose.position.x = self.sp[0]
                state.pose.position.y = self.sp[1]
                val(state)
                # 到目标点的距离
                self.d = math.sqrt((state.pose.position.x - self.gp[0])**2 + (state.pose.position.y - self.gp[1])**2)
            
        self.done_list = False  # episode结束的标志
        print("The environment has been reset!")     
        time.sleep(1.0)

        return randangle

    # 随机初始化全部障碍的位置
    def random_square(self, a):
        obstacle_robot_position = 2 * a * np.random.random_sample((2*self.num_obs, 2)) - a
        return obstacle_robot_position

    def get_env(self):
        env_info=[]
        # input2-->agent robot的v,w,d,theta
        selfstate = [0.0, 0.0, 0.0, 0.0]
        # robotstate--->x,y,v,w,yaw,vx,vy
        selfstate[0] = self.robotstate[2]  # v
        selfstate[1] = self.robotstate[3]  # w
        # d代表agent机器人距离目标的位置-->归一化[0,1]
        selfstate[2] = self.d/MAXENVSIZE
        dx = -(self.robotstate[0]-self.gp[0])
        dy = -(self.robotstate[1]-self.gp[1])
        xp = dx*math.cos(self.robotstate[4]) + dy*math.sin(self.robotstate[4])
        yp = -dx*math.sin(self.robotstate[4]) + dy*math.cos(self.robotstate[4])
        thet = math.atan2(yp, xp)
        selfstate[3] = thet/math.pi

        # input1-->雷达信息
        laser = []
        temp = []
        sensor_info = []
        for j in range(len(self.laser.ranges)):
            tempval = self.laser.ranges[j]
            # 归一化处理
            if tempval > MAXLASERDIS:
                tempval = MAXLASERDIS
            temp.append(tempval/MAXLASERDIS)
        laser = temp
        # 将agent robot的input2和input1合并成为一个vector:[input2 input1]

        # env_info.append(laser)
        # env_info.append(selfstate)
        for i in range(len(laser)+len(selfstate)):
            if i<len(laser):
                sensor_info.append(laser[i])
            else:
                sensor_info.append(selfstate[i-len(laser)])
        
        env_info.append(sensor_info)
        #print("The state is:{}".format(state))

        # input1-->相机
        # shape of image_matrix [768,1024,3]
        self.image_matrix = np.uint8(self.image_matrix_callback)
        self.image_matrix = cv2.resize(self.image_matrix, (self.img_size, self.img_size))
        # shape of image_matrix [80,80,3]
        self.image_matrix = cv2.cvtColor(self.image_matrix, cv2.COLOR_RGB2GRAY)
        # shape of image_matrix [80,80]
        self.image_matrix = np.reshape(self.image_matrix, (self.img_size, self.img_size))

        env_info.append(self.image_matrix)
        # print("shape of image matrix={}".format(self.image_matrix.shape))

        # 判断是否终止
        self.done_list = True
        # 是否到达目标点判断
        if self.d > self.dis:
            self.done_list = False  # 不终止
        else:
            self.done_list = True  # 终止
        
        # 是否与动态障碍发生碰撞
        if self.done_list == False:
            for i in range(self.num_obs):
                if math.sqrt((self.robotstate[0]-self.obs_robot_state[i][0])**2+(self.robotstate[1]-self.obs_robot_state[i][1])**2)>=1.0:
                    self.done_list = False
                else:
                    self.done_list = True
                    break         

        env_info.append(self.done_list)

        self.r = self.getreward()
        
        env_info.append(self.r)

        # 小车的坐标
        jackal_x = self.robotstate[0]
        jackal_y = self.robotstate[1]
        
        # 动态障碍的坐标
        jackal0_x = self.obs_robot_state[0][0]
        jackal0_y = self.obs_robot_state[0][1]
        jackal1_x = self.obs_robot_state[1][0]
        jackal1_y = self.obs_robot_state[1][1]
        jackal2_x = self.obs_robot_state[2][0]
        jackal2_y = self.obs_robot_state[2][1]
        jackal3_x = self.obs_robot_state[3][0]
        jackal3_y = self.obs_robot_state[3][1]
        jackal4_x = self.obs_robot_state[4][0]
        jackal4_y = self.obs_robot_state[4][1]
        jackal5_x = self.obs_robot_state[5][0]
        jackal5_y = self.obs_robot_state[5][1]
        jackal6_x = self.obs_robot_state[6][0]
        jackal6_y = self.obs_robot_state[6][1]
        jackal7_x = self.obs_robot_state[7][0]
        jackal7_y = self.obs_robot_state[7][1]
        jackal8_x = self.obs_robot_state[8][0]
        jackal8_y = self.obs_robot_state[8][1]
        jackal9_x = self.obs_robot_state[9][0]
        jackal9_y = self.obs_robot_state[9][1]        

        self.v_last = self.cmd[0]
        self.w_last = self.cmd[1]

        return env_info, jackal_x, jackal_y, jackal0_x, jackal0_y, jackal1_x, jackal1_y, jackal2_x, jackal2_y, jackal3_x, jackal3_y, jackal4_x, jackal4_y, jackal5_x, jackal5_y, jackal6_x, jackal6_y, jackal7_x, jackal7_y, jackal8_x, jackal8_y, jackal9_x, jackal9_y

    def run(self):
        rate = rospy.Rate(10)  # 10hz

        cmd_vel = Twist()
        for obs_robot_id in range(len(self.obs_robot_namelist)):
            # self.obs_robot_state------x,y,v,w,yaw,vx,vy
            self.obs_d[obs_robot_id] = math.sqrt((self.dynamic_obs_goal_x[obs_robot_id]-self.obs_robot_state[obs_robot_id][0])**2+(self.dynamic_obs_goal_y[obs_robot_id]-self.obs_robot_state[obs_robot_id][1])**2)
            if self.obs_d[obs_robot_id] > 0.1:
                qx = (self.dynamic_obs_goal_x[obs_robot_id] - self.obs_robot_state[obs_robot_id][0]) / self.obs_d[obs_robot_id]
                qy = (self.dynamic_obs_goal_y[obs_robot_id] - self.obs_robot_state[obs_robot_id][1]) / self.obs_d[obs_robot_id]
                vx = qx * self.V
                vy = qy * self.V
                #*************************CPF path planning*****************************#
                for obs_robot_id2 in range(len(self.obs_robot_namelist)):
                    # 其余4个动态障碍机器人对于某一个动态障碍机器人而言也是障碍
                    # 所以对于某一个动态障碍机器人而言，一共有9个障碍
                    if obs_robot_id2 < obs_robot_id:
                        self.xobs[-1-obs_robot_id2]  = self.obs_robot_state[obs_robot_id2][0]
                        self.yobs[-1-obs_robot_id2]  = self.obs_robot_state[obs_robot_id2][1]
                        self.vxobs[-1-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][5]
                        self.vyobs[-1-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][6]
                    if obs_robot_id2 > obs_robot_id:
                        self.xobs[-obs_robot_id2]  = self.obs_robot_state[obs_robot_id2][0]
                        self.yobs[-obs_robot_id2]  = self.obs_robot_state[obs_robot_id2][1]
                        self.vxobs[-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][5]
                        self.vyobs[-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][6]
                # print("after xobs={}".format(self.xobs))                
                self.planer.get_obs_state(self.xobs, self.yobs, self.robs, self.vxobs, self.vyobs, len(self.xobs))
                self.planer.get_target_state(self.dynamic_obs_goal_x[obs_robot_id], self.dynamic_obs_goal_y[obs_robot_id], 0, 0)
                vx, vy = self.planer.fn_pf_vc(self.obs_robot_state[obs_robot_id][0], self.obs_robot_state[obs_robot_id][1], self.obs_robot_state[obs_robot_id][5], self.obs_robot_state[obs_robot_id][6])
                
                if self.obs_d[obs_robot_id] < 0.5:
                    while True:
                        flag = True

                        rand_position = 2 * (MAXENVSIZE/2) * np.random.random_sample(2) - (MAXENVSIZE/2)
                        self.dynamic_obs_goal_x[obs_robot_id] = rand_position[0]
                        self.dynamic_obs_goal_y[obs_robot_id] = rand_position[1]
            
                        # 动态障碍机器人终点之间的距离要大于n米                
                        for j in range(len(self.obs_robot_namelist)):
                            if j!=obs_robot_id:
                                if math.sqrt((self.dynamic_obs_goal_x[obs_robot_id]-self.dynamic_obs_goal_x[j])**2+(self.dynamic_obs_goal_y[obs_robot_id]-self.dynamic_obs_goal_y[j])**2)<3.0:
                                    flag = False
                                    break
                        
                        # 如果起点和终点在动态障碍机器人的终点4m范围内则需要重新生成障碍物 
                        '''
                        if math.sqrt((self.sp[0]-self.dynamic_obs_goal_x[obs_robot_id])**2+(self.sp[1]-self.dynamic_obs_goal_y[obs_robot_id])**2) < 4.0:
                            flag = False
                        if math.sqrt((self.gp[0]-self.dynamic_obs_goal_x[obs_robot_id])**2+(self.gp[1]-self.dynamic_obs_goal_y[obs_robot_id])**2) < 4.0:
                            flag = False
                        '''
                        
                        if flag==True:
                            break

                    rospy.wait_for_service('/gazebo/set_model_state')
                    val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

                    state = ModelState()
                    for i in range(len(self.gazebo_model_states.name)):
                        # 放置动态障碍机器人的终点位置
                        NAME_GOAL = 'obs_point_goal' + str(obs_robot_id)
                        if self.gazebo_model_states.name[i] == NAME_GOAL:
                            state.reference_frame = 'world'
                            state.pose.position.z = 0.0
                            state.model_name = self.gazebo_model_states.name[i]
                            state.pose.position.x = self.dynamic_obs_goal_x[obs_robot_id]
                            state.pose.position.y = self.dynamic_obs_goal_y[obs_robot_id]    
                            val(state)
                #*************************path planning*****************************#
                yawcmd = math.atan2(vy, vx)

                vcmd = (vx ** 2 + vy ** 2) ** 0.5
                vcmd = limvar(vcmd, -self.V, self.V)
                cmd_vel.linear.x = vcmd

                if yawcmd - self.obs_robot_state[obs_robot_id][4] > math.pi:
                    yawcmd = yawcmd - 2 * math.pi
                elif yawcmd - self.obs_robot_state[obs_robot_id][4] < -math.pi:
                    yawcmd = yawcmd + 2 * math.pi
                wz = 3.0 * (yawcmd - self.obs_robot_state[obs_robot_id][4]) - 0.5 * self.obs_robot_state[obs_robot_id][3]
                if wz > 1.0:
                    wz = 1.0
                elif wz < -1.0:
                    wz = -1.0

                cmd_vel.angular.z = wz
        
            self.pub_obs[obs_robot_id].publish(cmd_vel)
        
        rate.sleep()
    
    def step(self, cmd=[1.0, 0.0]):
        # print("xobs={}".format(self.xobs))
        # print("yobs={}".format(self.yobs))
        self.d_last = math.sqrt((self.robotstate[0] - self.gp[0])**2 + (self.robotstate[1] - self.gp[1])**2)
        self.cmd[0] = cmd[0]
        self.cmd[1] = cmd[1]
        cmd_vel = Twist()
        cmd_vel.linear.x  = cmd[0]
        cmd_vel.angular.z = cmd[1]
        self.pub.publish(cmd_vel)
        
        time.sleep(0.05)

        self.d = math.sqrt((self.robotstate[0] - self.gp[0])**2 + (self.robotstate[1] - self.gp[1])**2)
        self.v_last = cmd[0]
        self.w_last = cmd[1]

if __name__ == '__main__':
    pass
