#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Author: Wangcai
Date: 06/2019
"""

# 环境模型：gazebo_env_D3QN_PER_image_add_sensor_obstacle_world_30m_test.py
# launch文件：one_jackal_image_add_sensor.launch
# world文件：obstacle_sensor.world

# 导入的网络模型：
# .../DRL_Path_Planning/src/tf_pkg/scripts/saved_networks/10_D3QN_PER_image_add_sensor_obstacle_world_30m_2_2019_06_02

# Import modules
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import cv2
import os
import math

from gazebo_env_D3QN_PER_image_add_sensor_obstacle_world_30m_test import envmodel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

env = envmodel()

# 动作指令集---> v,w
action_dict = {0: [1.0, -1.0], 1: [1.0, -0.5], 2: [1.0, 0.0],
               3: [1.0, 0.5], 4: [1.0, 1.0], 5: [0.5, -1.0],
               6: [0.5, 0.0], 7: [0.5, 1.0], 8: [0.0, -1.0],
               9: [0.0, 0.0], 10: [0.0, 1.0]}

class DQN:
    def __init__(self):
        # Algorithm Information
        self.algorithm = 'D3QN_PER'

        # Get parameters
        self.progress = ''
        self.Num_action = len(action_dict)

        # Initial parameters
        self.Num_start_training = 0
        self.Num_training       = 0
        self.Num_test           = 5000

        self.learning_rate      = 0.001
        self.Gamma              = 0.99

        self.Final_epsilon      = 0.1
        # self.Epsilon            = 1.0
        # 第二次训练
        self.Epsilon            = 0.5

        # 训练好的模型
        self.load_path = '.../DRL_Path_Planning/src/tf_pkg/scripts/saved_networks/10_D3QN_PER_image_add_sensor_obstacle_world_30m_2_2019_06_02'

        self.step    = 1
        self.score   = 0
        self.episode = 0

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        # parameters for skipping and stacking
        self.Num_skipFrame    = 1
        self.Num_stackFrame   = 4

        # Parameter for Experience Replay
        self.Num_replay_memory = 5000
        self.Num_batch         = 32
        self.Replay_memory     = []
        
        # Parameters for PER
        self.eps = 0.00001
        self.alpha = 0.6
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.TD_list = np.array([])

        # Parameter for Target Network
        self.Num_update = 500

        # Parameter for LSTM
        self.Num_dataSize  = 364  # 360 sensor size + 4 self state size
        self.Num_cellState = 512

        # Parameters for CNN
        self.first_conv          = [8,8,self.Num_stackFrame, 32]
        self.second_conv         = [4,4,32,64]
        self.third_conv          = [3,3,64,64]
        
        self.first_dense         = [10*10*64+self.Num_cellState, 512]
        self.second_dense_state  = [self.first_dense[1], 1]
        self.second_dense_action = [self.first_dense[1], self.Num_action]

        # Parameters for network
        self.img_size = 80  # input image size

        # Initialize agent robot
        self.agentrobot = 'jackal0'

        # Define the distance from start point to goal point
        self.d = 15.0

        # Define the step for updating the environment
        self.MAXSTEPS    = 300
        # self.MAXEPISODES = 500

        # Initialize Network
        self.output, self.output_target = self.network()
        self.train_step, self.action_target, self.y_target, self.loss_train, self.w_is, self.TD_error = self.loss_and_train()
        self.sess, self.saver = self.init_sess()
	
    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):  # 初始化偏置项
        return tf.Variable(self.xavier_initializer(shape))

    # Xavier Weights initializer
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(2.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    # Convolution function
    def conv2d(self, x, w, stride):  # 定义一个函数，用于构建卷积层
        return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
	
    def assign_network_to_target(self):
        # Get trainable variables
        trainable_variables = tf.trainable_variables()
        # network lstm variables
        trainable_variables_network = [var for var in trainable_variables if var.name.startswith('network')]

        # target lstm variables
        trainable_variables_target = [var for var in trainable_variables if var.name.startswith('target')]

        for i in range(len(trainable_variables_network)):
            self.sess.run(tf.assign(trainable_variables_target[i], trainable_variables_network[i]))

    def network(self):
        tf.reset_default_graph()
        
        # Input------image
        self.x_image = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, self.Num_stackFrame])
        self.x_normalize = (self.x_image - (255.0/2)) / (255.0/2)  # 归一化处理
        
        # Input------sensor
        self.x_sensor = tf.placeholder(tf.float32, shape=[None, self.Num_stackFrame, self.Num_dataSize])
        self.x_unstack = tf.unstack(self.x_sensor, axis=1)

        with tf.variable_scope('network'):
            # Convolution variables
            w_conv1 = self.weight_variable(self.first_conv)  # w_conv1 = ([8,8,4,32]) 
            b_conv1 = self.bias_variable([self.first_conv[3]])  # b_conv1 = ([32])
            
            # second_conv=[4,4,32,64]
            w_conv2 = self.weight_variable(self.second_conv)  # w_conv2 = ([4,4,32,64]) 
            b_conv2 = self.bias_variable([self.second_conv[3]])  # b_conv2 = ([64])
            
            # third_conv=[3,3,64,64]
            w_conv3 = self.weight_variable(self.third_conv)  # w_conv3 = ([3,3,64,64])
            b_conv3 = self.bias_variable([self.third_conv[3]])  # b_conv3 = ([64])
            
            # first_dense=[10x10x64,512]
            w_fc1_1 = self.weight_variable(self.first_dense)  # w_fc1_1 = ([6400,512])
            b_fc1_1 = self.bias_variable([self.first_dense[1]])  # b_fc1_1 = ([512])
            
            w_fc1_2 = self.weight_variable(self.first_dense)  # w_fc1_2 = ([6400,512])
            b_fc1_2 = self.bias_variable([self.first_dense[1]])  # b_fc1_2 = ([512])
            
            # second_dense_state  = [first_dense[1], 1]
            w_fc2_1 = self.weight_variable(self.second_dense_state)  # w_fc2_1 = ([512，1])
            b_fc2_1 = self.bias_variable([self.second_dense_state[1]])  # b_fc2_1 = ([1])
            
            # second_dense_action = [first_dense[1], Num_action]
            w_fc2_2 = self.weight_variable(self.second_dense_action)  # w_fc2_2 = ([512，5])
            b_fc2_2 = self.bias_variable([self.second_dense_action[1]])  # b_fc2_2 = ([5])

            # LSTM cell
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.Num_cellState)            
            rnn_out, rnn_state = tf.nn.static_rnn(inputs=self.x_unstack, cell=cell, dtype=tf.float32)

        h_conv1 = tf.nn.relu(self.conv2d(self.x_normalize, w_conv1, 4) + b_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_conv1, w_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, w_conv3, 1) + b_conv3)
        
        h_pool3_flat = tf.reshape(h_conv3, [-1, 10 * 10 * 64])  # 将tensor打平到vector中
        rnn_out = rnn_out[-1]
        
        h_concat = tf.concat([h_pool3_flat, rnn_out], axis=1)
        
        h_fc1_state  = tf.matmul(h_concat, w_fc1_1)+b_fc1_1
        h_fc1_action = tf.matmul(h_concat, w_fc1_2)+b_fc1_2
        # h_fc1_state  = tf.nn.relu(tf.matmul(h_concat, w_fc1_1)+b_fc1_1)
        # h_fc1_action = tf.nn.relu(tf.matmul(h_concat, w_fc1_2)+b_fc1_2)
        h_fc2_state = tf.matmul(h_fc1_state, w_fc2_1)+b_fc2_1
        h_fc2_action = tf.matmul(h_fc1_action, w_fc2_2)+b_fc2_2
        h_fc2_advantage = tf.subtract(h_fc2_action, tf.reduce_mean(h_fc2_action))
        
        output = tf.add(h_fc2_state, h_fc2_advantage)  # 神经网络的最后输出
        
        with tf.variable_scope('target'):
            # Convolution variables target
            w_conv1_target = self.weight_variable(self.first_conv)
            b_conv1_target = self.bias_variable([self.first_conv[3]])
            
            w_conv2_target = self.weight_variable(self.second_conv)
            b_conv2_target = self.bias_variable([self.second_conv[3]])
            
            w_conv3_target = self.weight_variable(self.third_conv)
            b_conv3_target = self.bias_variable([self.third_conv[3]])
            
            # Densely connect layer variables target
            w_fc1_1_target = self.weight_variable(self.first_dense)
            b_fc1_1_target = self.bias_variable([self.first_dense[1]])
            
            w_fc1_2_target = self.weight_variable(self.first_dense)
            b_fc1_2_target = self.bias_variable([self.first_dense[1]])
            
            w_fc2_1_target = self.weight_variable(self.second_dense_state)
            b_fc2_1_target = self.bias_variable([self.second_dense_state[1]])
            
            w_fc2_2_target = self.weight_variable(self.second_dense_action)
            b_fc2_2_target = self.bias_variable([self.second_dense_action[1]])


            # LSTM cell
            cell_target = tf.contrib.rnn.BasicLSTMCell(num_units = self.Num_cellState)            
            rnn_out_target, rnn_state_target = tf.nn.static_rnn(inputs = self.x_unstack, cell = cell_target, dtype = tf.float32)

        # Target Network
        h_conv1_target = tf.nn.relu(self.conv2d(self.x_normalize, w_conv1_target, 4) + b_conv1_target)
        h_conv2_target = tf.nn.relu(self.conv2d(h_conv1_target, w_conv2_target, 2) + b_conv2_target)
        h_conv3_target = tf.nn.relu(self.conv2d(h_conv2_target, w_conv3_target, 1) + b_conv3_target)
        
        h_pool3_flat_target = tf.reshape(h_conv3_target, [-1, 10 * 10 * 64])
        rnn_out_target = rnn_out_target[-1]
        h_concat_target = tf.concat([h_pool3_flat_target, rnn_out_target], axis=1)
        
        h_fc1_state_target  = tf.matmul(h_concat_target, w_fc1_1_target)+b_fc1_1_target
        h_fc1_action_target = tf.matmul(h_concat_target, w_fc1_2_target)+b_fc1_2_target
        # h_fc1_state_target = tf.nn.relu(tf.matmul(h_concat_target, w_fc1_1_target)+b_fc1_1_target)
        # h_fc1_action_target = tf.nn.relu(tf.matmul(h_concat_target, w_fc1_2_target)+b_fc1_2_target)
        h_fc2_state_target = tf.matmul(h_fc1_state_target,  w_fc2_1_target)+b_fc2_1_target
        h_fc2_action_target = tf.matmul(h_fc1_action_target, w_fc2_2_target)+b_fc2_2_target
        h_fc2_advantage_target = tf.subtract(h_fc2_action_target, tf.reduce_mean(h_fc2_action_target))
        
        output_target = tf.add(h_fc2_state_target, h_fc2_advantage_target)   # 目标网络的最后输出
        
        return output, output_target
	
    def loss_and_train(self):

        # Loss function and Train
        action_target = tf.placeholder(tf.float32, shape=[None, self.Num_action])

        y_target = tf.placeholder(tf.float32, shape=[None])
        # 这里的 y_target 就是 target Q 值

        y_prediction = tf.reduce_sum(tf.multiply(self.output, action_target), reduction_indices=1)

        # ################################################## PER ############################################################
        w_is = tf.placeholder(tf.float32, shape = [None])
        TD_error_tf = tf.subtract(y_prediction, y_target)

        # Loss = tf.reduce_mean(tf.square(y_prediction - y_target))
        Loss = tf.reduce_sum(tf.multiply(w_is, tf.square(y_prediction - y_target)))
        ###################################################################################################################

        # Loss = tf.reduce_mean(tf.square(y_prediction - y_target))

        train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-2).minimize(Loss)

        # return train_step, action_target, y_target, Loss
        return train_step, action_target, y_target, Loss, w_is, TD_error_tf

    def init_sess(self):
        # Initialize variables
        config = tf.ConfigProto()

        sess = tf.InteractiveSession(config=config)

        init = tf.global_variables_initializer()
        sess.run(init)

        # Load the file if the saved file exists
        saver = tf.train.Saver()
        check_save = input('Load Model? (1=yes/2=no): ')

        if check_save == 1:
            # Restore variables from disk.
            saver.restore(sess, self.load_path + "/model.ckpt")
            print("Model restored.")

            check_train = input('Inference or Training? (1=Inference / 2=Training): ')
            if check_train == 1:
                self.Num_start_training = 0
                self.Num_training = 0

        return sess, saver

    # Initialize input
    def input_initialization(self, env_info):
        state     = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []       
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)      
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros((self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))    
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 - (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)
        
        return observation_stack, observation_set, state_stack, state_set

    # Resize input information
    def resize_input(self, env_info, observation_set, state_set):
        observation = env_info[1]
        observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros((self.img_size, self.img_size, self.Num_stackFrame))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 - (self.Num_skipFrame * stack_frame)]
        del observation_set[0]
        observation_stack = np.uint8(observation_stack)

        state = env_info[0]
        state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.Num_dataSize))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame, :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]
        
        del self.state_set[0]

        return observation_stack, observation_set, state_stack, state_set

    # 根据进度选择动作
    def select_action(self, sess, observation_stack, state_stack):
        # 动作是具有最大Q值的动作
        Q_value = self.output.eval(feed_dict={self.x_image: [observation_stack], self.x_sensor: [state_stack]})
        action = np.zeros([self.Num_action])
        action[np.argmax(Q_value)] = 1
        
        return action, Q_value

    def Experience_Replay(self, observation, state, action, reward, next_observation, next_state, terminal):
		# If Replay memory is longer than Num_replay_memory, delete the oldest one
		if len(self.Replay_memory) >= self.Num_replay_memory:
			del self.Replay_memory[0]
			self.TD_list = np.delete(self.TD_list, 0)

		if self.progress == 'Exploring':
			self.Replay_memory.append([observation, state, action, reward, next_observation, next_state, terminal])
			self.TD_list = np.append(self.TD_list, pow((abs(reward) + self.eps), self.alpha))
		elif self.progress == 'Training':
			self.Replay_memory.append([observation, state, action, reward, next_observation, next_state, terminal])
			# ################################################## PER ############################################################
			Q_batch = self.output_target.eval(feed_dict = {self.x_image: [next_observation],self.x_sensor:[next_state]})

			if terminal == True:
				y = [reward]
			else:
				y = [reward + self.Gamma * np.max(Q_batch)]

			TD_error = self.TD_error.eval(feed_dict = {self.action_target: [action], self.y_target: y, self.x_image: [observation], self.x_sensor:[state]})[0]
			self.TD_list = np.append(self.TD_list, pow((abs(TD_error) + self.eps), self.alpha))
			# ###################################################################################################################
   
    def main(self):
        
        reward_list = []

        # 小车的坐标
        jackal_x  = []
        jackal_y  = []
        # 起点
        start_x   = []
        start_y   = []
        # 终点
        goal_x    = []
        goal_y    = []
        # 小车的起始转角
        randangle = []

        # 保存障碍
        obs_pos_x     = []
        obs_pos_y     = []
        for i in range(500):
            obs_pos_x.append([])
            obs_pos_y.append([])
            for j in range(10):
                obs_pos_x[i].append(0)
                obs_pos_y[i].append(0)

        filename1 = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_start_point_x.txt'
        with open(filename1, 'r') as f:
            lines = f.readlines()
            for line in lines:
                Start_x = [float(s) for s in line.split()]
                start_x.append(Start_x[0])
        filename2 = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_start_point_y.txt'
        with open(filename2, 'r') as f:
            lines = f.readlines()
            for line in lines:
                Start_y = [float(s) for s in line.split()]
                start_y.append(Start_y[0])
        filename3 = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_goal_point_x.txt'
        with open(filename3, 'r') as f:
            lines = f.readlines()
            for line in lines:
                Goal_x = [float(s) for s in line.split()]
                goal_x.append(Goal_x[0])
        filename4 = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_goal_point_y.txt'
        with open(filename4, 'r') as f:
            lines = f.readlines()
            for line in lines:
                Goal_y = [float(s) for s in line.split()]
                goal_y.append(Goal_y[0])  
        filename5 = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_randangle.txt'
        with open(filename5, 'r') as f:
            lines = f.readlines()
            for line in lines:
                Randangle = [float(s) for s in line.split()]
                randangle.append(Randangle[0])
        # 读取障碍
        filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_obs_pos_x.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                OBS_x = line.split(",")
                for j in range(10):     
                    obs_pos_x[i][j] = float(OBS_x[j])
                i += 1
        filename = '.../DRL_Path_Planning/src/tf_pkg/scripts/14_static_obstacle_obs_pos_y.txt'
        with open(filename, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                OBS_y = line.split(",")
                for j in range(10):
                    obs_pos_y[i][j] = float(OBS_y[j])
                i += 1

        env.reset_env(start=[start_x[0], start_y[0]], goal=[goal_x[0], goal_y[0]], Randangle=randangle[0], OBS_X=obs_pos_x[0], OBS_Y=obs_pos_y[0])
        env_info,jackal_x_temp,jackal_y_temp = env.get_env()
        jackal_x.append(jackal_x_temp)
        jackal_y.append(jackal_y_temp)
        # env.info为4维，第1维为相机消息，第2维为agent robot的self state，第3维为terminal，第4维为reward
        self.observation_stack, self.observation_set ,self.state_stack, self.state_set= self.input_initialization(env_info)
        
        step_for_newenv = 0
        
        i = 1

        while True:
            # 根据进度选取动作
            action, Q_value = self.select_action(self.sess, self.observation_stack, self.state_stack)
            action_in = np.argmax(action)
            cmd       = [0.0, 0.0] 
            v_cmd     = action_dict[action_in][0]
            omiga_cmd = action_dict[action_in][1]
            cmd[0]    = v_cmd
            cmd[1]    = omiga_cmd
            env.step(cmd)

            # Get information for update
            env_info ,jackal_x_temp, jackal_y_temp = env.get_env()
            jackal_x.append(jackal_x_temp)
            jackal_y.append(jackal_y_temp)

            self.next_observation_stack, self.observation_set, self.next_state_stack, self.state_set = self.resize_input(env_info, self.observation_set, self.state_set)  # 调整输入信息
            terminal = env_info[-2]  # 获取terminal
            reward = env_info[-1]  # 获取reward

            # Experience Replay
            self.Experience_Replay(self.observation_stack, self.state_stack, action, reward, self.next_observation_stack, self.next_state_stack, terminal)

            # Update information
            self.step += 1
            self.score += reward
            self.observation_stack = self.next_observation_stack
            self.state_stack = self.next_state_stack

            step_for_newenv = step_for_newenv + 1

            if step_for_newenv == self.MAXSTEPS:			
                terminal = True

            # If terminal is True
            if terminal == True:
                JACKAL_x = np.array(jackal_x)
                JACKAL_y = np.array(jackal_y)
                np.savetxt('.../DRL_Path_Planning/src/tf_pkg/scripts/10_D3QN_PER_image_add_sensor_obstacle_world_30m_test_JACKAL_PATH_x'+'_'+str(i)+'.txt', JACKAL_x, delimiter=',')
                np.savetxt('.../DRL_Path_Planning/src/tf_pkg/scripts/10_D3QN_PER_image_add_sensor_obstacle_world_30m_test_JACKAL_PATH_y'+'_'+str(i)+'.txt', JACKAL_y, delimiter=',')
                jackal_x  = []
                jackal_y  = []
                
                step_for_newenv = 0
                # Print informations
                print('step:'+str(self.step)+'/'+'episode:'+str(self.episode)+'/'+'progress:'+self.progress+'/'+'epsilon:'+str(self.Epsilon)+'/'+'score:'+ str(self.score))

                reward_list.append(self.score)
                reward_array = np.array(reward_list)
                # 第一次训练
                np.savetxt('.../DRL_Path_Planning/src/tf_pkg/scripts/10_D3QN_PER_image_add_sensor_obstacle_world_30m_reward_test.txt', reward_array, delimiter=',')
                # 第二次训练
                # np.savetxt('.../DRL_Path_Planning/src/tf_pkg/scripts/test_static_obstacle_world_30m_results/10_D3QN_PER_image_add_sensor_obstacle_world_30m_reward_2_test.txt', reward_array, delimiter=',')

                self.episode += 1

                self.score = 0

                # 随机初始化起点和终点的位置
                env.reset_env(start=[start_x[i], start_y[i]], goal=[goal_x[i], goal_y[i]], Randangle=randangle[i], OBS_X=obs_pos_x[i], OBS_Y=obs_pos_y[i])
                env_info, jackal_x_temp, jackal_y_temp = env.get_env()
                jackal_x.append(jackal_x_temp)
                jackal_y.append(jackal_y_temp)
                self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(env_info)
                i += 1
            
            if i==21:
                print("Finished!!!!")
                break

if __name__ == '__main__':
	agent = DQN()
	agent.main()
