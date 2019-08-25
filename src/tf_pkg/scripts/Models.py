#!/usr/bin/env python3

"""
Author: Wangcai
Date: 06/2019
"""

import math


def calthreat(x,y,env,x2,y2):

    t_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    if x < env.xmin + 0.5:
        t_list[0] = 1.0
    elif x < env.xmin + 1.5:
        t_list[0] = 0.5

    if x > env.xmax - 0.5:
        t_list[1] = 1.0
    elif x > env.xmax - 1.5:
        t_list[1] = 0.5

    if y < env.ymin + 0.5:
        t_list[2] = 1.0
    elif y < env.ymin + 1.5:
        t_list[2] = 0.5

    if y > env.ymax - 0.5:
        t_list[3] = 1.0
    elif y > env.ymax - 1.5:
        t_list[3] = 0.5

    if math.fabs(x-x2) < 1.0:
        if x-x2<0:
            t_list[4] = -1.0
        else:
            t_list[4] = 1.0
    elif math.fabs(x-x2) < 2.0:
        if x-x2<0:
            t_list[4] = -0.7
        else:
            t_list[4] = 0.7
    elif math.fabs(x-x2) < 3.0:
        if x-x2<0:
            t_list[4] = -0.3
        else:
            t_list[4] = 0.3

    if math.fabs(y-y2) < 1.0:
        if y-y2<0:
            t_list[5] = -1.0
        else:
            t_list[5] = 1.0
    elif math.fabs(y-y2) < 2.0:
        if y-y2<0:
            t_list[5] = -0.7
        else:
            t_list[5] = 0.7
    elif math.fabs(y-y2) < 3.0:
        if y-y2<0:
            t_list[5] = -0.3
        else:
            t_list[5] = 0.3

    xmin, xmax, ymin, ymax, d = env.getdistoboundary(x,y)
    if d < 0.5:
        t_list[6] = 1.0
    elif d < 1.5:
        t_list[6] = 0.7
    elif d < 2.5:
        t_list[6] = 0.3

    return t_list


def getvel(xcurrent, ycurrent, xgoal, ygoal,v=0.5):
    d = ((xcurrent-xgoal)**2 + (ycurrent-ygoal)**2)**0.5
    vx = 0.0
    vy = 0.0
    if d > 0.1:
        qx = (xgoal - xcurrent) / d
        qy = (ygoal - ycurrent) / d
        vx = qx*v
        vy = qy*v
        return vx,vy
    else:
        return vx,vy

def limvar(x,xmin,xmax):
    if x < xmin:
        return xmin
    elif x > xmax:
        return xmax
    else:
        return x


class Car:
    def __init__(self, x=0.0, y=0.0, yaw = 0.0, step=0.1):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.step = step
        self.count = 0
        self.wz = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.v = 0.0
        self.totaltime = 0.0
        self.maneuver_start_time = 0.0
        self.state = 'nomal'

    def move(self, vx, vy):
        v = (vx**2+vy**2)**0.5
        if v > 1.0:
            v = 1.0
        yawcmd = math.atan2(vy, vx)
        if yawcmd - self.yaw > math.pi:
            yawcmd = yawcmd - 2 * math.pi
        elif yawcmd - self.yaw < -math.pi:
            yawcmd = yawcmd + 2 * math.pi

        if math.fabs(yawcmd - self.yaw) > math.pi / 6:
            v = 0
        self.wz = 2.0*(yawcmd - self.yaw) - 0.0*self.wz
        self.wz = limvar(self.wz,-1.0,1.0)
        self.yaw = self.yaw + self.wz*self.step
        vx = v*math.cos(self.yaw)
        vy = v*math.sin(self.yaw)
        self.x = self.x + vx * self.step
        self.y = self.y + vy * self.step
        self.vx = vx
        self.vy = vy
        self.v = v
        self.count += 1
        return self.x,self.y

    def move1(self, vx, vy):
        v = (vx ** 2 + vy ** 2) ** 0.5
        if v > 1.0:
            v = 1.0
        yawcmd = math.atan2(vy, vx)
        vx = v * math.cos(yawcmd)
        vy = v * math.sin(yawcmd)
        self.x = self.x + vx * self.step
        self.y = self.y + vy * self.step
        self.vx = vx
        self.vy = vy
        self.v = v
        self.count += 1
        return self.x,self.y


class Env:
    def __init__(self, xmin=0.0, ymin=0.0, xmax=5.0, ymax=5.0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.A = 1.0
        self.B = 1.0
        self.C = 1.0
        self.stopsim = False

    def setgoal(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C

    def calreard(self, car1x, car1y, car2x, car2y):
        d = math.fabs(self.A*car1x + self.B*car1y + self.C)/((self.A**2 + self.B**2)**0.5)
        if ((car1x-car2x)**2 + (car1y-car2y)**2)**0.5 < 0.5:
            # print(((car1x-car2x)**2 + (car1y-car2y)**2)**0.5)
            self.stopsim = True
            return -1.0
        elif car1x < self.xmin or car1x > self.xmax or car1y < self.ymin or car1y > self.ymax:
            return -0.2
        elif d < 0.5:
            self.stopsim = True
            return 1.0
        else:
            return 0.0

    def getdistoboundary(self,x,y):
        xmin = x - self.xmin
        xmax = self.xmax - x
        ymin = y - self.ymin
        ymax = self.ymax - y
        d = math.fabs(self.A * x + self.B * y + self.C) / ((self.A ** 2 + self.B ** 2) ** 0.5)
        return xmin,xmax,ymin,ymax,d

