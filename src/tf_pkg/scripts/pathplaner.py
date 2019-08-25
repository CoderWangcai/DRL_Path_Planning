#!/usr/bin/env python
#coding=utf-8

"""
Author: Wangcai
Date: 06/2019
"""

import math


class CPFpathplaner:
    def __init__(self):
        # 慢速
        # self.vxmax = 0.2
        # self.vymax = 0.2
        # 快速
        self.vxmax = 0.4
        self.vymax = 0.4

        self.vmax = (self.vxmax**2 + self.vymax**2)**0.5

        self.kpv = 0.5#p-----v
        
        self.kap = 0.5
        self.kav = self.kpv*self.kap

        self.krp = 1.0
        self.krv = self.kpv*self.krp

        self.kx = 0.7
        self.ky = 0.7

        self.roup0 = 3.0

        self.xt = 0.0
        self.yt = 0.0
        self.vxt = 0.0
        self.vyt = 0.0

        self.numofobs = 9
        self.xobs = [0 for _ in range(self.numofobs)]
        self.yobs = [0 for _ in range(self.numofobs)]
        self.robs = [0 for _ in range(self.numofobs)]
        self.vxobs = [0 for _ in range(self.numofobs)]
        self.vyobs = [0 for _ in range(self.numofobs)]

        self.vxcmd = 0.0
        self.vycmd = 0.0

    def get_target_state(self,xt,yt,vxt,vyt):
        self.xt = xt
        self.yt = yt
        self.vxt = vxt
        self.vyt = vyt

    def get_obs_state(self,xobs,yobs,robs,vxobs,vyobs,numofobs):
        self.numofobs = numofobs
        for i in range(self.numofobs):
            self.xobs[i] = xobs[i]
            self.yobs[i] = yobs[i]
            self.robs[i] = robs[i]
            self.vxobs[i] = vxobs[i]
            self.vyobs[i] = vyobs[i]

    def limvar(self,x, xmin, xmax):
        if x < xmin:
            return xmin
        elif x > xmax:
            return xmax
        else:
            return x
    
    def getabsmax(self,data):
        val = 0.0
        maxval = 0.0
        index = 0
        for i in range(len(data)):
            if math.fabs(data[i]) > maxval:
                maxval = math.fabs(data[i])
                val = data[i]
                index = i
        
        return val,index

    def fn_pf_vc(self,x,y,vx,vy):
        d = ((x-self.xt)**2 + (y-self.yt)**2)**0.5

        Fap_x = -self.kap * (x - self.xt) / d
        Fap_y = -self.kap * (y - self.yt) / d

        Fav_x = -self.kav * (vx - self.vxt) / self.vmax
        Fav_y = -self.kav * (vy - self.vyt) / self.vmax

        Fa_x = (Fap_x + Fav_x)
        Fa_y = (Fap_y + Fav_y)

        Fr_xsum = 0.0
        Fr_ysum = 0.0

        Fr_x = [0 for _ in range(self.numofobs)]
        Fr_y = [0 for _ in range(self.numofobs)]
        Fr_p_norm = [0 for _ in range(self.numofobs)]

        for i in range(self.numofobs):
            if ((self.xobs[i]-x)**2 + (self.yobs[i]-y)**2)**0.5<self.roup0:
                dro = ((self.xobs[i]-x)**2 + (self.yobs[i]-y)**2)**0.5
                r_obs = self.robs[i]

                Frp_x = -self.krp * (self.xobs[i] - x) / (dro - r_obs)
                Frp_y = -self.krp * (self.yobs[i] - y) / (dro - r_obs)
                
                Frv_x = -self.krv * (vx - self.vxobs[i]) / self.vmax
                Frv_y = -self.krv * (vy - self.vyobs[i]) / self.vmax
                
                Fr_p_norm[i] = (Frp_x**2 + Frp_y**2)**0.5

                Frpt_x = -Frp_y
                Frpt_y = Frp_x

                if Frpt_x * (self.xt - x) + Frpt_y * (self.yt - y) < 0.0:
                    Frpt_x = -Frpt_x
                    Frpt_y = -Frpt_y

                Fr_x[i] = (Frp_x + Frv_x + Frpt_x)
                Fr_y[i] = (Frp_y + Frv_y + Frpt_y)
                
            else:
                Fr_x[i] = 0.0
                Fr_y[i] = 0.0
                Fr_p_norm[i] = 0.0

            Fr_xsum = Fr_xsum + Fr_x[i]
            Fr_ysum = Fr_ysum + Fr_y[i]
        
        val,index = self.getabsmax(Fr_p_norm)
        F_x = Fa_x + Fr_x[index]
        F_y = Fa_y + Fr_y[index]
#         F_x = Fa_x + Fr_xsum
#         F_y = Fa_y + Fr_ysum

        norm_F_sum = (F_x**2 + F_y**2)**0.5
        F_x = F_x / norm_F_sum
        F_y = F_y / norm_F_sum

        self.vxcmd = self.kx * F_x * self.vxmax + (1 - self.kx) * vx
        self.vycmd = self.ky * F_y * self.vymax + (1 - self.ky) * vy
        # print(self.vxcmd, self.vycmd, vx, vy)

        self.vxcmd = self.limvar(self.vxcmd,-self.vxmax,self.vxmax)
        self.vycmd = self.limvar(self.vycmd, -self.vymax, self.vymax)
        return self.vxcmd,self.vycmd