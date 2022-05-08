import cv2 as cv
import numpy as np
import math
import datetime
import time
from threading import Thread
import struct
import hashlib
import base64
import socket
import types
import multiprocessing
class Clock:
    def __init__(self,name,radius):
        self.name=name
        self.margin=10
        self.dial_radius=radius

        self.scale_begin_radius=radius-10
        self.hour_scale_end_radius=radius-20
        self.minute_scale_end_radius=radius-15

        self.hour_needle_length=radius-150
        self.minute_needle_length=radius-90
        self.second_needle_length=radius-30

        self.center=(self.dial_radius+self.margin,self.dial_radius+self.margin)

    def createBoard(self):
        self.drawImage=np.zeros((2*self.center[0],2*self.center[1],3),np.uint8)
        self.drawImage[:]=(255,255,255)
        cv.circle(self.drawImage,self.center,self.dial_radius,(0,0,0),thickness=5)

    def addTimeLine(self):
        minute_lines=[] # 原来类中的函数还可以定义不是类属性的变量
        hour_lines=[]

        for i in range(60):
            # 1：获取60根刻度线，终点指向圆心，其中有12根是时钟的
            start_x = int(self.center[0] + self.scale_begin_radius * math.sin(i * np.pi / 30))
            start_y = int(self.center[1] - self.scale_begin_radius * math.cos(i * np.pi / 30))

            if 0 != i % 5:
                # 分刻度线
                minute_x = int(self.center[0] + self.minute_scale_end_radius * math.sin(i * np.pi / 30))
                minute_y = int(self.center[1] - self.minute_scale_end_radius * math.cos(i * np.pi / 30))

                line = np.array([[start_x, start_y], [minute_x, minute_y]], np.int32).reshape((-1, 1, 2))
                minute_lines.append(line)
            else:
                # 时刻度线
                hour_x = int(self.center[0] + self.hour_scale_end_radius * math.sin(i * np.pi / 30))
                hour_y = int(self.center[1] - self.hour_scale_end_radius * math.cos(i * np.pi / 30))

                line = np.array([[start_x, start_y], [hour_x, hour_y]], np.int32).reshape((-1, 1, 2))
                hour_lines.append(line)

            # 2: 画出60条分刻度线和12条时刻度线
        cv.polylines(self.drawImage, minute_lines, True, (0, 0, 0), thickness=2)
        cv.polylines(self.drawImage, hour_lines, True, (0, 0, 0), thickness=8)

    def drawTime(self):
        """
        显示时针、分针、秒针和时间日期
        :return:
        """
        while (True):
            # 1: 拷贝图片，不然最后都会画在同一张图片上
            temp = np.copy(self.drawImage)

            # 2: 获取系统时间并显示
            now_time = datetime.datetime.now()
            current_date = now_time.strftime('%Y-%m-%d')
            current_time = now_time.strftime('%H:%M:%S')
            cv.putText(temp, current_time, (self.dial_radius - 80, self.dial_radius - 100), cv.FONT_HERSHEY_COMPLEX, 1,
                       (0, 0, 0,), 2)
            cv.putText(temp, current_date, (self.dial_radius - 100, self.dial_radius - 50), cv.FONT_HERSHEY_COMPLEX, 1,
                       (0, 0, 0,), 2)

            hour, minute, second = now_time.hour, now_time.minute, now_time.second

            # 3: 画时针线
            hour_angle = ((hour + minute / 60 + second / 3600) * np.pi) / 6
            hour_x = int(self.center[0] + self.hour_needle_length * math.sin(hour_angle))
            hour_y = int(self.center[1] - self.hour_needle_length * math.cos(hour_angle))
            cv.line(temp, self.center, (hour_x, hour_y), (169, 198, 26), 15)

            # 6: 画分针线
            minute_angle = ((minute + second / 60) * np.pi) / 30
            minute_x = int(self.center[0] + self.minute_needle_length * math.sin(minute_angle))
            minute_y = int(self.center[1] - self.minute_needle_length * math.cos(minute_angle))
            cv.line(temp, self.center, (minute_x, minute_y), (186, 199, 137), 8)

            # 7: 画秒针线
            second_angle = (second * np.pi) / 30
            second_x = int(self.center[0] + self.second_needle_length * math.sin(second_angle))
            second_y = int(self.center[1] - self.second_needle_length * math.cos(second_angle))
            cv.line(temp, self.center, (second_x, second_y), (203, 222, 166), 2)

            cv.imshow(self.name, temp)
            time.sleep(0.5)

            if 27 == cv.waitKey(1):  # ESC退出
                break

clock = Clock("dw", 300)
clock.createBoard()
clock.addTimeLine()
clock.drawTime()