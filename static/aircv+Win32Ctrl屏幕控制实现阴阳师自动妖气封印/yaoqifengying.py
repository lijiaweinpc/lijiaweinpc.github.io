# -*- coding: utf-8 -*-
"""
Spyder Editor

阴阳师：
妖气封印自动
"""
import aircv as ac
import numpy as np
import Win32Ctrl as ct
import time
from PIL import ImageGrab
zudui=ac.imread('zudui.png')
zudui2=ac.imread('zudui2.png')
yaoqifengyin=ac.imread('yaoqifengyin.png')
yaoqifengyin2=ac.imread('yaoqifengyin2.png')
shuaxin=ac.imread('shuaxin.png')
jiaru=ac.imread('jiaru.png')
zhunbei=ac.imread('zhunbei.png')
shengli=ac.imread('shengli.png')
zuduiyaoqi=ac.imread('zuduiyaoqi.png')
pipeizhong=ac.imread('pipeizhong.png')
zidongpipei=ac.imread('zidongpipei.png')
kaishizhandou=ac.imread('kaishizhandou.png')

'''
# 位置测试
while (1):
    x,y =ct.get_mouse_point()
    print (x,y)
    time.sleep(0.5)        
    if x < 10 and y < 10:
        break
'''
'''
# 现世鬼王
while (1):
    pic = np.array(ImageGrab.grab())
    #在主页点击组队
    pos=ac.find_template(zudui,pic)
    if type(pos)==dict:
        print('组队')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(230,600)
            time.sleep(1)
            ct.mouse_click(230,160)
            time.sleep(1)
    pos=ac.find_template(zudui2,pic)
    if type(pos)==dict:
        print('组队中')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            for i in range(10):
                ct.mouse_click(430,600)
                time.sleep(0.1)
                ct.mouse_click(950,230)
                time.sleep(0.1)
    #自己变成了队长点一次开始
    pos=ac.find_template(kaishizhandou,pic)
    if type(pos)==dict:
        print('开始战斗')
        print(pos['confidence'])
        if pos['confidence']>0.6:
            ct.mouse_click(900,570)
    #在开始战斗页面点准备
    pos=ac.find_template(zhunbei,pic)
    if type(pos)==dict:
        print('开始')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(1055,523)
    #在胜利页面随意点击
    pos=ac.find_template(shengli,pic)
    if type(pos)==dict:
        print('胜利')
        print(pos['confidence'])
        if pos['confidence']>0.6:
            ct.mouse_click(1055,523)
            time.sleep(2)
            ct.mouse_click(1055,523)
            time.sleep(2)
            ct.mouse_click(1055,523)
    x,y =ct.get_mouse_point()
    print (x,y)
    time.sleep(0.5)        
    if x < 10 and y < 10:
        break

   
# 妖气封印    
while (1):
    pic = np.array(ImageGrab.grab())
    #在主页点击组队
    pos=ac.find_template(zudui,pic)
    if type(pos)==dict:
        print('组队')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(220,575)
            time.sleep(1)
            #进入妖气封印界面
            while (1):
                pic = np.array(ImageGrab.grab())
                pos=ac.find_template(pipeizhong,pic)
                if type(pos)!=dict:
                    ct.mouse_click(220,575)
                    time.sleep(1)
                    #(410,250),360,470,520
                    ct.mouse_click(410,250)
                    time.sleep(1)
                    ct.mouse_click(660,570)
                    time.sleep(1)
                else:
                    break
    #自己变成了队长点一次开始
    pos=ac.find_template(kaishizhandou,pic)
    if type(pos)==dict:
        print('开始战斗')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(900,570)
    #在开始战斗页面点准备
    pos=ac.find_template(zhunbei,pic)
    if type(pos)==dict:
        print('开始')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(1055,523)
    #在胜利页面随意点击
    pos=ac.find_template(shengli,pic)
    if type(pos)==dict:
        print('胜利')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(1055,523)
            time.sleep(2)
            ct.mouse_click(1055,523)
            time.sleep(2)
            ct.mouse_click(1055,523)
    x,y =ct.get_mouse_point()
    print (x,y)
    time.sleep(0.5)        
    x,y =ct.get_mouse_point()
    print (x,y)
    time.sleep(0.5)
    if x < 10 and y < 10:
        break
'''

while (1):
    pic = np.array(ImageGrab.grab())
    #在主页点击组队
    pos=ac.find_template(zudui,pic)
    if type(pos)==dict:
        print('组队')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(220,575)
            time.sleep(1)
            #进入妖气封印界面
            while (1):
                pic = np.array(ImageGrab.grab())
                pos=ac.find_template(pipeizhong,pic)
                if type(pos)!=dict:
                    ct.mouse_click(700,600)
                    time.sleep(1)
                else:
                    break
    #自己变成了队长点一次开始
    pos=ac.find_template(kaishizhandou,pic)
    if type(pos)==dict:
        print('开始战斗')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(900,570)
    #在开始战斗页面点准备
    pos=ac.find_template(zhunbei,pic)
    if type(pos)==dict:
        print('开始')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(1055,523)
    #在胜利页面随意点击
    pos=ac.find_template(shengli,pic)
    if type(pos)==dict:
        print('胜利')
        print(pos['confidence'])
        if pos['confidence']>0.8:
            ct.mouse_click(1055,523)
            time.sleep(2)
            ct.mouse_click(1055,523)
            time.sleep(2)
            ct.mouse_click(1055,523)
    x,y =ct.get_mouse_point()
    print (x,y)
    time.sleep(0.5)        
    x,y =ct.get_mouse_point()
    print (x,y)
    time.sleep(0.5)
    if x < 10 and y < 10:
        break