---
title: aircv+Win32Ctrl屏幕控制实现阴阳师自动妖气封印
date: 2022-10-15
tags: [图像处理,游戏]
---

&emsp;&emsp;本文是对老项目的整理，之前使用过aircv+Win32Ctrl屏幕控制的思路进行过接近物理作弊，疯狂的刷阴阳师妖气封印的式神碎片。
<!--more-->
&emsp;&emsp;完整的代码见附件，关键点一个是使用**aircv这个组件进行图像识别**,一个是使用**Win32Ctrl来模拟键盘发布指令。**  
aircv不需要训练模型，只需要提供一个目标的照片即可进行识别，所以我截取了大量的关键选择位置点，具体来说：
1. 使用模拟器在PC运行阴阳师；  
2. 手动操作一遍妖气封印，并截取关键位置的截图；  
这里举几个例子：  
![](/static/aircv+Win32Ctrl屏幕控制实现阴阳师自动妖气封印/kaishizhandou.png)
![](/static/aircv+Win32Ctrl屏幕控制实现阴阳师自动妖气封印/zuduiyaoqi.png)
![](/static/aircv+Win32Ctrl屏幕控制实现阴阳师自动妖气封印/shengli.png)
可以看到阴阳师游戏中这些按钮和状态标志都是比较清晰的。
3. 完成判定逻辑的编写，即if发现了A在那个位置，发送命令点击什么位置。
```python
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
```
整个思路还是非常清晰简单的。

[yaoqifengying.py](/static/aircv+Win32Ctrl屏幕控制实现阴阳师自动妖气封印/yaoqifengying.py)
[Win32Ctrl.py](/static/aircv+Win32Ctrl屏幕控制实现阴阳师自动妖气封印/Win32Ctrl.py)
