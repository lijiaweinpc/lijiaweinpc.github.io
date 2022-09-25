---
title: 机器视觉运动控制系列2：Arduino舵机控制研究笔记
date: 2018-04-04
tags: [图像识别,运动控制]
---

&emsp;&emsp;采购了奥松的机械臂套件，本文记录对Arduino控制舵机的一些研究。
<!--more-->

&emsp;&emsp;先看一眼舵机是长这个样子的：

![](/static/机器视觉运动控制系列2：Arduino舵机控制研究笔记/舵机.jpg "奥松机械臂配备的舵机")

&emsp;&emsp;使用的板子是Arduino UNO，配备了16路舵机控制器：

![](/static/机器视觉运动控制系列2：Arduino舵机控制研究笔记/Arduino连接舵机控制器.jpg "Arduino UNO连接了舵机控制器")

&emsp;&emsp;这里注意舵机上的杜邦线和板子GVS可不敢接反了，接反了会烧坏舵机，不过从颜色深浅上给出了很明显的对应提示稍微注意一下。Arduino使用的是类C的语言开发，这就好办多了，先来调一下舵机让他转起来，直接向舵机写角度位置他就会移动过去，而一度一度的转动自然会显得专业一些，这个舵机的运动范围是180度，首先我们用正反转180**
测试下舵机运动情况**：

```c
#include <Servo.h> 
Servo myservo;  
//初始位置为先到0
int pos = 0;     
void setup() 
{
  //setup里只执行一次；舵机连接到1号位。
  myservo.attach(1);  
} 

void loop() 
{
  //正反转180度
  for(pos = 0; pos <= 180; pos += 1) 
  {                                  
    myservo.write(pos);     
    //1度20ms              
    delay(20);
  }
  for(pos = 180; pos>=0; pos-=1)   
  {                                
    myservo.write(pos);              
    delay(20);                  
  } 
} 
```

![](/static/机器视觉运动控制系列2：Arduino舵机控制研究笔记/servotest.gif "正反转测试舵机")

&emsp;&emsp;开始着手实时控制舵机，首先**测试串口接收数据**是否正常：

```c
//line存储传入的串行数据
char line[500] = "";  
int ret = 0;
void setup() 
{
  //打开串口，设置数据传输速率9600
  Serial.begin(9600);     
}

void loop() 
{
  // 在串口可用时才读取
  if (Serial.available() > 0) 
  {    
    // 读取数据存入line，读到'\n'为止，或者最多读500个字符
    ret = Serial.readBytesUntil('\n', line, 500);
    //打印读取到的内容：
    Serial.print("serial recevied:");
    Serial.println(line);   
  }
}
```

&emsp;&emsp;打开Arduino IDE：工具-&gt;串口监视器，随便输入点什么（英文哦）测试没问题后下来使用**串口输入字符**的方式来控制舵机运动：

```c
#include <Servo.h> 
Servo myservo;
//起始位置设为中间
int pos = 90;   
//用poslast记录上一个指令位置，也就是当前的位置
int poslast = 90;
//控制命令初值h(hold)，保持不变
char now = 'h';   
void setup() 
{
  Serial.begin(9600);  
  myservo.attach(9);
  myservo.write(pos);
}

void loop() 
{
  //把当前的位置记录下来，用pos去接下一个控制指令
  poslast = pos;
  switch(now)
  {
    //left减10度，right加10度
    case 'l':pos -= 10;break;
    case 'r':pos += 10;break;
    default:break;
  }
  //位置限制到0和180之间
  pos = constrain(pos,0,180);
  now = 'h';
  //当pos与poslast不等位置需要改变时
  if(!(pos == poslast))
  {
    //运动，给50ms时间
    myservo.write(pos);
    delay(50);
  }
  if (Serial.available() > 0) 
  {
    // 读取下一个命令
    now = Serial.read();
  }
}
```

&emsp;&emsp;直到前面几步的代码我是新拆了一个舵机来重新测试过的，下面搬出之前拼了一半的机械臂，之前已经搭建了4个舵机，我为它**
设计了两种位置形态**，分别是准备工作及休眠。

```c
#include <Servo.h> 
Servo myservo1;  
Servo myservo2;
Servo myservo3;  
Servo myservo4;
void setup() 
{
  //舵机从上到下依次连接接口2,6,9,12
  myservo1.attach(2); 
  myservo2.attach(6); 
  myservo3.attach(9); 
  myservo4.attach(12); 
  //归中位置（90,50,160,90），休眠位置（120,0,180,90）
  myservo1.write(120);   
  myservo2.write(0);
  myservo3.write(180);   
  myservo4.write(90); 
} 

void loop() 
{ 
} 
```

![](/static/机器视觉运动控制系列2：Arduino舵机控制研究笔记/准备和休息设置两种位置.jpg "为准备工作和休息设置两种位置")

&emsp;&emsp;**下来我们做点有意思的，写三个位置，让机械臂在他们之间循环：**

```c
#include <Servo.h> 
Servo myservo1;  
Servo myservo2;
Servo myservo3;  
Servo myservo4;
void setup() 
{ 
  myservo1.attach(2); 
  myservo2.attach(6); 
  myservo3.attach(9); 
  myservo4.attach(12); 
} 

void loop() 
{ 
  //每个位置休眠1.2秒
  myservo1.write(110);   
  myservo2.write(90);
  myservo3.write(90);   
  myservo4.write(180); 
  delay(1200);

  myservo1.write(90);   
  myservo2.write(50);
  myservo3.write(160);   
  myservo4.write(90); 
  delay(1200);
  
  myservo1.write(160);   
  myservo2.write(90);
  myservo3.write(90);   
  myservo4.write(70); 
  delay(1200);
} 
```

![](/static/机器视觉运动控制系列2：Arduino舵机控制研究笔记/3posmove.gif "随便写三个位置来循环")

&emsp;&emsp;这个循环是没有写delay的，会伤舵机其实我不太推荐，anyway
boss觉得科技感更强，我一度一度的转人家觉得太慢了这个更cooool。另外：一个好的习惯是在安装和使用的时候经常归中归位，记录下各舵机的位置信息。
