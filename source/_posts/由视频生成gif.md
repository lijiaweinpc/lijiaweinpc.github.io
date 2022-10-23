---
title: 由视频生成gif
date: 2022-10-15
tags: [技术杂文,图像处理]
---

&emsp;&emsp;本文研究如何用一段视频生成gif。具体分为两步，首先将视频抽帧，然后将选择的帧合并成一个gif文件。
<!--more-->
**首先是视频的抽帧。使用opencv完成：**
```python
import os
import shutil

from cv2 import imwrite
from cv2 import IMWRITE_PNG_COMPRESSION
from cv2 import VideoCapture


def extract_frames(source, target, s):
    video = VideoCapture()
    if not video.open(source):
        raise "Video can't open!"
    count = 0
    while True:
        _, frame = video.read()
        if frame is None:
            break
        if count % s == 0:
            save_path = "{}/{:>04d}.png".format(target, count)
            imwrite(save_path, frame, [IMWRITE_PNG_COMPRESSION, 9])  # 压缩比1~10默认3，数字越小压缩比越小
        count += 1
    video.release()


if __name__ == '__main__':
    video_path = r'C:\Users\LI\Desktop\Video_20221015120736.wmv'
    output_folder = 'frames'
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    index = 0
    step = 10  # 多少帧取一张
    extract_frames(video_path, output_folder, step)
```

&emsp;&emsp;这里IMWRITE_PNG_COMPRESSION确实可以使得保存出来的每一帧文件大小压缩，但是实测只是这里压缩并不会减小最终的gif，所以在合并保存时的resize仍是非常必要的。  

**下来是合并帧保存gif。使用PIL调整大小和压缩，imageio完成保存：**
```python
import os
import imageio
from PIL import Image

input_folder = 'frames'
output_path = 'showcase.gif'
# 压缩后的尺寸
x = 1024
y = 512
ims_list = []

for _, im_path in enumerate(os.listdir(input_folder)):
    im = Image.open(os.path.join(input_folder, im_path))
    frame = im.convert('RGB')
    if frame.size[0] > x and frame.size[1] > y:
        frame.thumbnail((x, y))
    ims_list.append(frame)
imageio.mimsave(output_path, ims_list, 'GIF', fps=3)

``` 

&emsp;&emsp;最后得到的git示意如下（录频工具使用FSCapture）：

![gif效果](/static/由视频生成gif/showcase.gif)
