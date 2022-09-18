# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

''''''
from PIL import Image
import numpy as np
import matplotlib.image
img=np.array(Image.open('/home/jiawei/Pictures/lookup-table.png'))
for i in range(512):
    for j in range(512):
        r=(i%64)*4
        g=(j%64)*4
        b=(i//64)*32+(j//64)*4
        img[i,j]=(r,g,b)
matplotlib.image.imsave('/home/jiawei/Pictures/oricard.png',img)

from PIL import Image
import numpy as np
import matplotlib.image
img=np.array(Image.open('/home/jiawei/Pictures/timg.jpeg'))
style=np.array(Image.open('/home/jiawei/Pictures/pink.jpg'))
rows,cols,dims=img.shape
for i in range(rows):
    for j in range(cols):
        r, g, b=img[i,j]
        m=b//4//8*64+r//4
        n=b//4%8*64+g//4
        img[i,j]=style[m,n]
matplotlib.image.imsave('/home/jiawei/Pictures/output.png',img)