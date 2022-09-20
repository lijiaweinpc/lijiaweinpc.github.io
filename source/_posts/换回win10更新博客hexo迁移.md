---
title: 换回win10更新博客hexo迁移
date: 2018-04-01
tags:
---

&emsp;&emsp;之前是使用ubuntu更新博客的，hexo+github已经配置好了，今天把他整体迁移到了win10上，记录下迁移hexo中遇到的坑和解决方案。
<!--more-->

&emsp;&emsp;工作站换回win10，主要是ubuntu找某些软件还是不太方便，娱乐机倒是从win7换到了fedora27，哈哈就是这么爱折腾。ubuntu新的LTS快来了，此时装个老版本心里感觉亏亏的。。。是个机会嘛装下fedora试试，好用就留下，不好用过不几天再重新装回去，fedora体验下来感觉上比ubuntu更纯净一点，装东西要繁琐一点，其他的娱乐机反正选gnome感觉上都差不多。

&emsp;&emsp;把hexo切给win10的过程，其实我比较暴力，用最简单的整体copy方式，主要要注意如何重建与github的关系：  
1.把整个blog文件夹copy到win10；  
2.重新安装hexo：按顺序nodejs，git，hexo；  
3.删除原blog的github SSH key重新建立，建立hexo和github连接过程参考这里的第5步：将Hexo与Github page联系起来，设置Git的user
name和email和第6步：配置Deployment，在其文件夹中，找到_config.yml文件，修改repo值（在末尾）：
**Z皓**：[使用Hexo+Github一步步搭建属于自己的博客（基础）](https://www.cnblogs.com/fengxiongZz/p/7707219.html)  
4.之后hexo g -d,OK了切换完成，原先的主题啊配置啊都没有变，就可以在win下更新啦。  
5.hexo new 换回win10更新博客hexo迁移，发现没有node-sass，npm install
node-sass提示需要python2，现在都3.6了可以用cnpm这样做：  

``` bash
npm i cnpm -g --registry=http://registry.npm.taobao.org
cnpm install node-sass@latest
```

大功告成。