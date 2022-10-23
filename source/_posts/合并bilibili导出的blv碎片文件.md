---
title: 合并bilibili导出的blv碎片文件
date: 2019-07-07
tags: 技术杂文
---

&emsp;&emsp;当使用手机缓存B站的视频时会发现，一个视频被拆成了很多blv格式的碎片，将大量的碎片合并成一个mp4文件，这里记录的方法作者原文在[https://www.cnblogs.com/FHC1994/p/10760809.html](https://www.cnblogs.com/FHC1994/p/10760809.html)
&emsp;&emsp;19.10.23更新：现在的策略是一个视频被拆成了video.m4s和audio.m4s两个部分，从合并上来说其实更简单了。（bilibili在我安卓手机的缓存路径为：此电脑\iQOO Neo3\内部存储设备\Android\data\tv.danmaku.bili\download\<某个视频>）
&emsp;&emsp;22.10.15更新：使用ffmpeg进行合并转换。
<!--more-->

&emsp;&emsp;22.10.15更新：使用ffmpeg进行合并转换。原来的moviepy太慢了，合并一个200M左右的视频要一个多小时，同样的ffmpeg命令行测试大约只要三分钟。。。
ffmpeg首先需要下载他的执行包，然后的核心命令是：
```bash
.\ffmpeg.exe -i video.m4s -i audio.m4s -codec copy Output.mp4
```
所以我们新加一个convert_m4s_ffmpeg方法如下：
```python
import os
def convert_m4s_ffmpeg(source_path: str, target_path: str, ffmpeg_path: str):
    folders = os.listdir(source_path)
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(source_path, folder)):
            if "video.m4s" in files and "audio.m4s" in files:
                video_path = os.path.join(root, "video.m4s")
                audio_path = os.path.join(root, "audio.m4s")
                target = os.path.join(target_path, folder + ".mp4")
                ffmpeg_cmd = ffmpeg_path + " -i " + video_path + " -i " + audio_path + " -codec copy " + target
                os.system(ffmpeg_cmd)
                print("{}---{}---拼接成功！".format(folder, "{}.mp4".format(folder)))


if __name__ == '__main__':
    # 复制:此电脑\iQOO Neo3\内部存储设备\Android\data\tv.danmaku.bili\download\465916979->D:\465916979
    source_dir = r"D:\465916979"
    ffmpeg = r"E:\ffmpeg\bin\ffmpeg.exe"
    convert_m4s_ffmpeg(source_dir, r'D:', ffmpeg)
```

-----
&emsp;&emsp;19.10.23更新bilibili现在的策略是一个视频被拆成了video.m4s和audio.m4s两个部分，从合并上来说其实更简单了，合并video.m4s和audio.m4s的方法：

```python
def convert_m4s(source_path: str, target_path: str):
    folders = os.listdir(source_path)
    for folder in folders:
        for root, dirs, files in os.walk(os.path.join(source_path, folder)):
            if "video.m4s" in files and "audio.m4s" in files:
                video_clip = me.VideoFileClip(os.path.join(root, "video.m4s"))
                audio_clip = me.AudioFileClip(os.path.join(root, "audio.m4s"))
                video_clip = video_clip.set_audio(audio_clip)
        target = os.path.join(target_path, folder + ".mp4")
        video_clip.write_videofile(target, fps=24, remove_temp=True)
        print("{}---{}---拼接成功！".format(folder, "{}.mp4".format(folder)))
```
&emsp;&emsp;需要先将缓存文件拷贝到本地，然后执行convert_m4s方法，之后还需要手工check每个文件的具体名字，所以不适合大批量分p的小文件操作。。。

-----
&emsp;&emsp;之前blv碎片的整合方式：
```python
def convert_blv(source_path: str, target_path: str):
    folders = os.listdir(source_path)
    for folder in folders:
        tmp_video = []
        # root指的是当前正在遍历的这个文件夹，dirs是该文件夹中所有的目录的名字(不包括子目录)，files是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(os.path.join(source_path, folder)):
            # 自然排序法
            files = natsorted(files)
            for file in files:
                # 如果后缀名为 .blv
                if os.path.splitext(file)[1] == '.blv':
                    # .blv格式视频的完整路径
                    file_path = os.path.join(root, file)
                    # 读取视频到内存
                    my_video = me.VideoFileClip(file_path)
                    # 添加到数组
                    tmp_video.append(my_video)
        # 对多个视频在时长上进行拼接
        final_clip = me.concatenate_videoclips(tmp_video)
        target = os.path.join(target_path, folder + ".mp4")
        final_clip.write_videofile(target, fps=24,
                                   # remove_temp=True表示生成的音频文件是临时存放的，视频生成后，单独音频文件会自动处理掉！
                                   remove_temp=True)
        print("{}---{}---拼接成功！".format(folder, "{}.mp4".format(folder)))
        kill_process('ffmpeg-win64-v4.1.exe')  # moviepy产生的特定进程自动关闭有异常
    cmd = 'shutdown -s -t 10'
    os.system(cmd)
```

完整的参考脚本：
[bili_collector.py](/static/scripts/bili_converter.py)
