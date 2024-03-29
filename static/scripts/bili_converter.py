# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import psutil

import moviepy.editor as me
from natsort import natsorted


def kill_process(name: str):
    try:
        pids = psutil.pids()
        for pid in pids:
            # Process方法查看单个进程
            p = psutil.Process(pid)
            # print('pid-%s,pname-%s' % (pid, p.name()))
            # 进程名
            if name in p.name():
                # 关闭任务 /f是强制执行，/im对应程序名
                kill_cmd = 'taskkill /f /im ' + p.name() + '  2>nul 1>null'
                # python调用Shell脚本执行cmd命令
                os.system(kill_cmd)
    except:
        pass


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
