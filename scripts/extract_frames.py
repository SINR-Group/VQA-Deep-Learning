import os
import math
import cv2

vid_dir = "/home/ysqyang/Documents/video-qoe-labeling/videos"
frame_dir = "/home/ysqyang/Documents/video-qoe-labeling/frames"

files = os.listdir(vid_dir)
print(files)
n = 1
for f in files:
    video = cv2.VideoCapture(vid_dir + '/' + f)
    if video.isOpened() is False:
        print("error opening video")
        continue
    framerate = video.get(5)
    os.makedirs(frame_dir + "/video_{}".format(n))
    while (video.isOpened()):
        frameId = video.get(1)
        success,image = video.read()
        if image is not None:
            image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
        if success is not True:
            print("error capturing frame")
            break
        if (frameId % math.floor(framerate) == 0):
            filename = frame_dir + "/video_{}".format(n) + "/frame_{}.jpg".format(int(frameId / math.floor(framerate))+1)
            cv2.imwrite(filename,image)
    video.release()
    print('done')
    n+=1