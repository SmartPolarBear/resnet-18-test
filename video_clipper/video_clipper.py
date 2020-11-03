import cv2
import os
import shutil

from os import listdir, getcwd
from os.path import isfile, join, splitext


def clip_video(video_name, interval):
    save_path, ext_name = splitext(video_name)
    is_exists = os.path.exists(save_path)

    if not is_exists:
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)

    video_capture = cv2.VideoCapture(video_name)

    i = 0
    while True:
        success, frame = video_capture.read()
        i += 1
        if not success:
            print('video is all read')
            break
        if i % interval == 0:
            save_name = join(save_path, str(i) + '.jpg')
            cv2.imwrite(save_name, frame)
            print('image of %s is saved' % save_name)
       


if __name__ == '__main__':
    data_path = join(getcwd(), "video_clipper\\video")
    mp4_files = [join(data_path, f) for f in listdir(data_path) if isfile(
        join(data_path, f)) and splitext(f)[1].lower() == '.mp4']
    for f in mp4_files:
        clip_video(f, 5)
