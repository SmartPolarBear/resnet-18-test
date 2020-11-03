import cv2
import random
import numpy as np
from os import listdir, getcwd
from os.path import isfile, join, splitext


def affine(img):
    rows,cols=img.shape[:2]
    point1=np.float32([[50,50],[300,50],[50,200]])
    point2=np.float32([[10,100],[300,50],[100,250]])
    M=cv2.getAffineTransform(point1,point2)
    dst=cv2.warpAffine(img,M,(cols,rows),borderValue=(255,255,255))
    return dst


if __name__ == '__main__':
    rotations = [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE,
                 cv2.ROTATE_90_COUNTERCLOCKWISE]
    data_path = join(getcwd(), "data_augmentation\\block")
    jpg_files = [join(data_path, f) for f in listdir(data_path) if isfile(
        join(data_path, f)) and splitext(f)[1].lower() == '.jpg']

    for f in jpg_files:
        img = cv2.imread(f)
        blur = cv2.bilateralFilter(img, 9, 75, 75)
        name, ext = splitext(f)
        new_name = name+"_mb"+ext
        cv2.imwrite(new_name, blur)
        print(new_name+" written")

    for f in jpg_files:
        img = cv2.imread(f)
        rt = cv2.rotate(img, random.choice(rotations))
        name, ext = splitext(f)
        new_name = name+"_rt"+ext
        cv2.imwrite(new_name, rt)
        print(new_name+" written")

    for f in jpg_files:
        img = cv2.imread(f)
        distort = affine(img)
        name, ext = splitext(f)
        new_name = name+"_af"+ext
        cv2.imwrite(new_name, distort)
        print(new_name+" written")
