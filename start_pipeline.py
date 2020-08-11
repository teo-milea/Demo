import sys

import argparse
import cv2

from own_darknet import darknet
from control import control_interface, init_interface, reset_object, shutdown_server
import tensorflow as tf
import numpy as np
from ctypes import *
import math
import random
import os
import time

netMain = None
metaMain = None
altNames = None

buffer = []
buffer_size = 3 
labels = ['fist', 'ok', 'palm', 'thumb down', 'thumb up']
classifier = None

def add_line_to_buffer(line):
    global buffer
    buffer.append(line)
    buffer = buffer[-buffer_size:]

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img, label):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    label +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

def init_yolov4():
    global metaMain, netMain, altNames
    configPath = "./own_darknet/cfg/yolo-hand.cfg"
    weightPath = "./own_darknet/yolo-hand_last.weights"
    metaPath = "./own_darknet/data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass

def darknet_image_transform(frame):
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    frame_resized = cv2.resize(frame_rgb,
                            (darknet.network_width(netMain),
                            darknet.network_height(netMain)),
                            interpolation=cv2.INTER_LINEAR)
    darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    return darknet_image, frame_resized

def classify_detection(image, detection):
    width_margin = int(darknet.network_width(netMain) / 10)
    height_margin = int(darknet.network_height(netMain) / 10)

    x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]

    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    seg = image[max(ymin - height_margin,0):ymax + height_margin,\
                    max(xmin - width_margin,0):xmax + width_margin]
    
    cv2.imshow('seg', seg)
    
    resize = cv2.resize(seg, dsize=(224,224))
    batch_img = np.expand_dims(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)/255, axis=0)
    scores = classifier.predict(batch_img)
    print(scores)
           
    label = labels[np.argmax(scores)]
    return label

def new_pipeline():
    init_yolov4()
    global classifier
    classifier = tf.keras.models.load_model("./resnet_v2_50_1")

    vc = cv2.VideoCapture(0)
    vc.set(3, 1280)
    vc.set(4, 720)

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)
        
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        
        detections = list(map(list, detections))
        if len(detections) != 0:
            detections = [detections[0]]

        for detection in detections:
            # if detection[1] < 0.5:
            #     continue
            print(detection)
            # add_line_to_buffer(detection[2])
            # detection[2] = np.mean(buffer, axis=0)
            label = classify_detection(frame_resized, detection)
            image = cvDrawBoxes([detection], frame_resized, label)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow("raw", frame)
        rval, frame = vc.read()

        key = cv2.waitKey(3)
        if key == 27:  # exit on ESC
            break
            cv2.destroyAllWindows()
    vc.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--port', default=10000, help='Port for Bleder')
    args = ap.parse_args()

    new_pipeline()
    # demo(args)
