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

from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, Registration, Frame
from pylibfreenect2 import CudaPacketPipeline

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

def find_quarter(cx, cy, img_w, img_h):
    center_x = 3/8 * img_w
    center_y = 5/8 * img_h
    q = center_x < cx + (center_y < cy) * 2
    new_cx = 0
    new_cy = 0
    k = 0.4
    k2 = 0.1
    new_cx = cx + (-1) ** (center_x > cx) * abs(cx - center_x) * k - img_w * k2
    new_cy = cy + (-1) ** (center_y > cy) * abs(cy - center_y) * k/8 - img_h * k2/2
    return new_cx, new_cy

def cvDrawBoxes(detections, img, label, depth=False):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        
        if depth:
            pt1 = (int(xmin * 23 / 29 - 0/29 * 1280), int(ymin * 14.5 / 16.4 + 0.0 * 720 / 16.4))
            pt2 = (int(xmax * 23 / 29 - 0/29 * 1280), int(ymax * 14.5 / 16.4 + 0.0 * 720 / 16.4))

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

def classify_detection(image, detection, depth_image):
    width_margin = int(darknet.network_width(netMain) / 10)
    height_margin = int(darknet.network_height(netMain) / 10)

    x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]

    xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
    seg = image[max(ymin - height_margin,0):ymax + height_margin,\
                    max(xmin - width_margin,0):xmax + width_margin]
    
    #cv2.imshow('seg', seg)
    
    resize = cv2.resize(seg, dsize=(224,224))
    batch_img = np.expand_dims(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)/255, axis=0)
    scores = classifier.predict(batch_img)
    #print(scores)
           
    label = labels[np.argmax(scores)]
    nx, ny = find_quarter(x * 1280/416, y*720/416, 1280, 720)
    nd = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)

    d = nd[min(int(ny), 719),min(int(nx),1279)]
    control_interface(x * 1280/416,y*720/416, w*1280/416, h*720/416,1280, 720, label, nd, d)
    return label

def new_pipeline():
    init_yolov4()
    listener, registration, device, fn = init_kinect()
    global classifier
    classifier = tf.keras.models.load_model("./resnet_v2_50_1")


    undistorted = Frame(512, 424, 4)
    registered = Frame(512, 424, 4)
    bigdepth = Frame(1920, 1082, 4)
    color_depth_map = np.zeros((424, 512),  np.int32).ravel()

    while True:
        frames = listener.waitForNewFrame()
        color = frames["color"]
        ir = frames["ir"]
        depth = frames["depth"]

        frame = color.asarray()
        frame_depth = cv2.resize(depth.asarray() / 4500., (int(1280), int(720)))
       
        registration.apply(color, depth, undistorted, registered,
                       bigdepth=bigdepth,
                       color_depth_map=color_depth_map)
        
        darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb,
                                (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                                interpolation=cv2.INTER_LINEAR)
        frame_depth_resized = cv2.resize(depth.asarray() / 4500.,
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
            #print(detection)
            # add_line_to_buffer(detection[2])
            # detection[2] = np.mean(buffer, axis=0)
            label = classify_detection(frame_depth_resized, detection, frame_depth)
            image = cvDrawBoxes([detection], frame_resized, label)
            frame_depth = cvDrawBoxes([detection], frame_depth, label)
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            
        frame = cv2.line(frame, (int(1280/3), 0), (int(1280/3), 720), 50, 2)

        frame = cv2.line(frame, (int(2 * 1280/3), 0), (int(2 * 1280/3), 720),50, 2)

        frame = cv2.line(frame, (0, int(720/3)), (1280, int(720/3)), 50, 2)

        frame = cv2.line(frame, (0, int(2 * 720/3)), (1280, int(2 * 720/3)), 50, 2)
        #cv2.imshow("raw", frame)
        #cv2.imshow("depth_dec", frame_depth)
        
        listener.release(frames)

        key = cv2.waitKey(3)
        if key == 27:  # exit on ESC
            break
            cv2.destroyAllWindows()
    
    device.stop()
    device.close()


def init_kinect():
    pipeline = CudaPacketPipeline()
    fn = Freenect2()
    num_devices = fn.enumerateDevices()
    if num_devices == 0:
        print("no devices")
        sys.exit(1)

    serial = fn.getDeviceSerialNumber(0)
    device = fn.openDevice(serial, pipeline=pipeline)
    types = 0 | FrameType.Color | (FrameType.Ir | FrameType.Depth)

    listener = SyncMultiFrameListener(types)
    device.setColorFrameListener(listener)
    device.setIrAndDepthFrameListener(listener)

    device.start()
    registration = Registration(device.getIrCameraParams(),
                                device.getColorCameraParams())
    print("INIT")
    return listener, registration, device, fn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--port', default=10000, help='Port for Bleder')
    args = ap.parse_args()

    init_interface(int(args.port))
    # reset_object()
    new_pipeline()
    # shutdown_server()
    # demo(args)
