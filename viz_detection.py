import sys
# sys.path.append('yolo-hand-detection')

import argparse
import cv2

from yolov3.yolo import YOLO as YOLOv3
from darknet import darknet
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

def visualize_detection(args):
    if args.network == "normal":
        print("loading yolo...")
        yolo = YOLOv3("yolov3/models/cross-hands.cfg", "yolov3/models/cross-hands.weights", ["hand"])
    elif args.network == "prn":
        print("loading yolo-tiny-prn...")
        yolo = YOLOv3("yolov3/models/cross-hands-tiny-prn.cfg", "yolov3/models/cross-hands-tiny-prn.weights", ["hand"])
    elif args.network == "tiny":
        print("loading yolo-tiny...")
        yolo = YOLOv3("yolov3/models/cross-hands-tiny.cfg", "yolov3/models/cross-hands-tiny.weights", ["hand"])
    else:
        print('ERROR NETWORK ARGUMENT INVALID')
        exit()


    global metaMain, netMain, altNames
    configPath = "./darknet/cfg/yolo-hand.cfg"
    weightPath = "./darknet/yolo-hand_last.weights"
    metaPath = "./darknet/data/obj.data"
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


    physical_devices = tf.config.experimental.list_physical_devices('GPU') 
    for physical_device in physical_devices: 
        tf.config.experimental.set_memory_growth(physical_device, True)

    if args.version == "3":
        new_version = False
    elif args.version == "4":
        new_version = True
    else:
        print("ERROR: invalid version")
        print("Please use \"-v 3\" for YOLO v3 or \"-v 4\" for YOLO v4")

    yolo.size = int(args.size)
    yolo.confidence = float(args.confidence)

    classifier = tf.keras.models.load_model("yolov3/resnet_v2_50_1")

    port = int(args.port)
    # init_interface(port)

    print("starting webcam...")
    # cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    vc.set(3, 1280)
    vc.set(4, 720)

    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    labels = ['fist', 'ok', 'palm', 'thumb down', 'thumb up']

    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        if new_version == False:
            width, height, inference_time, results = yolo.inference(frame)
            for detection in results:
                print(detection)
                id, name, confidence, x, y, w, h = detection
                
                # draw a bounding box rectangle and label on the image
                color = (0, 255, 0)
                # print(x, y)
                width_margin = int(width / 10)
                height_margin = int(height / 10)
                seg = frame[max(y - height_margin,0):y+h + height_margin,\
                             max(x - width_margin,0):x+w + width_margin]
                # cv2.imshow('seg', seg)

                resize = cv2.resize(seg, dsize=(224,224))
                batch_img = np.expand_dims(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB)/255, axis=0)
                scores = classifier.predict(batch_img)


                label = labels[np.argmax(scores)]
                # print(label)

                # control_interface(x, y, w, h, width, height, label)


                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "%s (%s)" % (label, round(confidence, 2))
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
            detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
            print(detections)

            width_margin = int(1200 / 10)
            height_margin = int(720 / 10)

            if len(detections) != 0:
                detection = detections[0]

                x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]

                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
            
                seg = frame_rgb[max(ymin - height_margin,0):ymax + height_margin,\
                                max(xmin - width_margin,0):xmax + width_margin]

                cv2.imshow("seg", seg)
                # cv2.waitKey()

                resize = cv2.resize(seg, dsize=(224,224))
                batch_img = np.expand_dims(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB), axis=0)
                scores = classifier.predict(batch_img)
                label = labels[np.argmax(scores)]

                image = cvDrawBoxes(detections, frame_resized, label)
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow("preview", frame)

        rval, frame = vc.read()

        key = cv2.waitKey(3)
        if key == 27:  # exit on ESC
            break
            cv2.destroyAllWindows()
        if key == ord('r'):
            reset_object()


    # cv2.destroyWindow("preview")
    vc.release()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
    ap.add_argument('-d', '--device', default=0, help='Device to use')
    ap.add_argument('-s', '--size', default=416, help='Size for yolo')
    ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
    ap.add_argument('-p', '--port', default=10000, help='Port for Bleder')
    ap.add_argument('-v', '--version', default="3", help='Version for the YOLO Detector')
    args = ap.parse_args()

    visualize_detection(args)