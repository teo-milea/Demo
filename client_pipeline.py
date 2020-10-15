import socket
import time
from imutils.video import VideoStream
import imagezmq
import cv2

sender = imagezmq.ImageSender(connect_to='tcp://10.202.130.10:5555')
image_hub = imagezmq.ImageHub(open_port='tcp://*:5555')

# sender = imagezmq.ImageSender(connect_to='tcp://127.0.0.1:5555')

rpi_name = socket.gethostname()
picam = VideoStream().start()
time.sleep(2.0)
while True:
    image = picam.read()
    cv2.imshow('image', image)
    cv2.waitKey(1)
    sender.send_image(rpi_name, image)
    client_rpi_name, recv_image = image_hub.recv_image()
    cv2.imshow('recv_image', recv_image)
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')
