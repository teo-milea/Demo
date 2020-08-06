import bpy
from mathutils import *
import math
import socket
import threading
import queue
from time import sleep

running = True
Q = queue.Queue()

class Server:
    def __init__(self, host="", port=10001):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#        self.socket.setblocking(False)
        self.socket.bind((host,port))
        print(self.socket.getsockname())
        
    def receive(self):
        global running
        while running:
            try:
                data, addr = self.socket.recvfrom(1024)
                print('Am primit')
                print(data)
                Q.put(data.decode())
            except socket.error:
                print(socket.error)
                continue
        self.socket.shutdown(socket.SHUT_RDWR)
        self.socket.close()
        
    def execute(self):
        self.thread = threading.Thread(target=self.receive)
        self.thread.start()
                         
class Transformer():
    
    def _execute(self):
        rover = bpy.data.objects["Cube"]
        angle = math.radians(0.5)
        
        x_pos_rot = Euler((angle, 0, 0))
        y_pos_rot = Euler((0, angle, 0))
        z_pos_rot = Euler((0, 0, angle))
        
        x_neg_rot = Euler((-angle, 0, 0))
        y_neg_rot = Euler((0, -angle, 0))
        z_neg_rot = Euler((0, 0, -angle))
        
        scale_factor = 1.01

        delta_loc = 0.1
        
        global running
        while running:
            command = Q.get()
            print(command)
            
            # Object rotations
            if command == "X_UP":
                rover.rotation_euler.rotate(x_pos_rot)
            if command == "X_DOWN":
                rover.rotation_euler.rotate(x_neg_rot)
            
            if command == "Y_LEFT":
                rover.rotation_euler.rotate(y_neg_rot)
            if command == "Y_RIGHT":
                rover.rotation_euler.rotate(y_pos_rot)
            
            if command == "Z_FRONT":
                rover.rotation_euler.rotate(z_pos_rot)
            if command == "Z_BACK":
                rover.rotation_euler.rotate(z_neg_rot)
            
            # Object translations
            if command == "RIGHT":
                rover.location += Vector((0, delta_loc, 0))
            if command == "LEFT":
                rover.location += Vector((0, -delta_loc, 0))

            if command == "FRONT":
                rover.location += Vector((delta_loc, 0, 0))
            if command == "BACK":
                rover.location += Vector((-delta_loc, 0, 0))

            if command == "UP":
                rover.location += Vector((0, 0, delta_loc))
            if command == "BACK":
                rover.location += Vector((0, 0, -delta_loc))
                

            # Object all axis scale
            if command == "SCALE_UP":
                rover.scale *= scale_factor
            if command == "SCALE_DOWN":
                rover.scale /= scale_factor
            
            # Position, Orientation and Scale Reset
            if command == "RESET":
                rover.scale *= 0
                rover.location *= 0
                rover.rotation_euler = Euler((0,0,0))

            # Command that closes the communication
            # and shuts down the server
            if command == "SHUTDOWN":
                running = False

    def execute(self):
        self.thread = threading.Thread(target=self._execute)
        self.thread.start()


def register():
    bpy.types.Scene.watcher_running = bpy.props.BoolProperty(default=False)
    for cls in classes:
        bpy.utils.register_class(cls)
    
def unregister():
    del bpy.types.Scene.watcher_running
    for cls in classes:
        bpy.utils.unregister_class(cls) 


if __name__ == '__main__':
    server = Server()
    server.execute()
    trans = Transformer()
    trans.execute()
    
#    server.thread.join()
#    trans.thread.join()
    
#    print("FINISH SCRIPT")
