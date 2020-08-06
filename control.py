import socket
import collections

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
INIT_COMMAND = 'INIT'
ROTATION_COMMANDS = ['X_UP', 'Y_LEFT', 'Z_FRONT', 'X_DOWN', 'Y_RIGHT', 'Z_BACK']
rotations = ['X', 'Y', 'Z'] 
SCALE_COMMAND = 'SCALE '
MOVE_COMMAND = 'MOVE '
FIST = 'fist'
OK = 'ok'
PALM = 'palm'
THUMB_DOWN = 'thumb down'
THUMB_UP = 'thumb up'
hand_position = None
hand_scale = None
position_queue = []
scale_queue = []
label_queue = []
axis_rotation = 0
axis_counts = 0
init_counts = 0

def init_interface(port):
    sock.connect(('0.0.0.0', port))
    sock.sendall(INIT_COMMAND.encode())

def reset_object():
    global hand_position, hand_scale,\
        position_queue, scale_queue, label_queue, \
        axis_rotation, axis_counts, init_counts

    sock.sendall(INIT_COMMAND.encode())
    hand_position = None
    hand_scale = None
    position_queue = []
    scale_queue = []
    label_queue = []
    axis_rotation = 0
    axis_counts = 0
    init_counts = 0


def get_movement(img_w, img_h):
    global hand_position
    dx = 0
    dy = 0

    pred_x = sum(map(lambda x: x[0], position_queue)) / len(position_queue)
    pred_y = sum(map(lambda x: x[1], position_queue)) / len(position_queue)

    dx = pred_x - hand_position[0]
    dy = pred_y - hand_position[1]

    if abs(dx) / img_w < 0.1:
        dx = 0
    if abs(dy) / img_h < 0.1:
        dy = 0
    
    hand_position = (hand_position[0] + dx, hand_position[1] + dy)

    if pred_x / img_w < 0.3:
        dx = -img_w
    if pred_x / img_w > 0.7:
        dx = img_w
    if pred_y / img_h < 0.3:
        dy = -img_h
    if pred_y / img_h > 0.7:
        dy = img_h

    return dx / img_w, dy / img_h

def get_scale_factor(img_w, img_h, multiplier):
    global hand_scale
    

    pred_scale = sum(scale_queue) / len(scale_queue)
    # scale_factor = (pred_scale - hand_scale) / (img_w * img_h) 
    
    # if abs(scale_factor) < 0.005:
    #     return 1.0
    
    # hand_scale = pred_scale

    scale_factor = 0

    if pred_scale / (img_w * img_h) * multiplier > 0.1:
        scale_factor = 1
    if pred_scale / (img_w * img_h) * multiplier < 0.1:
        scale_factor = -1

    return scale_factor


def control_interface(hand_x, hand_y, hand_w, hand_h, img_w, img_h, label):
    global position_queue, scale_queue, label_queue, \
        hand_position, hand_scale, axis_rotation, \
        axis_counts, init_counts
    
    cx = hand_x + hand_w / 2
    cy = hand_y + hand_h / 2
    scale = hand_w * hand_h 

    position_queue.append((cx, cy))
    scale_queue.append(scale)
    label_queue.append(label)

    if hand_scale == None:
        hand_scale = scale
    
    if hand_position == None:
        hand_position = (cx, cy)

    if len(label_queue) == 3:
        filtered_label = collections.Counter(label_queue).most_common(1)[0][0]
        
        if filtered_label == PALM:
            dx, dy = get_movement(img_w, img_h)
            dz = get_scale_factor(img_w, img_h, 1)
            # command = MOVE_COMMAND + str(-dz) + " " + str(-dx) + " " + str(-dy)
            if dx > 0:
                sock.sendall(b'RIGHT')
            if dx < 0:
                sock.sendall(b'LEFT')
            if dy < 0:
                sock.sendall(b'UP')
            if dy > 0:
                sock.sendall(b'DOWN')
            if dz > 0:
                sock.sendall(b'FRONT')
            if dz < 0:
                sock.sendall(b'BACK')
            # sock.sendall(command.encode())
            if init_counts == -1:
                init_counts = 0

            if init_counts == 1:
                init_counts = -1
            

        elif filtered_label == OK:
            axis_counts += 1
            if axis_counts == 3:
                axis_rotation = (axis_rotation + 1) % 3
                axis_counts = 0
                print('AM SCHIMBAT AXA, AXA CURENTA: ', rotations[axis_rotation])
                if init_counts == 0:
                    init_counts = 1

                if init_counts == -1:
                    init_count = 0
                    sock.sendall(INIT_COMMAND.encode())

        elif filtered_label == FIST:
            scale_factor = get_scale_factor(img_w, img_h, 2)
            # command = SCALE_COMMAND + str(scale_factor)
            # sock.sendall(command.encode())
            if scale_factor > 0:
                sock.sendall(b'SCALE_UP')
            if scale_factor < 0:
                sock.sendall(b'SCALE_DOWN')

        elif filtered_label == THUMB_DOWN:
            # print('CE PLM', ROTATION_COMMANDS[axis_rotation])
            sock.sendall(ROTATION_COMMANDS[axis_rotation].encode())

        elif filtered_label == THUMB_UP:
            # print('CE PLM', ROTATION_COMMANDS[axis_rotation + 3])
            sock.sendall(ROTATION_COMMANDS[axis_rotation + 3].encode())

        label_queue = []
        position_queue = []
        scale_queue = []

def shutdown_server():
    sock.sendall(b'SHUTDOWN')
    sock.sendall(b'')
    sock.close()