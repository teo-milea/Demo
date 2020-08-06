import socket




if __name__ == '__main__':
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(('0.0.0.0', 10002))


    sock.shutdown(socket.SHUT_RDWR)

