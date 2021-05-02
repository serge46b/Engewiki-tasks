import socket

clisock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

clisock.connect(('localhost', 8000))


def receive_msg(client_socket):
    symbol = ''
    retmsg = ''
    while symbol != b'\n':
        symbol = client_socket.recv(1)
        if symbol != b'\n':
            retmsg += symbol.decode()
    return retmsg


def send_msg(sock, msg):
    msg += '\n'
    msg = msg.encode()
    send_len = 0
    while send_len < len(msg):
        snd_bytes = sock.send(msg[send_len:])
        if snd_bytes == 0:
            raise RuntimeError("connection error")
        send_len += snd_bytes


while True:
    try:
        send_msg(clisock, input())
        ans = receive_msg(clisock)
        print('returned: ' + ans)
    except KeyboardInterrupt:
        print("interrupted")
