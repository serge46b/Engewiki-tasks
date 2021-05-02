import socket

servsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

servsock.bind(('localhost', 8000))

servsock.listen(2)


def send_msg(sock, msg):
    msg += '\n'
    msg = msg.encode()
    send_len = 0
    while send_len < len(msg):
        snd_bytes = sock.send(msg[send_len:])
        if snd_bytes == 0:
            raise RuntimeError("connection error")
        send_len += snd_bytes


def receive_msg(client_socket):
    symbol = ''
    retmsg = ''
    while symbol != b'\n':
        symbol = client_socket.recv(1)
        if symbol != b'\n':
            retmsg += symbol.decode()
        elif symbol == b'':
            raise RuntimeError("connection was closed")
    return retmsg


(clisock, address) = servsock.accept()
print('New user address ' + str(address[0]) + ':' + str(address[1]))
while True:
    ans = receive_msg(clisock)
    print('received message: ' + ans)
    print('sending answer...')
    send_msg(clisock, ans)


