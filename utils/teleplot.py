import socket

# Where live plots are sent: the VS Code Teleplot extension on this machine.
TELEPLOT_IP = "127.0.0.1"
TELEPLOT_PORT = 47269


class Teleplot:
    def __init__(self, ip=TELEPLOT_IP, port=TELEPLOT_PORT):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def sendValue(self, name, value):
        try:
            message = f"{name}:{value}"
            self.sock.sendto(message.encode('utf-8'), (self.ip, self.port))
        except Exception:
            pass

    def sendGroup(self, widget, values):
        """Plot several named signals in one Teleplot widget and UDP packet."""
        try:
            items = values.items() if hasattr(values, "items") else values
            message = "\n".join(
                f"{name},{widget}:{value}" for name, value in items
            )
            self.sock.sendto(message.encode("utf-8"), (self.ip, self.port))
        except Exception:
            pass

    def close(self):
        try:
            self.sock.close()
        except:
            pass
