import socket

class Mocap_trigger:
    def __init__(self, server_ip, port_number):
        self.server_ip = server_ip
        self.port_number = port_number
        self.trigger_msg = " "
        self.client = None

    def start_client(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            print("[CONNECTING] Connecting to server...")
            self.client.connect((self.server_ip, self.port_number))
            print(f"[CONNECTED] Connected to server at {self.server_ip}:{self.port_number}")
        except ConnectionRefusedError:
            print(f"[ERROR] Cannot connect to server at {self.server_ip}:{self.port_number}")
            return

    def wait_for_trigger(self):
        while self.trigger_msg != "exo on":
            #print("[WAITING] Waiting for trigger message...")
            try:
                # Receive echoed message from the server
                self.trigger_msg = self.client.recv(1024).decode('utf-8')
                # print(f"Server: {self.trigger_msg}")
            except ConnectionResetError:
                print("[ERROR] Server closed the connection.")

        print("[TRIGGERED]")
        self.client.close()