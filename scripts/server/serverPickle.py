import zmq
import pickle
import time
import numpy as np

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)  # PULL socket to receive data
    socket.bind("tcp://0.0.0.0:5555")  # Listen on all available IPs and port 5555

    print("Server is listening on port 5555...")

    while True:
        # Receive the message from the client
        message = socket.recv()

        # Unpack the message (timestamp, color image, and depth image)
        data = pickle.loads(message)
        timestamp = data["timestamp"]
        color_image = data["color_image"]
        depth_image = data["depth_image"]

        # Measure the time difference (latency)
        current_time = time.time()
        latency = current_time - timestamp

        print(f"Images received. Latency: {latency:.6f} seconds")

if __name__ == "__main__":
    run_server()
