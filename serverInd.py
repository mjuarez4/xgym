import zmq
import numpy as np
import time

def run_server():
    context = zmq.Context()
    socket = context.socket(zmq.PULL)  # PULL socket to receive data
    socket.bind("tcp://0.0.0.0:5555")  # Listen on all available IPs and port 5555

    print("Server is listening on port 5555...")

    while True:
        # Receive the timestamp, color image, and depth image
        timestamp_data = socket.recv()  # Receive the timestamp as bytes
        color_image_data = socket.recv()  # Receive the color image as bytes
        depth_image_data = socket.recv()  # Receive the depth image as bytes

        # Convert the byte data back into usable formats
        timestamp = np.frombuffer(timestamp_data, dtype=np.float64)[0]  # Convert timestamp b>
        color_image = np.frombuffer(color_image_data, dtype=np.uint8).reshape((224, 224, 3)) >
        depth_image = np.frombuffer(depth_image_data, dtype=np.uint16).reshape((224, 224, 1))>

        # Measure the time difference (latency)
        current_time = time.time()
        latency = current_time - timestamp

        print(f"Images received. Latency: {latency:.6f} seconds")

if __name__ == "__main__":
    run_server()
