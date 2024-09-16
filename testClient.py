import zmq
import time
import numpy as np
import pickle
from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids, _de>

def capture_images(rs_camera):
    start = time.time()

    color_image, depth_image = rs_camera.read(img_size=(224, 224))

    capturetime = time.time() - start

    print(f"Capture Image Time: {capturetime:.6f} seconds")
    return color_image, depth_image

def send_images_and_timestamp_to_server_via_zmq(socket, color_image, depth_imag>
    timestamp = time.time()

    # Package the timestamp, color image, and depth image together
    message = {
        "timestamp": timestamp,
        "color_image": color_image,
        "depth_image": depth_image
    }

    #Serialize the message using pickle
    socket.send(pickle.dumps(message))

    # Send timestamp, color image, and depth image as raw bytes
    #socket.send(np.array([timestamp], dtype=np.float64).tobytes(), zmq.SNDMORE>
    #socket.send(color_image.tobytes(), zmq.SNDMORE)
    #socket.send(depth_image.tobytes())
def run_capture_and_send_loop_zmq(server_ip="3.22.95.196", port=5555, duration=>

    device_ids = get_device_ids()
    if len(device_ids) == 0:
        print("No RealSense devices found.") 
        return

    # Initialize the RealSenseCamera
    rs_camera = RealSenseCamera(flip=False, device_id=device_ids[0])

    _debug_read(rs_camera, save_datastream=False)

    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.connect(f"tcp://{server_ip}:{port}")

    start_time = time.time()
    num_updates = 0

    while time.time() - start_time < duration:
        # Capture the color and depth images from the camera and resize them to>
        color_img, depth_img = capture_images(rs_camera)
        if color_img is None or depth_img is None:
            print("No image captured. Exiting.")
            break

        # Send the color and depth images along with a timestamp to the server >
        send_images_and_timestamp_to_server_via_zmq(socket, color_img, depth_im>
        num_updates += 1

    elapsed_time = time.time() - start_time
    print(f"Total updates: {num_updates}, Time: {elapsed_time:.2f} seconds, Upd>

    socket.close()
    context.term()

if __name__ == "__main__":
    run_capture_and_send_loop_zmq()

