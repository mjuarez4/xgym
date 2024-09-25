import time
import numpy as np
import requests
import json_numpy
import cv2
import threading
import logging
from typing import Dict, Any, Optional

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.robots.xarm_robot import XArmRobot  # Ensure the correct import path
from boundary_manager import BoundaryManager
from box_boundary import BoxBoundary

# ------------------------------
# Configure Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# Helper Functions
# ------------------------------

def capture_and_resize(rs_camera):
    """
    Captures an image from the RealSense camera and resizes it to 224x224 pixels.

    Args:
        rs_camera (RealSenseCamera): Instance of the RealSenseCamera.

    Returns:
        np.ndarray: Resized image.
    """
    color_image, _ = rs_camera.read()
    resized_image = cv2.resize(color_image, (224, 224))
    return resized_image

def send_observation_and_get_action(server_ip: str, port: int, payload: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Sends the observation payload to the CrossFormer API and retrieves the action.

    Args:
        server_ip (str): IP address of the CrossFormer server.
        port (int): Port number of the CrossFormer API.
        payload (Dict[str, Any]): Observation payload.

    Returns:
        Optional[np.ndarray]: Action array if successful, None otherwise.
    """
    url = f"http://{server_ip}:{port}/query"
    try:
        response = requests.post(url, json=payload, timeout=10)  # Set timeout to avoid hanging
        response.raise_for_status()
        action = response.json()
        action_array = np.array(action)
        logger.info(f"Received action: {action_array}")
        return action_array
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return None
    except ValueError as ve:
        logger.error(f"Invalid action data: {ve}")
        return None

# ------------------------------
# Configuration Phase Functions
# ------------------------------

def capture_positions_continuously(robot: XArmRobot, positions: list, capture_event: threading.Event):
    """
    Continuously captures the robot's Cartesian positions until the capture_event is cleared.

    Args:
        robot (XArmRobot): Instance of the XArmRobot.
        positions (list): List to store captured positions.
        capture_event (threading.Event): Event to control capturing.
    """
    logger.info("Capturing positions... Move the robot within the desired operational space.")
    while capture_event.is_set():
        position_state = robot.get_position()
        if position_state:
            current_position = np.array([position_state.x, position_state.y, position_state.z]) / 1000.0  # Convert mm to meters
            positions.append(current_position)
            logger.debug(f"Captured position: {current_position}")
        else:
            logger.error("Failed to retrieve robot position.")
        time.sleep(0.1)  # capturing rate

def confirm_and_define_boxes(boundary_manager: BoundaryManager) -> None:
    """
    Allows the user to define multiple bounding boxes interactively.

    Args:
        boundary_manager (BoundaryManager): Instance managing the bounding boxes.
    """
    logger.info("=== Define Safety Boxes ===")
    while True:
        define_box = input("Do you want to define a new safety box? (yes/no): ").strip().lower()
        if define_box == "yes":
            try:
                min_x = float(input("Enter min X (meters): "))
                max_x = float(input("Enter max X (meters): "))
                min_y = float(input("Enter min Y (meters): "))
                max_y = float(input("Enter max Y (meters): "))
                min_z = float(input("Enter min Z (meters): "))
                max_z = float(input("Enter max Z (meters): "))
                
                if min_x > max_x or min_y > max_y or min_z > max_z:
                    logger.error("Minimum values must be less than maximum values.")
                    continue

                box = BoxBoundary(
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    min_z=min_z,
                    max_z=max_z
                )
                boundary_manager.add_box(box)
                logger.info(f"Added box: {box}")
            except ValueError:
                logger.error("Invalid input. Please enter numeric values.")
        elif define_box == "no":
            break
        else:
            logger.error("Invalid input. Please type 'yes' or 'no'.")

# ------------------------------
# Live Operation Functions
# ------------------------------

def execute_action_with_safety(robot: XArmRobot, action: np.ndarray, boundary_manager: BoundaryManager) -> bool:
    """
    Executes the action and checks if the new position is within any safety boundaries.

    Args:
        robot (XArmRobot): Instance of the XArmRobot.
        action (np.ndarray): Action array representing cartesian movements [x, y, z, roll, pitch, yaw].
        boundary_manager (BoundaryManager): Manager handling multiple bounding boxes.

    Returns:
        bool: True if action executed successfully and within boundaries, False otherwise.
    """
    try:
        # Execute the action using set_servo_cartesian
        mvpose = action.tolist()  # Ensure it's a list
        code = robot.set_servo_cartesian(
            mvpose=mvpose,
            speed=None,       # Use default speed
            mvacc=None,       # Use default acceleration
            mvtime=0,         # Reserved parameter
            is_radian=True,   # Assuming angles are in radians
            is_tool_coord=False  # Use base coordinate system
        )
        
        if code != 0:
            logger.error(f"Action execution failed with code {code}.")
            return False

        # Allow some time for the robot to complete the movement
        time.sleep(0.5)  # Adjust potentially

        # Get the new Cartesian position
        position_state = robot.get_position(is_radian=True)
        if position_state:
            new_position = np.array([position_state.x, position_state.y, position_state.z]) / 1000.0  # Convert mm to meters
            logger.info(f"New position after action: {new_position}")
        else:
            logger.error("Failed to retrieve robot position after action.")
            return False

        # Check if the new position is within any boundary box
        if not boundary_manager.is_inside(new_position):
            logger.error(f"New position {new_position} is outside all safety boxes. Initiating emergency stop.")
            robot.stop()  # Emergency stop
            return False

        return True

    except Exception as e:
        logger.error(f"Failed to execute action with safety: {e}")
        robot.stop()  # Ensure the robot is stopped in case of unexpected errors
        return False

# ------------------------------
# Main Client Function
# ------------------------------

def run_client(server_ip: str, port: int = 8001):
    """
    Runs the client script to capture images, send them to the CrossFormer API, receive actions,
    and execute them on the robot while ensuring safety boundaries.

    Args:
        server_ip (str): IP address of the CrossFormer server.
        port (int, optional): Port number of the CrossFormer API. Defaults to 8001.
    """
    # ------------------------------
    # Initialize Camera
    # ------------------------------
    device_ids = get_device_ids()
    if len(device_ids) == 0:
        logger.error("No RealSense devices found.")
        return
    rs_camera = RealSenseCamera(flip=False, device_id=device_ids[0])
    rs_camera.start()  # Start the camera stream

    # ------------------------------
    # Initialize Robot
    # ------------------------------
    robot_ip = "192.168.1.226"  # Replace with your robot's IP
    robot = XArmRobot(ip=robot_ip, real=True)

    # ------------------------------
    # Initialize Boundary Manager
    # ------------------------------
    boundary_manager = BoundaryManager()

    try:
        # ------------------------------
        # Configuration Phase
        # ------------------------------
        logger.info("=== Configuration Phase ===")

        # Set robot to mode 2 for free movement
        set_mode_code = robot.set_mode(2)
        if set_mode_code != 0:
            logger.error("Failed to set robot to mode 2 for free movement. Exiting.")
            return
        else:
            logger.info("Robot set to mode 2 for free movement.")

        input("Press Enter to start defining safety boundaries by moving the robot manually...")

        # Allow free movement without capturing positions
        input("After moving the robot to desired positions, press Enter to proceed to define safety boxes...")

        # Define safety boxes manually
        confirm_and_define_boxes(boundary_manager)

        if len(boundary_manager.boxes) == 0:
            logger.error("No safety boxes were defined. Exiting.")
            return

        # Set robot back to servo mode
        set_mode_code = robot.set_mode(1)
        if set_mode_code != 0:
            logger.error("Failed to set robot back to servo mode. Exiting.")
            return
        else:
            logger.info("Robot set back to servo mode.")

        # ------------------------------
        # Live Operation Phase
        # ------------------------------
        logger.info("=== Live Operation Phase ===")
        input("Press Enter to start live operation (sending camera feed to CrossFormer API)...")

        num_requests = 0
        total_latency = 0
        timings = []
        operation_running = True

        json_numpy.patch()  # Handle NumPy arrays in JSON

        while operation_running:
            # Capture and resize image
            image = capture_and_resize(rs_camera)

            # Create the payload with all required fields
            payload = {
                "observation": {
                    "image_high": image.tolist()
                },
                "modality": "v",
                "ensemble": True,
                "model": "crossformer",
                "dataset_name": "bridge_dataset"  # Ensure this matches the server's dataset_name
            }

            # Send observation to server and get action
            request_start = time.time()
            action = send_observation_and_get_action(server_ip, port, payload)
            request_end = time.time()

            if action is not None:
                latency = request_end - request_start
                total_latency += latency
                num_requests += 1
                timings.append(latency)

                # Execute action with safety checks
                success = execute_action_with_safety(robot, action, boundary_manager)
                if not success:
                    logger.error("Action execution halted due to safety perimeter breach.")
                    break

            else:
                logger.error("No valid action received. Stopping live operation.")
                break

            # Optional: Add a small delay to control request rate (in case the inferences are too fast)
            time.sleep(0.1)

    finally:
        # ------------------------------
        # Cleanup Resources
        # ------------------------------
        logger.info("Cleaning up resources.")
        rs_camera.stop()
        robot.stop()

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    run_client(server_ip="10.22.74.18", port=8001)
