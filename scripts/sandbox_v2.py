import subprocess
from xgym.controllers import (KeyboardController, 
                              ScriptedController, SpaceMouseController)
from xgym.model_controllers import ModelController
from xgym.gyms import Base, Lift, Stack
from xgym.utils import boundary as bd
from xgym.utils import camera as cu
from xgym.utils.boundary import PartialRobotState as RS

import signal
import time
import sys
import atexit
from dataclasses import dataclass, field

from tqdm import tqdm
import os.path as osp
import cv2
import draccus
import numpy as np
@dataclass
class RunCFG:

    task: str = input("Task: ").lower()
    base_dir: str = osp.expanduser("~/data")
    time: str = time.strftime("%Y%m%d-%H%M%S")
    env_name: str = f"xgym-sandbox-{task}-v0-{time}"
    data_dir: str = osp.join(base_dir, env_name)

    nsteps: int = 30 
    nepisodes: int = 100


def start_scripts(scripts):
    processes = []
    for script in scripts:
        print(f"Starting {script}...")
        p = subprocess.Popen([sys.executable, script])
        processes.append(p)
    return processes

def cleanup(processes):
    """
    Terminates all subprocesses provided in the list.
    """
    print("Cleaning up child processes...")
    for p in processes:
        if p.poll() is None:  # Process is still running
            p.terminate()
    # Optionally, wait for processes to terminate cleanly
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print(f"Process {p.pid} did not exit in time; killing it.")
            p.kill()

@draccus.wrap()
def main(cfg: RunCFG):
    # Define the scripts you want to run concurrently.
    scripts_to_run = ["camera_node.py", "environment_node.py"]
    recorder_script = ["recorder_node.py"]
    agent = SpaceMouseController()
    env = Lift(out_dir = cfg.data_dir, random=True)

    freq = 10  # Frequency in Hz
    dt = 1 / freq

    processes = start_scripts(scripts_to_run)
    for ep in tqdm(range(cfg.nepisodes), desc="Episodes"):
        print(f"\n=== Episode {ep+1} starting ===")

        recorder_processes = start_scripts(recorder_script)
        obs = env.reset()
        env.set_mode(7)
        env.start_record()


        # Progress bar for steps within each episode
        for step in tqdm(range(int(cfg.nsteps * freq)), desc=f"Episode {ep+1} Steps", leave=False):

            np.set_printoptions(suppress=True)

            action = agent.read()
            action[-1] += env.gripper / env.GRIPPER_MAX
            print(f"action: {action.round(4)}")

            pose = env.position.to_vector()
            pose[:3] /= int(1e3)
            pose[-1] /= env.GRIPPER_MAX
            # hist = np.vstack([hist, pose])
            # img = plot(hist)

            # action[:3] *= int(1e2)
            # action[-1] =  0.2 if action[-1] < 0.8 else 1  # less gripper

            # cv2.imshow( "data Environment", img,)
                # cv2.cvtColor(cu.tile(cu.writekeys(obs["img"])), cv2.COLOR_RGB2BGR),
            # cv2.waitKey(1)  # 1 ms delay to allow for rendering

            env.send(action)

            # obs, done, info = env.observation(), False, {}
            done = env._done



            print(f"done: {done}")
            if done:
                break
        env.stop_record()
        #env.flush()
        # env.auto_reset()

        env.reset()
        env.close()


        cleanup(recorder_processes)
        print(f"=== Episode {ep+1} ended; processes terminated. Restarting for next episode. ===\n")

        cleanup(processes)
        quit()
if __name__ == "__main__":
    main()
