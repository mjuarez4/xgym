import time

import gymnasium as gym
import numpy as np

from xgym.controllers import (KeyboardController, ModelController,
                              ScriptedController)
from xgym.gyms import Base
from xgym.utils import boundary as bd
from xgym.utils.boundary import PartialRobotState as RS

from pynput import keyboard

def main():
    # env: LUCBase = gym.make("luc-base")
    controller = KeyboardController()
    controller.register(keyboard.Key.space, lambda : env.stop(toggle=True))
    env = Base()
    env.render()
    obs = env.reset()

    # env.go(RS(cartesian=[75, -125, 125], aa=[0, 0, 0]), relative=True)
    # [3.133083, 0.013429, -1.608588]

    # rpy=[np.pi, 0, -np.pi/2]
    # print(env.angles)
    # quit()
    # env.go(RS(cartesian=env.cartesian, aa=rpy), relative=False)

    # print(env.rpy)
    # quit()
    # rpy = np.array(env.rpy) + np.array([0.0,0,-0.1])
    # rpy[1] = 0
    # env.go(RS(cartesian=env.cartesian, aa=rpy), relative=False)

    # action = [0.1, 0.1, 0.1, 0, 0, 0, 0]
    print(f'entering loop')
    for i in range(1000):
        time.sleep(0.01)
        env.render()

        # action = controller(obs)
        mode = 'all'
        if mode == 'cart':
            action = np.random.normal(0, 1, 7) * 25
            action[3:6] = 0
        elif mode == 'rot':
            action = np.random.normal(0, 1, 7) * 0.05
            action[:3] = 0
        else:
            action = np.random.normal(0, 1, 7) * 25
            action[3:6] = np.random.normal(0, 1, 3) * 0.05 # rotation

        # gripper is x100 but only active 20% of the time
        gripper = np.random.choice([0, 1], p=[0.6, 0.4])
        action[6] *= gripper * 4

        action = controller(obs)
        while action.sum()==0:
            action = controller(obs)
        print()
        print(action)
        # continue

        # things = env.step()
        # obs, reward, truncated, terminated, info = env.step(env.action_space.sample())
        things = env.step(action)
        # print(things)
        env.render()

    # env.close()
    controller.close()
    quit()


if __name__ == "__main__":
    main()
