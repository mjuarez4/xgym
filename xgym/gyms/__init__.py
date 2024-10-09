from .base import Base 
from .lift import Lift
from collections import OrderedDict


from gymnasium.envs.registration import register
register(
    id="xgym/base-v0",
    entry_point="xgym.gyms:Base",
    max_episode_steps=50,
    # kwargs={"obs_type": "state"},
)


TASKS = OrderedDict(
    (
        # (
        #     "reach",
        #     {
        #         "env": Reach,
        #         "action_space": "xyz",
        #         "episode_length": 50,
        #         "description": "Reach a target location with the end effector",
        #     },
        # ),
        # (
        #     "push",
        #     {
        #         "env": Push,
        #         "action_space": "xyz",
        #         "episode_length": 50,
        #         "description": "Push a cube to a target location",
        #     },
        # ),
        # (
        #     "peg_in_box",
        #     {
        #         "env": PegInBox,
        #         "action_space": "xyz",
        #         "episode_length": 50,
        #         "description": "Insert a peg into a box",
        #     },
        # ),
        ( 'base', {
            'env': Base,
            'action_space': 'xyzw',
            'episode_length': 50,
            'description': 'Base environment for testing purposes.'
        }),

        (
            "lift",
            {
                "env": Lift,
                "action_space": "xyzw",
                "episode_length": 50,
                "description": "Lift a cube above a height threshold",
            },
        ),
    )
)
