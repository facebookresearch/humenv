# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from gymnasium.envs.registration import register
from humenv.misc.motionlib import MotionBuffer
from multiprocessing.managers import BaseManager
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from humenv.env import HumEnv
from typing import Callable, Sequence, List
import gymnasium
import random

DEFAULT_MAX_EPISODE_STEPS = 300

register(
    id="humenv/HumEnv-v0.0.1",
    entry_point="humenv.env:HumEnv",
    max_episode_steps=DEFAULT_MAX_EPISODE_STEPS,
)


class CustomManager(BaseManager):
    pass


CustomManager.register("MotionBuffer", MotionBuffer)


def make_humenv(
    num_envs: int = 1,
    vectorization_mode: str = "async",
    motions: str | List[str] | None = None,
    motion_base_path: str | None = None,
    wrappers: Sequence[Callable[[gymnasium.Env], gymnasium.Wrapper]] | None = None,
    **kwargs,
):
    def create_single_env(motion_buffer, **kwargs) -> gymnasium.Env:
        def trunk():
            import humenv
            from humenv.env import HumEnv

            env = gymnasium.make(
                id="humenv/HumEnv-v0.0.1",
                motion_buffer=motion_buffer,
                **kwargs,
            )
            if wrappers is None:
                return env

            for wrapper in wrappers:
                env = wrapper(env)
            return env

        return trunk

    manager = None
    kwargs["motion_base_path"] = motion_base_path
    mp_context = kwargs.pop("context", None)
    if num_envs > 1:
        assert vectorization_mode in ["async", "sync"], "supported vectorization modes are 'sync' and 'async'"
        shared_lib = None
        if motions is not None:
            manager = CustomManager()
            manager.start()
            shared_lib = manager.MotionBuffer(files=motions, base_path=motion_base_path)
        env = [create_single_env(motion_buffer=shared_lib, **kwargs) for _ in range(num_envs)]
        if vectorization_mode == "sync":
            env = SyncVectorEnv(env)
        else:
            env = AsyncVectorEnv(env, context=mp_context)
    else:
        env = create_single_env(motion_buffer=motions, **kwargs)()
    seed = kwargs.get("seed", random.randint(0, 9999))
    env.reset(seed=seed)  # this is used to pass the seed to the environment
    return env, manager


##########
# TASKS  #
##########

STAND_TASKS = ["move-ego-0-0", "move-ego-low-0-0", "headstand"]

LOCOMOTION_TASKS = []
for angle in [0, -90, 90, 180]:
    for speed in [2, 4]:
        LOCOMOTION_TASKS.append(f"move-ego-{angle}-{speed}")

LOCOMOTION_LOW_TASKS = []
for angle in [0, -90, 90, 180]:
    for speed in [2]:
        LOCOMOTION_LOW_TASKS.append(f"move-ego-low-{angle}-{speed}")

JUMP_TASKS = ["jump-2"]

ROTATION_TASKS = []
for axis in ["x", "y", "z"]:
    for speed in [-5, 5]:
        ROTATION_TASKS.append(f"rotate-{axis}-{speed}-0.8")

RAISE_ARMS_TASKS = []
for left in ["l", "m", "h"]:
    for right in ["l", "m", "h"]:
        RAISE_ARMS_TASKS.append(f"raisearms-{left}-{right}")

SITTING_LIEONGROUND_TASKS = [
    "crouch-0",
    "sitonground",
    "lieonground-up",
    "lieonground-down",
    "split-0.5",
    "split-1",
]

CRAWL_TASKS = []
for d in ["u", "d"]:
    for h in [0.4, 0.5]:
        for speed in [0, 2]:
            CRAWL_TASKS.append(f"crawl-{h}-{speed}-{d}")

MOVE_AND_RAISE_HANDS_TASKS = []
for angle in [0]:
    for speed in [2]:
        for left in ["l", "m", "h"]:
            for right in ["l", "m", "h"]:
                MOVE_AND_RAISE_HANDS_TASKS.append(f"move-ego-{angle}-{speed}-raisearms-{left}-{right}")

STANDARD_TASKS = (
    STAND_TASKS
    + LOCOMOTION_TASKS
    + LOCOMOTION_LOW_TASKS
    + JUMP_TASKS
    + ROTATION_TASKS
    + RAISE_ARMS_TASKS
    + SITTING_LIEONGROUND_TASKS
    + CRAWL_TASKS
)


ALL_TASKS = STANDARD_TASKS + MOVE_AND_RAISE_HANDS_TASKS

__version__ = "0.1.1"
