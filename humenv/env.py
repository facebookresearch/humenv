# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import sys
import inspect
import numpy as np
import mujoco
import gymnasium as gym
from enum import Enum
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from pathlib import Path
import humenv.reset
import humenv.rewards
import humenv.utils
from humenv.misc.motionlib import MotionBuffer


_XML = "assets/robot.xml"  # this is a copy of robot_july5_mpd_kp3_kd2.xml
_ROBOT_IDX_START: int = 1
_ROBOT_IDX_END: int = 25
_NUM_RIGID_BODIES: int = 24
_NUM_VEL_LIMIT: int = 72


class StateInit(Enum):
    Default = 0
    Fall = 1
    MoCap = 2
    DefaultAndFall = 3
    MoCapAndFall = 4


class HumEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task: humenv.rewards.RewardFunction | str | None = None,
        xml: str = _XML,  # it can be the path to a file or an xml string
        state_init: str | StateInit = "Default",
        camera: str = "front_side",
        render_width: int = 640,
        render_height: int = 480,
        seed: Optional[int] = None,
        render_mode: str | None = "rgb_array",
        fall_prob: float = 0.3,
        motion_buffer: MotionBuffer | str | None | List[str] = None,
        motion_base_path: str | None = None,
    ) -> None:
        self.xml = xml
        self.state_init = state_init
        self.camera = camera
        self.render_width = render_width
        self.render_height = render_height
        self.seed = seed
        self.render_mode = render_mode
        self.fall_prob = fall_prob
        self._init_motion_buffer(motion_buffer, base_path=motion_base_path)
        self.set_task(task)
        simulation_dt = 1.0 / 450.0
        self.action_repeat = 15
        module_path = Path(humenv.__file__).resolve().parent
        if Path(self.xml).exists():
            self.model = mujoco.MjModel.from_xml_path(self.xml)
        elif (module_path / self.xml).exists():
            self.model = mujoco.MjModel.from_xml_path(str(module_path / self.xml))
        else:
            self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = simulation_dt
        obs = self.get_obs()
        self.observation_space = gym.spaces.Dict(
            {
                k: gym.spaces.Box(
                    -np.inf * np.ones_like(v),
                    np.inf * np.ones_like(v),
                    dtype=v.dtype,
                )
                for k, v in obs.items()
            }
        )
        # Get the actuator range from mujoco
        low, high = [], []
        for i in range(self.model.nu):
            assert bool(self.model.actuator(i).ctrllimited)
            low.append(self.model.actuator(i).ctrlrange[0])
            high.append(self.model.actuator(i).ctrlrange[1])
        self.action_space = gym.spaces.Box(
            low=np.array(low),
            high=np.array(high),
            shape=(self.model.nu,),
            dtype=np.float64,
        )
        self.state_init = StateInit[self.state_init]
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]
        self.renderer = None
        super().reset(seed=self.seed)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed = seed
        if options:
            self.set_physics(**options)
        else:
            self.reset_humanoid()
        observation = self.get_obs()
        info = self.get_info()
        return observation, info

    def step(self, action: np.ndarray):
        # actual step with mujoco
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.action_repeat)
        if self.data.warning.number.any():
            warning_index = np.nonzero(self.data.warning.number)[0][0]
            warning = mujoco.mjtWarning(warning_index).name
            raise ValueError(f"UNSTABLE MUJOCO. Stopped due to divergence ({warning}).\n")
        mujoco.mj_step1(self.model, self.data)

        # compute returns
        observation = self.get_obs()
        # np.testing.assert_allclose(self.data.ctrl, action)
        # self.data.ctrl[:] = action
        reward = self.task.compute(self.model, self.data)
        terminated = self.is_terminated()
        truncated = False
        info = self.get_info()
        return observation, reward, terminated, truncated, info

    def is_terminated(self) -> bool:
        return False

    def get_info(self) -> Dict[str, np.ndarray]:
        physics = {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}
        return physics

    def get_obs(self) -> Dict[str, np.ndarray]:
        mujoco.mj_kinematics(self.model, self.data)
        obs_dict = compute_humanoid_self_obs_v2(
            self.model,
            self.data,
            upright_start=False,
            root_height_obs=True,
            humanoid_type="smpl",
        )
        return {"proprio": np.concatenate([v.ravel() for v in obs_dict.values()], axis=0, dtype=np.float64)}

    def set_physics(self, qpos: np.ndarray, qvel: np.ndarray | None = None, ctrl: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos[:]
        if qvel is not None:
            self.data.qvel[:] = qvel[:]
        if ctrl is not None:
            self.data.ctrl[:] = ctrl[:]
        mujoco.mj_forward(self.model, self.data)

    def reset_humanoid(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if self.state_init == StateInit.Default:
            qpos, qvel = humenv.reset.tpose(self.model, self.data, self.np_random)
        elif self.state_init == StateInit.MoCap:
            batch = self.motion_buffer.sample()
            qpos, qvel = batch["qpos"][0], batch["qvel"][0]
        elif self.state_init == StateInit.Fall:
            qpos, qvel = humenv.reset.fall(
                self.model,
                self.data,
                self.np_random,
                self.action_space.shape[0],
                self.action_repeat,
            )
        elif self.state_init == StateInit.DefaultAndFall:
            qpos, qvel = humenv.reset.default_and_fall(
                self.model,
                self.data,
                self.np_random,
                self.fall_prob,
                self.action_space.shape[0],
                self.action_repeat,
            )
        elif self.state_init == StateInit.MoCapAndFall:
            if self.np_random.random() < self.fall_prob:
                qpos, qvel = humenv.reset.fall(
                    self.model,
                    self.data,
                    self.np_random,
                    self.action_space.shape[0],
                    self.action_repeat,
                )
            else:
                batch = self.motion_buffer.sample()
                qpos, qvel = batch["qpos"][0], batch["qvel"][0]

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def render(self) -> np.ndarray | None:
        if self.render_mode is not None:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(
                    self.model,
                    width=self.render_width,
                    height=self.render_height,
                )
                mujoco.mj_forward(self.model, self.data)
            self.renderer.update_scene(self.data, camera=self.camera)
            pixels = self.renderer.render()
            return pixels
        else:
            return None

    def set_task(self, task: humenv.rewards.RewardFunction | str | None) -> None:
        if task is None:
            self.task = humenv.rewards.ZeroReward()
        elif isinstance(task, str):
            self.task = make_from_name(task)
        else:
            self.task = task

    def _init_motion_buffer(self, motion_buffer: MotionBuffer | str | List[str] | None, base_path: str | None) -> None:
        self.motion_buffer = motion_buffer
        if isinstance(motion_buffer, str) or isinstance(motion_buffer, List):
            self.motion_buffer = MotionBuffer(motion_buffer, base_path=base_path)


def compute_humanoid_self_obs_v2(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    upright_start: bool,
    root_height_obs: bool,
    humanoid_type: str,
) -> Dict[str, np.ndarray]:
    body_pos = data.xpos.copy()[_ROBOT_IDX_START:_ROBOT_IDX_END][None,]
    body_rot = data.xquat.copy()[_ROBOT_IDX_START:_ROBOT_IDX_END][None,]
    body_vel = data.sensordata[:_NUM_VEL_LIMIT].reshape(_NUM_RIGID_BODIES, 3).copy()[None,]
    body_ang_vel = data.sensordata[_NUM_VEL_LIMIT : 2 * _NUM_VEL_LIMIT].reshape(_NUM_RIGID_BODIES, 3).copy()[None,]

    obs = OrderedDict()

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    if not upright_start:
        root_rot = humenv.utils.remove_base_rot(root_rot, humanoid_type)

    heading_rot_inv = humenv.utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    if root_height_obs:
        obs["root_h_obs"] = root_h

    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = humenv.utils.quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = humenv.utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = humenv.utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    ###### Velocity ######
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = humenv.utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"] = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = humenv.utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    return obs


def make_from_name(
    name: str | None = None,
):
    all_rewards = inspect.getmembers(sys.modules["humenv.rewards"], inspect.isclass)
    for reward_class_name, reward_cls in all_rewards:
        if not inspect.isabstract(reward_cls):
            reward_obj = reward_cls.reward_from_name(name)
            if reward_obj is not None:
                return reward_obj
    raise ValueError(f"Unknown reward name: {name}")
