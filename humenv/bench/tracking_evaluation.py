# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence, Callable
import dataclasses
import gymnasium
import numpy as np
from gymnasium.vector.utils import CloudpickleWrapper
import functools
import torch
from humenv.misc.motionlib import MotionBuffer
from humenv.bench.utils.metrics import distance_proximity, emd, phc_metrics
from humenv.bench.gym_utils.episodes import Episode
from humenv import make_humenv, CustomManager
from concurrent.futures import ThreadPoolExecutor
from packaging.version import Version
from tqdm import tqdm

if Version("0.26") <= Version(gymnasium.__version__) < Version("1.0"):
    cast_obs_wrapper = lambda env: gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32))
else:
    cast_obs_wrapper = lambda env: gymnasium.wrappers.TransformObservation(env, lambda obs: obs.astype(np.float32), env.observation_space)


@dataclasses.dataclass(kw_only=True)
class TrackingEvaluation:
    motions: str | List[str]
    motion_base_path: str | None = None
    # environment parameters
    num_envs: int = 1
    wrappers: Sequence[Callable[[gymnasium.Env], gymnasium.Wrapper]] = dataclasses.field(
        default_factory=lambda: [gymnasium.wrappers.FlattenObservation, cast_obs_wrapper]
    )
    env_kwargs: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_envs > 1:
            self.mp_manager = CustomManager()
            self.mp_manager.start()
            self.motion_buffer = self.mp_manager.MotionBuffer(
                files=self.motions, base_path=self.motion_base_path, keys=["qpos", "qvel", "observation"]
            )
        else:
            self.mp_manager = None
            self.motion_buffer = MotionBuffer(files=self.motions, base_path=self.motion_base_path, keys=["qpos", "qvel", "observation"])

    def run(self, agent: Any) -> Dict[str, Any]:
        metrics = {}
        ids = self.motion_buffer.get_motion_ids()
        num_workers = min(self.num_envs, len(ids))
        motions_per_worker = np.array_split(ids, num_workers)
        f = functools.partial(
            _async_tracking_worker,
            env_fn=CloudpickleWrapper(lambda: make_humenv(num_envs=1, wrappers=self.wrappers, **self.env_kwargs)),
            agent=agent,
            motion_buffer=self.motion_buffer,
        )
        if num_workers == 1:
            results = f((motions_per_worker[0], 0))
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                inputs = [(x, y) for x, y in zip(motions_per_worker, range(len(motions_per_worker)))]
                list_res = pool.map(f, inputs)
                results = []
                for el in list_res:
                    results.extend(el)
        # Compute metrics
        for ep in results:
            metr = {}
            next_obs = torch.tensor(ep["observation"][1:], dtype=torch.float32)
            tracking_target = torch.tensor(ep["tracking_target"], dtype=torch.float32)
            dist_prox_res = distance_proximity(next_obs=next_obs, tracking_target=tracking_target)
            metr.update(dist_prox_res)
            emd_res = emd(next_obs=next_obs, tracking_target=tracking_target)
            metr.update(emd_res)
            phc_res = phc_metrics(next_obs=next_obs, tracking_target=tracking_target)
            metr.update(phc_res)
            for k, v in metr.items():
                if isinstance(v, torch.Tensor):
                    metr[k] = v.tolist()
            metr["motion_id"] = ep["motion_id"]
            metrics[ep["motion_file"]] = metr
        return metrics

    def close(self) -> None:
        if self.mp_manager is not None:
            self.mp_manager.shutdown()


def _async_tracking_worker(inputs, env_fn, agent: Any, motion_buffer: MotionBuffer):
    motion_ids, pos = inputs
    env = env_fn()[0]
    episodes = []
    for m_id in tqdm(motion_ids, position=pos, leave=False):
        ep_ = motion_buffer.get(m_id)
        # we ignore the first state since we need to pass the next observation
        tracking_target = ep_["observation"][1:]
        ctx = agent.tracking_inference(next_obs=tracking_target)
        ctx = [None] * tracking_target.shape[0] if ctx is None else ctx
        observation, info = env.reset(options={"qpos": ep_["qpos"][0], "qvel": ep_["qvel"][0]})
        _episode = Episode()
        _episode.initialise(observation, info)
        for i in range(len(ctx)):
            action = agent.act(observation, ctx[i])
            observation, reward, terminated, truncated, info = env.step(action)
            _episode.add(observation, reward, action, terminated, truncated, info)
        tmp = _episode.get()
        tmp["tracking_target"] = tracking_target
        tmp["motion_id"] = m_id
        tmp["motion_file"] = motion_buffer.get_name(m_id)
        episodes.append(tmp)
    env.close()
    return episodes
