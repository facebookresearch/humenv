# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Any, Dict, Sequence, Callable
import dataclasses
from .gym_utils.rollouts import rollout
import tqdm
from humenv import make_humenv
from humenv.bench import convert_dict_of_lists
import gymnasium


@dataclasses.dataclass(kw_only=True)
class RewardEvaluation:
    tasks: List[str]
    motion_base_path: str | None = None
    motions: str | List[str] | None = None
    num_contexts: int = 1
    num_episodes: int = 1
    # environment parameters
    num_envs: int = 1
    vectorization_mode: str = "async"
    wrappers: Sequence[Callable[[gymnasium.Env], gymnasium.Wrapper]] = dataclasses.field(
        default_factory=lambda: [gymnasium.wrappers.FlattenObservation]
    )
    env_kwargs: dict = dataclasses.field(default_factory=dict)

    def run(self, agent: Any) -> Dict[str, Any]:
        metrics = {}
        self.env_kwargs["task"] = None
        penv, manager = make_humenv(
            num_envs=self.num_envs,
            vectorization_mode=self.vectorization_mode,
            motions=self.motions,
            motion_base_path=self.motion_base_path,
            wrappers=self.wrappers,
            **self.env_kwargs,
        )

        def reset_task(task):
            if not isinstance(penv, (gymnasium.vector.AsyncVectorEnv, gymnasium.vector.SyncVectorEnv)):
                penv.unwrapped.set_task(task)
            else:
                penv.call("set_task", task)

        pbar = tqdm.tqdm(self.tasks, leave=False)
        for task in pbar:
            pbar.set_description(f"task {task}")
            reset_task(task)
            local_stats = []
            for _ in range(self.num_contexts):
                pbar.set_description(f"task {task} (inference)")
                ctx = agent.reward_inference(task=task)
                pbar.set_description(f"task {task} (rollout)")
                ctx = [None] * self.num_envs if ctx is None else ctx.repeat(self.num_envs, 1)
                st, _ = rollout(
                    penv,
                    agent=agent,
                    num_episodes=self.num_episodes,
                    ctx=ctx,
                )  # return statistics and episodes
                local_stats.append(st)
            local_stats = convert_dict_of_lists(local_stats)
            metrics[task] = local_stats
        penv.close()
        if manager is not None:
            manager.shutdown()
        return metrics
