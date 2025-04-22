# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium
import numpy as np
import pytest

from humenv import make_humenv


@pytest.mark.parametrize("num_envs", [2, 5])
@pytest.mark.parametrize("seed", [None, 1])
@pytest.mark.parametrize("state_init", ["Default", "Fall"])
def test_make_humenv(num_envs, seed, state_init):
    penv, _ = make_humenv(
        num_envs=num_envs,
        vectorization_mode="sync",
        motions=None,
        motion_base_path=None,
        wrappers=[gymnasium.wrappers.FlattenObservation],
        state_init=state_init,
    )
    for k in range(2):
        observations, infos = penv.reset(seed=seed)
        for j in range(50):
            for i in range(len(observations) - 1):
                if state_init == "Default":
                    np.testing.assert_allclose(observations[i], observations[i + 1])
                else:
                    assert not np.allclose(observations[i], observations[i + 1])
            actions = penv.action_space.sample()
            # always use the same action:
            actions = np.repeat(actions[0][np.newaxis, :], num_envs, axis=0)
            observations, rewards, terminations, truncations, infos = penv.step(actions)
    penv.close()
