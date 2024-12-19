

<div id="user-content-toc">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1> HumEnv: Humanoid Environment for Reinforcement Learning</h1>
    </summary>
  </ul>
</div>

# Overview

HumEnv is an environment based on SMPL humanoid which aims for reproducible studies of humanoid control. It is designed to facilitate algorithmic research on reinforcement learning (RL), goal-based RL, unsupervised RL, and imitation learning. It consists of a basic environment interface, as well as an optional benchmark to evaluate agents on different tasks.

## Features

 * An environment that enables simulation of a realistic humanoid on a range of proprioceptive tasks
 * A MuJoCo-based humanoid robot definition tuned for more realistic behaviors (friction, joint actuation, and movement range) 
 * 9 configurable reward classes to enable learning basic humanoid skills, including locomotion, spinning, jumping, crawling, and more
 * Benchmarking code to evaluate RL agents on three classes of tasks: reward-based, goal-reaching and motion tracking
 * Various initialisation options: a static "T-pose", random fall, frame from MoCap data, and their combinations
 * Full compatibility with Gymnasium 

# Installation

Basic installation with full support of the environment functionalities (it requires Python 3.9+):

```bash
pip install "git+https://github.com/facebookresearch/HumEnv.git"
```

To use the MoCap and MoCapAndFall initalization schemes, you must prepare licensed datasets according to [these instructions](data_preparation/README.md).

Full installation that includes all the benchmarking features:

```bash
pip install "humenv[bench] @ git+https://github.com/facebookresearch/HumEnv.git"
```

# Quickstart

Once installed, you can create an environment using `humenv.make_humenv` which has a similar interface as `gymnasium.make_vec`.
Here is a simple example:

```python
from humenv import make_humenv
env, _ = make_humenv()
observation, info = env.reset()
frames = [env.render()]
for i in range(60):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
    frames.append(env.render())
# render frames at 30fps
```

More examples are available in the [tutorial](tutorial.ipynb).

# Citation
```
@article{tirinzoni2024metamotivo,
  title={Zero-shot Whole-Body Humanoid Control via Behavioral Foundation Models},
  author={Tirinzoni, Andrea and Touati, Ahmed and Farebrother, Jesse and Guzek, Mateusz and Kanervisto, Anssi and Xu, Yingchen and Lazaric, Alessandro and Pirotta, Matteo},
}
```

# Acknowledgments

 * [SMPL](https://smpl.is.tue.mpg.de/) and [AMASS](https://amass.is.tue.mpg.de/) for the humanoid skeleton and motions used to initialise realistic positions for the tracking benchmark
 * [PHC](https://github.com/ZhengyiLuo/PHC) for the data process and calculation of some Goal-reaching metrics
 * [SMPLSm](https://github.com/ZhengyiLuo/SMPLSim) for scripts used to process SMPL and AMASS datasets, and for humanoid processing utils
 * [smplx](https://github.com/vchoutas/smplx.git) for removing chumpy dependency
 * [MuJoCo](https://github.com/google-deepmind/mujoco) for the backend simulation engine
 * [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) for the API 

# License

Humenv is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.
