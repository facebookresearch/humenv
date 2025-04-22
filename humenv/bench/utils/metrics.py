# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import torch
import numpy as np
import ot


def distance_smpl(next_obs, goal):
    return torch.norm(next_obs[..., :214] - goal[..., :214], dim=-1)


def get_episode_goal_stats(episodes: dict, device="cpu", bound: float = 2.0, margin: float = 2.0) -> torch.tensor:
    stats = defaultdict(list)
    for ep in episodes:
        done_mask = torch.tensor(ep["terminated"] | ep["truncated"], device=device)  # ep.get(("next", "done"))
        next_obs = torch.tensor(ep["observation"][1:], device=device)  # ep.get(("next", "observation"))  # n_parallel x max_steps x S
        assert "goal" in ep.keys(), "the goal state must be added to the rollouts' tensordict"
        goal = torch.tensor(ep["goal"], device=device)  # n_parallel x max_steps x S
        dist = distance_smpl(next_obs, goal)

        in_bounds_mask = dist <= bound
        out_bounds_mask = dist > bound + margin

        success = torch.cumsum(in_bounds_mask, axis=0)
        proximity = torch.cumsum(in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask), axis=0)
        distance = torch.cumsum(dist, axis=0)

        succ_i = success[done_mask.squeeze()]
        succ_i[1:] = succ_i[1:] - succ_i[:-1]
        succ_i = succ_i >= 1
        stats["success"] += succ_i.cpu().detach().numpy().tolist()
        lengths = torch.cat([-torch.ones(1), torch.argwhere(done_mask.squeeze()).reshape(1)], dim=0)
        lengths = lengths[1:] - lengths[:-1]

        for k, v in zip(["proximity", "distance"], [proximity, distance]):
            stat_i = v[done_mask.squeeze()]
            stat_i[1:] = stat_i[1:] - stat_i[:-1]
            stat_i = stat_i / lengths
            stats[k] += stat_i.cpu().detach().numpy().tolist()

        if "id" in ep.keys():
            for el in ep["id"][done_mask.squeeze()]:
                stats["id"].append(el)

    return stats


#####################
# TRACKING METRICS
#####################
def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2):
    stats = {}
    dist = distance_smpl(next_obs, tracking_target)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats["proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats["distance"] = dist.mean()
    stats["success"] = in_bounds_mask.min().float()
    return stats


def get_pose(obs: torch.Tensor):
    return obs[:, :214]


def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    # clamp is needed to avoid negative values due to numerical errors
    return torch.sqrt(torch.clamp(val, min=0))


def emd(next_obs: torch.Tensor, tracking_target: torch.Tensor, device: str = "cpu"):
    # keep only pose part of the observations
    agent_obs = get_pose(next_obs).to(device)
    tracked_obs = get_pose(tracking_target).to(device)
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs)
    X_pot = torch.ones(agent_obs.shape[0], device=agent_obs.device) / agent_obs.shape[0]
    Y_pot = torch.ones(tracked_obs.shape[0], device=agent_obs.device) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {"emd": transport_cost.item()}


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor):
    # keep only pose part of the observations
    agent_obs = get_pose(next_obs).to("cpu")
    tracked_obs = get_pose(tracking_target).to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {"emd": transport_cost}


ROOT_H_OBS = 1
LOCAL_BODY_POS = 69


def phc_metrics(next_obs: torch.Tensor, tracking_target: torch.Tensor):
    """
    Calculation of metrics used in the PHC paper.
    Adapted from: https://github.com/ZhengyiLuo/SMPLSim/blob/main/smpl_sim/smpllib/smpl_eval.py
    """
    stats = {}
    # if self_obs_v == 2 we can get xpos in the following way (it does not contain root)
    xpos_idxs = [ROOT_H_OBS, ROOT_H_OBS + LOCAL_BODY_POS]
    # Next observation should match the desired target (if possible in 1 step)
    jpos_pred = next_obs[:, xpos_idxs[0] : xpos_idxs[1]]  # num_parallel_env x time x 69
    jpos_gt = tracking_target[:, xpos_idxs[0] : xpos_idxs[1]]  # num_parallel_env x time x 69
    # this is global and uses xpos
    stats["mpjpe_g"] = torch.norm(jpos_gt - jpos_pred, dim=1).mean() * 1000

    # we compute the velocity as finite difference
    vel_gt = jpos_gt[1:, :] - jpos_gt[:-1, :]
    vel_pred = jpos_pred[1:, :] - jpos_pred[:-1, :]
    stats["vel_dist"] = torch.norm(vel_pred - vel_gt, dim=1).mean() * 1000

    # Computes acceleration error:
    accel_gt = jpos_gt[:-2, :] - 2 * jpos_gt[1:-1, :] + jpos_gt[2:, :]
    accel_pred = jpos_pred[:-2, :] - 2 * jpos_pred[1:-1:, :] + jpos_pred[2:, :]
    stats["accel_dist"] = torch.norm(accel_pred - accel_gt, dim=1).mean() * 1000

    # the success measure used in PHC
    jpos_pred = jpos_pred.reshape(jpos_pred.shape[0], -1, 3)  # length x 23 x 3
    jpos_gt = jpos_gt.reshape(jpos_gt.shape[0], -1, 3)  # length x 23 x 3
    stats["success_phc_linf"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1) <= 0.5).float()
    stats["success_phc_mean"] = torch.all(torch.norm(jpos_pred - jpos_gt, dim=-1).mean(dim=-1) <= 0.5).float()

    return stats
