# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.


from typing import Optional, Tuple

import torch


@torch.no_grad()
def compute_symmetric_states(
    env=None,
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    obs_type: str = "policy",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Get symmetric (mirrored) states for K1 robot.

    This function mirrors observations and actions by swapping left/right limbs
    and flipping appropriate signs for roll and yaw components.

    The function signature matches the expected interface from RslRlSymmetryCfg.
    The actual implementation uses 'actions' (plural) to match the calling convention.

    Args:
        env: The environment object (VecEnv). Used to access environment properties if needed.
        obs: The observation tensor of shape [batch_size, obs_dim] (flattened history).
            Can be None if only actions need to be mirrored.
        actions: The action tensor of shape [batch_size, action_dim].
            Can be None if only observations need to be mirrored.
        obs_type: Type of observation - "policy" for actor obs, "critic" for critic obs.

    Returns:
        Tuple of (mirrored_obs, mirrored_actions) where each contains both
        original and mirrored data concatenated along batch dimension.
        If obs is None, returns (None, mirrored_actions).
        If actions is None, returns (mirrored_obs, None).
    """
    # Constants for K1 observation structure
    ACTOR_OBS_DIM = 57  # ang_vel(3) + projected_gravity(3) + command(3) + joint_pos(16) + joint_vel(16) + action(16)
    CRITIC_OBS_DIM = 64  # 60 # 64 # root_lin_vel(3) + actor_obs(57) + feet_contact(4)
    HISTORY_LENGTH = 10

    mirrored_obs = None
    mirrored_actions = None

    # Process observations
    if obs is not None:
        batch_size = obs.shape[0]

        # Determine observation dimensions based on obs_type
        if obs_type == "policy":
            obs_dim = ACTOR_OBS_DIM
            is_critic = False
        elif obs_type == "critic":
            obs_dim = CRITIC_OBS_DIM
            is_critic = True
        else:
            raise ValueError(f"Unexpected observation type: {obs_type}. Expected 'policy' or 'critic'.")

        # Unflatten observation history: [batch, history * obs_dim] -> [batch, history, obs_dim]
        obs_unflattened = obs.view(batch_size, HISTORY_LENGTH, obs_dim)

        # Create mirrored observations
        mirrored_obs_unflattened = obs_unflattened.clone()

        # Mirror each timestep in history
        for t in range(HISTORY_LENGTH):
            obs_t = obs_unflattened[:, t, :]
            # mirrored_obs_t = mirrored_obs_unflattened[:, t, :]

            if is_critic:
                # Critic obs: [root_lin_vel(3), actor_obs(47), feet_contact(2)]
                mirrored_obs_unflattened[:, t, :] = mirror_critic_obs(obs_t)
            else:
                # Actor obs: [ang_vel(3), projected_gravity(3), command(3), joint_pos(12), joint_vel(12), action(12), angle_diff(1), cmd_is_zero(1)]
                mirrored_obs_unflattened[:, t, :] = mirror_actor_obs(obs_t)

        # Flatten back: [batch, history, obs_dim] -> [batch, history * obs_dim]
        mirrored_obs = mirrored_obs_unflattened.view(batch_size, HISTORY_LENGTH * obs_dim)

        # Concatenate original and mirrored observations
        mirrored_obs = torch.cat([obs, mirrored_obs], dim=0)

    # Process actions
    if actions is not None:
        batch_size = actions.shape[0]
        mirrored_actions = mirror_actions(actions)

        # Concatenate original and mirrored actions
        mirrored_actions = torch.cat([actions, mirrored_actions], dim=0)

    return mirrored_obs, mirrored_actions


def mirror_critic_obs(critic_obs: torch.Tensor) -> torch.Tensor:
    """Mirror critic observation by swapping left/right and flipping signs.

    Critic obs structure: [
        root_lin_vel(3),
        actor_obs(47),
        feet_contact(4)
    ]

    Args:
        critic_obs: Critic observation tensor of shape [batch, 52]

    Returns:
        Mirrored critic observation tensor
    """
    mirrored = critic_obs.clone()

    # Mirror root_lin_vel: flip y-component (index 1)
    mirrored[:, 1] = -mirrored[:, 1]

    ACTOR_OBS_DIM = 57

    # Mirror actor obs part (indices 3:50)
    actor_obs = critic_obs[:, 3 : 3 + ACTOR_OBS_DIM]
    mirrored_actor_obs = mirror_actor_obs(actor_obs)
    mirrored[:, 3 : 3 + ACTOR_OBS_DIM] = mirrored_actor_obs

    # Mirror feet_contact: swap left(0) and right(1)
    mirrored[:, 3 + ACTOR_OBS_DIM + 0] = critic_obs[:, 3 + ACTOR_OBS_DIM + 1]  # left <- right
    mirrored[:, 3 + ACTOR_OBS_DIM + 1] = critic_obs[:, 3 + ACTOR_OBS_DIM + 0]  # right <- left
    mirrored[:, 3 + ACTOR_OBS_DIM + 2] = critic_obs[:, 3 + ACTOR_OBS_DIM + 3]  # left <- right
    mirrored[:, 3 + ACTOR_OBS_DIM + 3] = critic_obs[:, 3 + ACTOR_OBS_DIM + 2]  # right <- left

    return mirrored


def mirror_actor_obs(actor_obs: torch.Tensor) -> torch.Tensor:
    """Mirror actor observation by swapping left/right and flipping signs.

    Actor obs structure: [
        ang_vel(3),
        projected_gravity(3),
        command(3),
        joint_pos(16),
        joint_vel(16),
        action(16),
    ]

    Args:
        actor_obs: Actor observation tensor of shape [batch, 47]

    Returns:
        Mirrored actor observation tensor
    """
    NUM_JOINTS = 16

    mirrored = actor_obs.clone()

    # Mirror ang_vel: flip y (index 1) and z (index 2) components
    mirrored[:, 0] = -mirrored[:, 0]  # flip y (roll)
    mirrored[:, 2] = -mirrored[:, 2]  # flip z (yaw)

    # Mirror projected_gravity: flip y-component (index 1)
    mirrored[:, 4] = -mirrored[:, 4]

    # Mirror command: flip y (index 1) and z (index 2) components
    # command is [lin_vel_x, lin_vel_y, ang_vel_z]
    mirrored[:, 7] = -mirrored[:, 7]  # flip lin_vel_y
    mirrored[:, 8] = -mirrored[:, 8]  # flip ang_vel_z

    # Mirror joint positions (indices 9:21)
    mirrored[:, 9 : 9 + NUM_JOINTS] = mirror_joints(mirrored[:, 9 : 9 + NUM_JOINTS])

    # Mirror joint velocities (indices 21:33)
    mirrored[:, 9 + NUM_JOINTS : 9 + NUM_JOINTS * 2] = mirror_joints(mirrored[:, 9 + NUM_JOINTS : 9 + NUM_JOINTS * 2])

    # Mirror actions (indices 33:45)
    mirrored[:, 9 + NUM_JOINTS * 2 : 9 + NUM_JOINTS * 3] = mirror_joints(mirrored[:, 9 + NUM_JOINTS * 2 : 9 + NUM_JOINTS * 3])

    return mirrored


def mirror_joints(joint_data: torch.Tensor) -> torch.Tensor:
    """Mirror joint data by swapping left/right limbs and applying flip masks.

    Joint order: [left_arm(0-3), right_arm(4-7), left_leg(8-13), right_leg(14-19)]

    Args:
        joint_data: Joint data tensor of shape [batch, 12]

    Returns:
        Mirrored joint data tensor
    """
    mirrored = joint_data.clone()
    mirrored_temp = joint_data.clone()

    leg_flip_mask = torch.tensor([-1.0, 1.0, 1.0, 1.0]).to(joint_data.device)

    # "isaac_to_mujoco_idx": [3, 7, 11, 15, 1, 5, 9, 13, 2, 6, 10, 14, 0, 4, 8, 12],
    # "mujoco_to_isaac_idx": [12, 4, 8, 0, 13, 5, 9, 1, 14, 6, 10, 2, 15, 7, 11, 3],
    urdf_to_lab = torch.tensor([3, 7, 11, 15, 1, 5, 9, 13, 2, 6, 10, 14, 0, 4, 8, 12]).to(joint_data.device)
    lab_to_urdf = torch.tensor([12, 4, 8, 0, 13, 5, 9, 1, 14, 6, 10, 2, 15, 7, 11, 3]).to(joint_data.device)

    mirrored = mirrored[:, lab_to_urdf]
    mirrored_temp = mirrored_temp[:, lab_to_urdf]

    # Swap left leg (0-5) and right leg (6-11), then apply flip mask
    mirrored[:, 0:4] = mirrored_temp[:, 4:8] * leg_flip_mask
    mirrored[:, 4:8] = mirrored_temp[:, 0:4] * leg_flip_mask
    mirrored[:, 8:12] = mirrored_temp[:, 12:16] * leg_flip_mask
    mirrored[:, 12:16] = mirrored_temp[:, 8:12] * leg_flip_mask

    mirrored = mirrored[:, urdf_to_lab]

    return mirrored


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror actions by swapping left/right limbs and applying flip masks.

    Args:
        actions: Action tensor of shape [batch, 12]

    Returns:
        Mirrored action tensor
    """
    return mirror_joints(actions)
