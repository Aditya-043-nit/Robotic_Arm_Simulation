"""
reward.py

Reward computation for the Kinova Gen3 reach/pick task, pulled out of
arm_env.py so it can be tuned and inspected independently.

Design: a single function that returns both the scalar reward AND a
breakdown dict. The breakdown costs almost nothing to compute and is
invaluable once you're staring at a flat/declining reward curve in
TensorBoard trying to figure out which term is the problem.
"""

import numpy as np


class RewardConfig:
    """
    All reward magic numbers live here, not scattered through the function.
    Tune these first when reward shaping needs adjusting -- don't touch
    the logic below unless you're changing the reward's actual structure.
    """
    SUCCESS_DISTANCE = 0.05       # meters -- how close counts as "reached"
    SUCCESS_BONUS = 100.0
    DISTANCE_WEIGHT = 1.0         # scales the -distance term
    STEP_PENALTY = 0.01           # small constant cost per step, encourages speed
    JOINT_LIMIT_MARGIN = 0.1      # rad -- how close to a limit before penalizing
    JOINT_LIMIT_PENALTY = 5.0
    ACTION_MAGNITUDE_WEIGHT = 0.0 # start at 0; enable later if motion is too jerky


def compute_reward(ee_position, target_position, joint_positions,
                    joint_lower, joint_upper, action, config=RewardConfig):
    """
    Returns (reward: float, breakdown: dict, terminated: bool)

    breakdown lets you log each term separately, e.g. to TensorBoard via
    a custom SB3 callback -- so you can see "oh, the joint limit penalty
    is dominating everything" instead of just a mysterious flat reward.
    """
    breakdown = {}

    # --- Distance term: the main learning signal ---
    # Negative distance means reward increases as the arm gets closer.
    # This is a "dense" reward -- feedback every single step, not just
    # at success/failure. Dense rewards are much easier for PPO to learn
    # from than sparse ones (reward only at the very end), especially
    # early in training when the policy is essentially acting randomly.
    distance = float(np.linalg.norm(ee_position - target_position))
    breakdown['distance'] = distance
    breakdown['distance_reward'] = -config.DISTANCE_WEIGHT * distance

    # --- Step penalty: discourages stalling/wandering ---
    breakdown['step_penalty'] = -config.STEP_PENALTY

    # --- Joint limit penalty: discourages the policy from driving joints
    # into their mechanical limits, which in real hardware would be
    # damaging and in Gazebo often causes unstable/glitchy physics ---
    near_lower = joint_positions < (joint_lower + config.JOINT_LIMIT_MARGIN)
    near_upper = joint_positions > (joint_upper - config.JOINT_LIMIT_MARGIN)
    joints_near_limit = int(np.sum(near_lower | near_upper))
    breakdown['joint_limit_penalty'] = -config.JOINT_LIMIT_PENALTY * joints_near_limit

    # --- Action magnitude penalty (optional, off by default) ---
    # Enable this later if the trained policy moves the arm too jerkily --
    # penalizing large actions encourages smoother motion. Left at 0 for
    # now because adding too many reward terms at once makes it hard to
    # tell which one is actually shaping behavior.
    action_penalty = -config.ACTION_MAGNITUDE_WEIGHT * float(np.linalg.norm(action))
    breakdown['action_penalty'] = action_penalty

    # --- Success bonus: sparse, but large ---
    terminated = distance < config.SUCCESS_DISTANCE
    breakdown['success_bonus'] = config.SUCCESS_BONUS if terminated else 0.0

    reward = sum(breakdown[k] for k in
                 ('distance_reward', 'step_penalty', 'joint_limit_penalty',
                  'action_penalty', 'success_bonus'))
    breakdown['total'] = reward

    return reward, breakdown, terminated


