"""
arm_env.py

Gymnasium environment for the Kinova Gen3 pick/reach task.
Works identically whether backed by RealArmBackend or MockArmBackend —
see arm_backend.py.
"""

import gymnasium as gym
import numpy as np


class KinovaArmEnv(gym.Env):
    JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4',
                   'joint_5', 'joint_6', 'joint_7']

    # Kinova Gen3 joint limits (radians) — confirm against your URDF later
    JOINT_LOWER = np.array([-2.41] * 7, dtype=np.float32)
    JOINT_UPPER = np.array([2.41] * 7, dtype=np.float32)

    MAX_ACTION_DELTA = 0.1   # radians per step — caps how far a single action can move a joint
    MAX_EPISODE_STEPS = 200
    SUCCESS_DISTANCE = 0.05  # meters

    def __init__(self, backend, target_position=None):
        super().__init__()
        self.backend = backend  # <-- dependency injection: Real or Mock, doesn't matter here

        # Observation: 7 joint pos + 7 joint vel + 3 ee pos + 3 target pos = 20
        obs_low = np.concatenate([
            self.JOINT_LOWER, -np.ones(7) * 5.0, -np.ones(3) * 2.0, -np.ones(3) * 2.0
        ])
        obs_high = np.concatenate([
            self.JOINT_UPPER, np.ones(7) * 5.0, np.ones(3) * 2.0, np.ones(3) * 2.0
        ])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: per-joint delta, not absolute position.
        # Why deltas? Absolute-position actions require the policy to output
        # the "right" joint angle from scratch every step, which is a much
        # harder function to learn. Deltas let it nudge incrementally toward
        # the goal, closer to how a PID controller thinks — easier to explore
        # and easier for PPO's Gaussian action noise to behave sensibly.
        self.action_space = gym.spaces.Box(
            low=-self.MAX_ACTION_DELTA, high=self.MAX_ACTION_DELTA,
            shape=(7,), dtype=np.float32
        )

        self._fixed_target = np.array(target_position, dtype=np.float32) if target_position is not None else None
        self.target_position = None
        self.current_positions = np.zeros(7, dtype=np.float32)
        self.step_count = 0

    def _sample_target(self):
        if self._fixed_target is not None:
            return self._fixed_target.copy()
        # Random reachable-ish point in front of the arm.
        # Tighten/loosen this box once you know real reachable workspace.
        return np.array([
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.2, 0.5),
        ], dtype=np.float32)

    def _get_obs(self):
        pos, vel = self.backend.get_joint_state()
        ee_pos, _ = self.backend.get_ee_pose()
        if ee_pos is None:
            ee_pos = np.zeros(3, dtype=np.float32)  # fallback if TF not ready yet
        return np.concatenate([pos, vel, ee_pos, self.target_position]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.backend.reset()
        self.step_count = 0
        self.target_position = self._sample_target()

        # Let a few spins pass so joint_state/TF actually reflect the reset
        for _ in range(5):
            self.backend.spin(timeout_sec=0.1)

        self.current_positions, _ = self.backend.get_joint_state()
        obs = self._get_obs()
        info = {}
        return obs, info

from reward import compute_reward, RewardConfig

def step(self, action):
    self.step_count += 1

    target = np.clip(
        self.current_positions + action,
        self.JOINT_LOWER, self.JOINT_UPPER
    )
    self.backend.send_joint_target(target, duration_sec=0.5)
    self.backend.spin(timeout_sec=0.5)

    self.current_positions, _ = self.backend.get_joint_state()
    ee_pos, _ = self.backend.get_ee_pose()
    obs = self._get_obs()

    # Fallback if TF/pose isn't available yet -- keeps the reward
    # function from crashing on a None, at the cost of a "fake far away"
    # distance for that one step.
    if ee_pos is None:
        ee_pos = self.target_position + 10.0  # guarantees a large distance, not a crash

    reward, reward_breakdown, terminated = compute_reward(
        ee_position=ee_pos,
        target_position=self.target_position,
        joint_positions=self.current_positions,
        joint_lower=self.JOINT_LOWER,
        joint_upper=self.JOINT_UPPER,
        action=action,
    )
    truncated = self.step_count >= self.MAX_EPISODE_STEPS

    info = {'distance': reward_breakdown['distance'], 'reward_breakdown': reward_breakdown}
    return obs, reward, terminated, truncated, info


    def close(self):
        pass  # backend cleanup (rclpy.shutdown etc.) handled by whoever created it
"""
arm_env.py

Gymnasium environment for the Kinova Gen3 pick/reach task.
Works identically whether backed by RealArmBackend or MockArmBackend —
see arm_backend.py.
"""

import gymnasium as gym
import numpy as np


class KinovaArmEnv(gym.Env):
    JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4',
                   'joint_5', 'joint_6', 'joint_7']

    # Kinova Gen3 joint limits (radians) — confirm against your URDF later
    JOINT_LOWER = np.array([-2.41] * 7, dtype=np.float32)
    JOINT_UPPER = np.array([2.41] * 7, dtype=np.float32)

    MAX_ACTION_DELTA = 0.1   # radians per step — caps how far a single action can move a joint
    MAX_EPISODE_STEPS = 200
    SUCCESS_DISTANCE = 0.05  # meters

    def __init__(self, backend, target_position=None):
        super().__init__()
        self.backend = backend  # <-- dependency injection: Real or Mock, doesn't matter here

        # Observation: 7 joint pos + 7 joint vel + 3 ee pos + 3 target pos = 20
        obs_low = np.concatenate([
            self.JOINT_LOWER, -np.ones(7) * 5.0, -np.ones(3) * 2.0, -np.ones(3) * 2.0
        ])
        obs_high = np.concatenate([
            self.JOINT_UPPER, np.ones(7) * 5.0, np.ones(3) * 2.0, np.ones(3) * 2.0
        ])
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action: per-joint delta, not absolute position.
        # Why deltas? Absolute-position actions require the policy to output
        # the "right" joint angle from scratch every step, which is a much
        # harder function to learn. Deltas let it nudge incrementally toward
        # the goal, closer to how a PID controller thinks — easier to explore
        # and easier for PPO's Gaussian action noise to behave sensibly.
        self.action_space = gym.spaces.Box(
            low=-self.MAX_ACTION_DELTA, high=self.MAX_ACTION_DELTA,
            shape=(7,), dtype=np.float32
        )

        self._fixed_target = np.array(target_position, dtype=np.float32) if target_position is not None else None
        self.target_position = None
        self.current_positions = np.zeros(7, dtype=np.float32)
        self.step_count = 0

    def _sample_target(self):
        if self._fixed_target is not None:
            return self._fixed_target.copy()
        # Random reachable-ish point in front of the arm.
        # Tighten/loosen this box once you know real reachable workspace.
        return np.array([
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.2, 0.5),
        ], dtype=np.float32)

    def _get_obs(self):
        pos, vel = self.backend.get_joint_state()
        ee_pos, _ = self.backend.get_ee_pose()
        if ee_pos is None:
            ee_pos = np.zeros(3, dtype=np.float32)  # fallback if TF not ready yet
        return np.concatenate([pos, vel, ee_pos, self.target_position]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.backend.reset()
        self.step_count = 0
        self.target_position = self._sample_target()

        # Let a few spins pass so joint_state/TF actually reflect the reset
        for _ in range(5):
            self.backend.spin(timeout_sec=0.1)

        self.current_positions, _ = self.backend.get_joint_state()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.step_count += 1

        target = np.clip(
            self.current_positions + action,
            self.JOINT_LOWER, self.JOINT_UPPER
        )
        self.backend.send_joint_target(target, duration_sec=0.5)
        self.backend.spin(timeout_sec=0.5)

        self.current_positions, _ = self.backend.get_joint_state()
        ee_pos, _ = self.backend.get_ee_pose()
        obs = self._get_obs()

        distance = np.linalg.norm(ee_pos - self.target_position) if ee_pos is not None else 10.0
        reward = -distance - 0.01  # closer = better, small per-step cost to encourage speed
        terminated = bool(distance < self.SUCCESS_DISTANCE)
        if terminated:
            reward += 100.0
        truncated = self.step_count >= self.MAX_EPISODE_STEPS

        info = {'distance': distance}
        return obs, reward, terminated, truncated, info

    def close(self):
        pass  # backend cleanup (rclpy.shutdown etc.) handled by whoever created it

