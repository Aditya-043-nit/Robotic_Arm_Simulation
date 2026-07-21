"""
arm_backend.py

Defines the interface arm_env.py depends on, plus two implementations:
- RealArmBackend: talks to actual ROS2/Gazebo (TrajectoryClient + RobotState)
- MockArmBackend: simulates joint motion in pure Python, no ROS2 needed

arm_env.py only ever calls methods on this interface, so swapping
backends is a one-line change in how the environment is constructed.
"""

from abc import ABC, abstractmethod
import numpy as np


class ArmBackend(ABC):
    """Abstract interface: anything arm_env.py needs from 'the robot'."""

    @abstractmethod
    def reset(self):
        """Return the arm to a known starting configuration."""
        ...

    @abstractmethod
    def send_joint_target(self, positions: np.ndarray, duration_sec: float) -> bool:
        """Command the arm toward the given joint angles. Returns success."""
        ...

    @abstractmethod
    def get_joint_state(self):
        """Returns (positions, velocities) as np.float32 arrays."""
        ...

    @abstractmethod
    def get_ee_pose(self):
        """Returns (position_xyz, quaternion_xyzw), or (None, None) if unavailable."""
        ...

    @abstractmethod
    def spin(self, timeout_sec: float):
        """Let one control-loop tick pass (process ROS callbacks, or advance fake time)."""
        ...


class RealArmBackend(ArmBackend):
    """Wraps your existing TrajectoryClient + RobotState over live ROS2."""

    def __init__(self, node, joint_names, base_frame='base_link', ee_frame='tool_frame'):
        # Import here, not at module top, so MockArmBackend can be used
        # in environments without rclpy installed at all (e.g. quick unit tests).
        from trajectory_client import TrajectoryClient
        from robot_state import RobotState
        import rclpy

        self._rclpy = rclpy
        self.node = node
        self.trajectory_client = TrajectoryClient(node, joint_names)
        self.robot_state = RobotState(node, joint_names, base_frame, ee_frame)

    def reset(self):
        # Send the arm home. In a later iteration you'll likely also call
        # Gazebo's /reset_simulation service here to reset object poses etc.
        home = [0.0] * len(self.trajectory_client.joint_names)
        self.send_joint_target(np.array(home), duration_sec=3.0)

    def send_joint_target(self, positions, duration_sec=1.0):
        return self.trajectory_client.send_goal(positions.tolist(), duration_sec)

    def get_joint_state(self):
        return self.robot_state.get_joint_state()

    def get_ee_pose(self):
        return self.robot_state.get_ee_pose()

    def spin(self, timeout_sec=0.1):
        self._rclpy.spin_once(self.node, timeout_sec=timeout_sec)


class MockArmBackend(ArmBackend):
    """
    Simulates a 7-DOF arm with no ROS2/Gazebo involved.

    Physically crude on purpose: joints move toward the commanded target
    at a fixed max angular speed, one "spin" = one fixed timestep of motion.
    This is NOT meant to be physically accurate — it exists so arm_env.py's
    reset/step/reward logic can be built and tested today, independent of
    whatever's broken in the Gazebo build right now.
    """

    def __init__(self, joint_names, joint_limits_lower, joint_limits_upper,
                 max_joint_speed=1.0, dt=0.1):
        self.joint_names = joint_names
        self.n_joints = len(joint_names)
        self.lower = np.array(joint_limits_lower, dtype=np.float32)
        self.upper = np.array(joint_limits_upper, dtype=np.float32)
        self.max_joint_speed = max_joint_speed  # rad/s
        self.dt = dt                             # seconds per spin() call

        self._positions = np.zeros(self.n_joints, dtype=np.float32)
        self._velocities = np.zeros(self.n_joints, dtype=np.float32)
        self._target = np.zeros(self.n_joints, dtype=np.float32)

    def reset(self):
        self._positions[:] = 0.0
        self._velocities[:] = 0.0
        self._target[:] = 0.0

    def send_joint_target(self, positions, duration_sec=1.0):
        self._target = np.clip(positions, self.lower, self.upper)
        return True  # mock always "accepts" the goal

    def get_joint_state(self):
        return self._positions.copy(), self._velocities.copy()

    def get_ee_pose(self):
        # Crude placeholder forward kinematics — not physically meaningful,
        # just needs to move smoothly and consistently so reward shaping
        # can be developed. Swap for real TF lookups once on RealArmBackend.
        q = self._positions
        x = 0.4 * np.cos(q[0]) * np.cos(q[1])
        y = 0.4 * np.sin(q[0]) * np.cos(q[1])
        z = 0.4 * np.sin(q[1]) + 0.3
        position = np.array([x, y, z], dtype=np.float32)
        quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # identity
        return position, quaternion

    def spin(self, timeout_sec=0.1):
        # Move each joint toward its target at max_joint_speed, capped by dt.
        error = self._target - self._positions
        max_step = self.max_joint_speed * self.dt
        step = np.clip(error, -max_step, max_step)
        self._velocities = step / self.dt
        self._positions = self._positions + step

