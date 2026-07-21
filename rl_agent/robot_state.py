"""
Reads the arm's current state:
- Joint positions and velocities (from /joint_states)
- End-effector pose (from the TF tree)

Design note: takes an existing rclpy Node rather than creating one,
so it can share a single node (and a single spin loop) with
TrajectoryClient inside arm_env.py.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import tf2_ros
from tf2_ros import TransformException


class RobotState:
    def __init__(self, node: Node, joint_names: list,
                 base_frame: str = 'base_link',
                 ee_frame: str = 'tool_frame'):
        self.node = node
        self.joint_names = joint_names
        self.base_frame = base_frame
        self.ee_frame = ee_frame

        # --- Joint state storage ---
        # Initialized to zeros; real values arrive asynchronously
        # via the subscription callback below.
        self._positions = np.zeros(len(joint_names), dtype=np.float32)
        self._velocities = np.zeros(len(joint_names), dtype=np.float32)
        self._joint_state_received = False

        self.node.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10  # QoS queue depth
        )

        # --- TF2 setup for end-effector pose ---
        # Buffer stores recent transforms; Listener subscribes to
        # /tf and /tf_static and fills the buffer automatically.
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self.node)

    def _joint_state_callback(self, msg: JointState):
        """
        Called automatically whenever a new /joint_states message arrives.
        We can't assume msg.name is in the same order as self.joint_names,
        so we look up each joint's index explicitly.
        """
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        for i, jname in enumerate(self.joint_names):
            if jname in name_to_index:
                idx = name_to_index[jname]
                self._positions[i] = msg.position[idx]
                # Velocity isn't always published depending on hardware/sim
                # setup, so fall back to 0.0 if missing.
                if msg.velocity and len(msg.velocity) > idx:
                    self._velocities[i] = msg.velocity[idx]

        self._joint_state_received = True

    def get_joint_state(self):
        """Returns (positions, velocities) as numpy arrays, in self.joint_names order."""
        return self._positions.copy(), self._velocities.copy()

    def get_ee_pose(self):
        """
        Returns (position, quaternion) for the end effector relative to
        base_frame, or (None, None) if the transform isn't available yet
        (e.g. TF hasn't received data, or the frame name is wrong).
        """
        try:
            transform = self._tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time()  # "latest available" rather than a specific timestamp
            )
        except TransformException as ex:
            self.node.get_logger().warn(
                f'Could not get transform {self.base_frame} -> {self.ee_frame}: {ex}',
                throttle_duration_sec=2.0
            )
            return None, None

        t = transform.transform.translation
        q = transform.transform.rotation
        position = np.array([t.x, t.y, t.z], dtype=np.float32)
        quaternion = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        return position, quaternion

    @property
    def is_ready(self) -> bool:
        """True once we've received at least one joint state message."""
        return self._joint_state_received
