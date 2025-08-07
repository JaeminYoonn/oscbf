"""Testing the performance of OSCBF under many different safety constraints, namely:

1. End-effector set containment
2. Joint limit avoidance
3. Singularity avoidance
4. Collision avoidance
5. Whole-body set containment

While there are many other safety constraints that we could also account for, this
should give a good view of the controller's performance under common situations
encountered in practice.
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from cbfpy import CBF

from oscbf.core.manipulator import Manipulator, load_panda, load_fr3
from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.core.controllers import PoseTaskTorqueController

from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as sciR
import rclpy
import time

exit_mode = False


@jax.tree_util.register_static
class CombinedConfig(OSCBFTorqueConfig):

    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE Set Containment
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]

        # Whole-body Set Containment
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        return jnp.concatenate(
            [
                h_ee_safe_set,
                h_joint_limits,
                h_singularity,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    # Set nullspace desired joint position
    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )

    # Compute nominal control
    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3],
        des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15],
        des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3),
        des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal,
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    # Apply the CBF safety filter
    return cbf.safety_filter(z, u_nom)


class ROS2Node(Node):
    def __init__(self):
        super().__init__("oscbf")

        self.dt = 0.01

        self.des_pose_subscriber = self.create_subscription(
            PoseStamped,
            "/des_pose",
            self.sub_des_pose,
            10,
        )

        self.robot = load_fr3()
        ee_pos_min = np.array([0.15, -0.25, 0.25])
        ee_pos_max = np.array([0.75, 0.25, 0.75])
        wb_pos_min = np.array([-0.5, -0.4, 0.0])
        wb_pos_max = np.array([0.75, 0.4, 1.0])
        config = CombinedConfig(
            self.robot,
            ee_pos_min,
            ee_pos_max,
            wb_pos_min,
            wb_pos_max,
        )
        self.cbf = CBF.from_config(config)
        self.env = FrankaTorqueControlEnv(
            config.pos_min,
            config.pos_max,
            wb_xyz_min=wb_pos_min,
            wb_xyz_max=wb_pos_max,
            load_floor=False,
            bg_color=(1, 1, 1),
            real_time=True,
        )

        self.env.client.resetDebugVisualizerCamera(
            cameraDistance=2,
            cameraPitch=-27.80,
            cameraYaw=36.80,
            cameraTargetPosition=(0.08, 0.49, -0.04),
        )

        kp_pos = 50.0
        kp_rot = 20.0
        kd_pos = 20.0
        kd_rot = 10.0
        kp_joint = 10.0
        kd_joint = 5.0
        self.osc_controller = PoseTaskTorqueController(
            n_joints=self.robot.num_joints,
            kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
            kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
            kp_joint=kp_joint,
            kd_joint=kd_joint,
            # Note: torque limits will be enforced via the QP. We'll set them to None here
            # because we don't want to clip the values before the QP
            tau_min=None,
            tau_max=None,
        )

        self.des_pose = None
        self.timer = self.create_timer(self.dt, self.loop_callback)

        import functools

        self.compute_control_jit = jax.jit(
            functools.partial(
                compute_control, self.robot, self.osc_controller, self.cbf
            )
        )

    def sub_des_pose(self, msg: PoseStamped):
        position = msg.pose.position
        orientation = msg.pose.orientation
        quat = np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float32,
        )
        mat_R = sciR.from_quat(quat).as_matrix()

        self.des_pose = np.eye(4)
        self.des_pose[0, 3] = position.x
        self.des_pose[1, 3] = position.y
        self.des_pose[2, 3] = position.z
        self.des_pose[:3, :3] = mat_R

    def get_desired_ee_state(self):
        pos = self.des_pose[:3, 3]
        vel = np.zeros(3)
        rot = self.des_pose[:3, :3].ravel()
        omega = np.zeros(3)
        return np.array([*pos, *rot, *vel, *omega])

    def loop_callback(self):
        if self.des_pose is not None:
            joint_state = self.env.get_joint_state()
            ee_state_des = self.get_desired_ee_state()
            t1 = time.time()
            tau = self.compute_control_jit(joint_state, ee_state_des)
            # print(time.time() - t1)
            self.env.apply_control(tau)
            self.env.step()


if __name__ == "__main__":
    rclpy.init()
    node = ROS2Node()
    try:
        while rclpy.ok() and not exit_mode:
            rclpy.spin_once(node, timeout_sec=0.1)
    except:
        pass
        # traceback.print_exc()
    finally:
        node.destroy_node()
        # rclpy.shutdown()
