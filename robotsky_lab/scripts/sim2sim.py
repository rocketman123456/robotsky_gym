import os
import sys
import glob
import time
import select
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import mujoco.viewer

try:
    import glfw

    GLFW_AVAILABLE = True
except ImportError:
    GLFW_AVAILABLE = False
    print("[WARNING] glfw module not available. Keyboard pause functionality may be limited.")


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def quat_to_rpy(quat):
    x, y, z, w = quat
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w /= norm
    x /= norm
    y /= norm
    z /= norm
    roll = np.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.asin(2 * (w * y - x * z))
    yaw = np.atan2(2 * (w * z + x * y), 1 - 2 * (z**2 + y**2))
    return roll, pitch, yaw


# def load_motor_config(config_path: str) -> dict:
#     """Load motor configuration from JSON file."""
#     with open(config_path, "r", encoding="utf-8") as f:
#         config = json.load(f)

#     motors = {}
#     for name, data in config.items():
#         # Skip non-motor data fields (like comments)
#         if name == "comment" or not isinstance(data, dict):
#             continue
#         motors[name] = data

#     return motors


# def compute_max_torque_at_velocity(velocity: float, motor_config: dict) -> float:
#     """
#     Compute maximum torque at given velocity based on motor envelope.

#     Motor envelope:
#     - For |v| <= corner_velocity: max_torque available
#     - For corner_velocity < |v| <= max_velocity: linear decrease (constant power region)
#     - For |v| > max_velocity: 0

#     Args:
#         velocity: Joint velocity (rad/s)
#         motor_config: Motor configuration dict with max_velocity, max_torque, corner_velocity, corner_torque

#     Returns:
#         Maximum allowed torque (Nm)
#     """
#     max_velocity = motor_config["max_velocity"]
#     max_torque = motor_config["max_torque"]
#     corner_velocity = motor_config["corner_velocity"]
#     corner_torque = motor_config["corner_torque"]

#     abs_vel = abs(velocity)

#     # If velocity exceeds max_velocity, torque is zero
#     if abs_vel >= max_velocity:
#         return 0.0

#     # If velocity is within constant torque region
#     if abs_vel <= corner_velocity:
#         return max_torque

#     # Constant power region: linear interpolation from (corner_velocity, corner_torque) to (max_velocity, 0)
#     # Slope = -corner_torque / (max_velocity - corner_velocity)
#     slope = -corner_torque / (max_velocity - corner_velocity)
#     max_torque_at_vel = corner_torque + slope * (abs_vel - corner_velocity)

#     return max_torque_at_vel


# def clip_torque_by_velocity(torque: np.ndarray, velocity: np.ndarray, actuator_names: list, motor_mapping: dict, motor_configs: dict) -> np.ndarray:
#     """
#     Clip torque based on joint velocity and motor limits.

#     Args:
#         torque: Torque array (Nm)
#         velocity: Velocity array (rad/s)
#         actuator_names: List of actuator names (same length as torque/velocity)
#         motor_mapping: Dict mapping actuator name patterns to motor model names
#         motor_configs: Dict of motor configurations

#     Returns:
#         Clipped torque array
#     """
#     clipped_torque = torque.copy()

#     for i in range(len(torque)):
#         vel = velocity[i]
#         torq = torque[i]
#         actuator_name = actuator_names[i] if i < len(actuator_names) else None

#         # Find motor for this actuator by matching name patterns
#         motor_name = None
#         if actuator_name:
#             for pattern, motor in motor_mapping.items():
#                 if pattern == "default":
#                     continue
#                 if pattern in actuator_name:
#                     motor_name = motor
#                     break

#         # Use default motor if no pattern matched
#         if not motor_name and "default" in motor_mapping:
#             motor_name = motor_mapping["default"]

#         if motor_name and motor_name in motor_configs:
#             motor_config = motor_configs[motor_name]
#             max_torque = compute_max_torque_at_velocity(vel, motor_config)

#             # # Check if torque and velocity have the same sign (both positive or both negative)
#             # # If they have opposite signs or velocity is zero, motor is braking/stationary
#             # # and we don't apply velocity-based clipping
#             # same_direction = (vel > 0 and torq > 0) or (vel < 0 and torq < 0)

#             # if same_direction:
#             #     # Motor is accelerating: apply velocity-based torque limit
#             #     max_torque = compute_max_torque_at_velocity(vel, motor_config)
#             #     # Clip torque to motor envelope (respecting sign)
#             #     if torq > 0:
#             #         clipped_torque[i] = np.clip(torq, 0.0, max_torque)
#             #     else:
#             #         clipped_torque[i] = np.clip(torq, -max_torque, 0.0)
#             # else:
#             #     # Motor is braking or stationary: only clip to max_torque, ignore velocity limit
#             #     # This allows full braking torque even at high velocities
#             #     max_torque = motor_config["max_torque"]
#             #     clipped_torque[i] = np.clip(torq, -max_torque, max_torque)

#             # Clip torque to motor envelope
#             clipped_torque[i] = np.clip(torq, -max_torque, max_torque)
#         # If no motor found, don't clip (use original torque)

#     return clipped_torque


def get_robot_preset(robot_type):
    """Get preset configuration for different robot types.

    robotsky_wq values align with RobotSkyWQFlatEnvCfg / WheelLeggedRobotCfg / RobotSkyWQEnv
    (robotsky_wq_config.py, robotsky_wq_env.py) for sim2sim transfer.
    """
    presets = {
        "robotsky_wq": {
            "model_path": "robotsky_lab/assets/robotsky_wq/mjcf/robotsky_wq.xml",
            "num_action": 16,
            # 3 (ang_vel) + 3 (projected_gravity) + 3 (command) + 16 (joint_pos) + 16 (joint_vel) + 16 (action)
            "num_obs_per_step": 57,
            "actor_obs_history_length": 10,
            # ROBOTSKY_WQ_CFG.init_state.pos
            "init_pos": [0.0, 0.0, 0.5],
            "init_rot": [1.0, 0.0, 0.0, 0.0],
            # Indices in *policy / Isaac* observation & action layout (RF,LF,RB,LB × Roll,Hip,Knee,Wheel)
            "leg_index": [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14],
            "wheel_index": [3, 7, 11, 15],
            # NormalizationCfg.obs_scales (RobotSkyWQFlatEnvCfg)
            "obs_scale_ang_vel": 0.5,  # 1.0 # 0.5,
            "obs_scale_projected_gravity": 1.0,
            "obs_scale_commands": 1.0,
            "obs_scale_joint_pos": 1.0,
            "obs_scale_joint_vel_leg": 0.05,  # 0.05,
            "obs_scale_joint_vel_wheel": 0.1,
            "obs_scale_actions": 1.0,
            "clip_observations": 100.0,
            "clip_actions": 100.0,
            # WheelLeggedRobotCfg
            "action_scale": 0.25,
            "wheel_action_scale": 4.0,
            "default_joint_angles": {
                "RF_Roll": 0.1,
                "RF_Hip": -0.5,
                "RF_Knee": 1.0,
                "RF_Wheel": 0.0,
                "LF_Roll": -0.1,
                "LF_Hip": -0.5,
                "LF_Knee": 1.0,
                "LF_Wheel": 0.0,
                "RB_Roll": 0.1,
                "RB_Hip": 0.5,
                "RB_Knee": -1.0,
                "RB_Wheel": 0.0,
                "LB_Roll": -0.1,
                "LB_Hip": 0.5,
                "LB_Knee": -1.0,
                "LB_Wheel": 0.0,
                "default": 0.0,
            },
            "stiffness": {
                "Roll": 20.0,
                "Hip": 20.0,
                "Knee": 40.0,
                # wheel velocity-controlled in Isaac; in MuJoCo keep it low-stiffness
                "Wheel": 0.0,
            },
            "damping": {
                "Roll": 1.0,
                "Hip": 1.0,
                "Knee": 1.0,
                "Wheel": 2.0,
            },
            "friction": {
                "Roll": 1e-4,
                "Hip": 1e-4,
                "Knee": 1e-4,
                "Wheel": 1e-4,
            },
            # "motor_mapping": {
            #     # If you have a motor config JSON, map by joint name substring.
            #     # Wheels usually have different envelope; adjust as needed.
            #     "Roll_Joint": "4310",
            #     "Hip_Joint": "6408",
            #     "Knee_Joint": "6416",
            #     "Wheel_Joint": "4310",
            #     "default": "4310",
            # },
            # Same convention as RobotSkyWQEnv (isaac2urdf_idx / urdf2isaac_idx):
            # - isaac_to_mujoco_idx[i] = MuJoCo actuator index for policy dimension i (gather obs in Isaac order)
            # - mujoco_to_isaac_idx[u] = policy index that drives MuJoCo actuator u (scatter actions)
            "mujoco_to_isaac_idx": [3, 7, 11, 15, 1, 5, 9, 13, 2, 6, 10, 14, 0, 4, 8, 12],
            "isaac_to_mujoco_idx": [12, 4, 8, 0, 13, 5, 9, 1, 14, 6, 10, 2, 15, 7, 11, 3],
        },
    }
    return presets.get(robot_type.lower(), None)


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim simulation with support for different robots.")
    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        choices=["robotsky_wq"],
        help="Robot type (robotsky_wq)",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        required=False,
        default=None,
        help="Path to the exported policy JIT model. If not specified, will try to auto-detect from robot type.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="Path to the MuJoCo XML model file. Overrides robot preset.",
    )
    parser.add_argument(
        "--num_action",
        type=int,
        required=False,
        default=None,
        help="Number of actions. Overrides robot preset.",
    )
    parser.add_argument(
        "--num_obs_per_step",
        type=int,
        required=False,
        default=None,
        help="Number of observations per step. Overrides robot preset.",
    )
    parser.add_argument(
        "--sim_duration",
        type=float,
        required=False,
        default=100.0,
        help="Simulation duration in seconds (default: 100.0)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        required=False,
        default=0.005,
        help="Simulation timestep (default: 0.005)",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        required=False,
        default=4,
        help="Control decimation factor (default: 4)",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        required=False,
        default=None,
        help="Action scaling factor. Overrides robot preset.",
    )
    parser.add_argument(
        "--wheel_action_scale",
        type=float,
        required=False,
        default=None,
        help="Action scaling factor. Overrides robot preset.",
    )
    parser.add_argument(
        "--smooth_factor",
        type=float,
        required=False,
        default=0.0,
        help="Action smoothing factor (default: 0.0)",
    )
    parser.add_argument(
        "--csv_output",
        type=str,
        required=False,
        default=None,
        help="Path to output CSV file for recording robot state. If not specified, no CSV will be written.",
    )
    # parser.add_argument(
    #     "--motor_config",
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="Path to motor configuration JSON file. If not specified, will try to use default path.",
    # )
    # parser.add_argument(
    #     "--enable_motor_clipping",
    #     action="store_true",
    #     help="Enable motor torque clipping based on velocity limits.",
    # )
    args = parser.parse_args()

    # Get robot preset
    robot_preset = get_robot_preset(args.robot_type)
    if robot_preset is None:
        raise ValueError(f"Unknown robot type: {args.robot_type}")

    # Use provided values or preset values
    policy_path = args.policy_path
    if policy_path is None:
        # Try to auto-detect policy path
        # Common pattern: logs/{robot_type}_walk/*/exported/exported_policy_{robot_type}.pt
        possible_paths = glob.glob(f"logs/*{args.robot_type}*/**/exported/exported_policy*.pt", recursive=True)
        if possible_paths:
            # Get the most recent one
            policy_path = max(possible_paths, key=os.path.getmtime)
            print(f"[INFO] Auto-detected policy path: {policy_path}")
        else:
            raise ValueError(
                f"Could not auto-detect policy path. Please specify --policy_path or ensure model exists at logs/*{args.robot_type}*/**/exported/exported_policy*.pt"
            )

    model_path = args.model_path if args.model_path is not None else robot_preset["model_path"]
    num_action = args.num_action if args.num_action is not None else robot_preset["num_action"]
    num_obs_per_step = args.num_obs_per_step if args.num_obs_per_step is not None else robot_preset["num_obs_per_step"]
    action_scale = args.action_scale if args.action_scale is not None else robot_preset["action_scale"]
    wheel_action_scale = args.wheel_action_scale if args.wheel_action_scale is not None else robot_preset["wheel_action_scale"]
    leg_index = np.array(robot_preset["leg_index"], dtype=np.int32)
    wheel_index = np.array(robot_preset["wheel_index"], dtype=np.int32)
    obs_s = {
        "ang_vel": robot_preset["obs_scale_ang_vel"],
        "projected_gravity": robot_preset["obs_scale_projected_gravity"],
        "commands": robot_preset["obs_scale_commands"],
        "joint_pos": robot_preset["obs_scale_joint_pos"],
        "joint_vel_leg": robot_preset["obs_scale_joint_vel_leg"],
        "joint_vel_wheel": robot_preset["obs_scale_joint_vel_wheel"],
        "actions": robot_preset["obs_scale_actions"],
    }
    clip_obs = robot_preset["clip_observations"]
    clip_act = robot_preset["clip_actions"]

    print(f"[INFO] Robot type: {args.robot_type}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Policy path: {policy_path}")
    print(f"[INFO] Number of actions: {num_action}")
    print(f"[INFO] Number of observations per step: {num_obs_per_step}")
    print(f"[INFO] Action scale: {action_scale}")

    # Load policy
    policy = torch.jit.load(policy_path, map_location="cpu")

    # Load MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.timestep = args.dt
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)

    # Load motor configuration if motor clipping is enabled
    # motor_configs = {}
    # actuator_names = []
    # motor_config_path = None
    # if args.enable_motor_clipping:
    #     # Determine motor config path
    #     if args.motor_config:
    #         motor_config_path = args.motor_config
    #     else:
    #         # Try default path relative to script location
    #         script_dir = Path(__file__).parent
    #         default_motor_config = script_dir.parent / "assets" / "motor" / "motor_config.json"
    #         if default_motor_config.exists():
    #             motor_config_path = str(default_motor_config)
    #         else:
    #             # Try alternative path from workspace root
    #             alt_motor_config = Path("legged_lab/assets/motor/motor_config.json")
    #             if alt_motor_config.exists():
    #                 motor_config_path = str(alt_motor_config)
    #             else:
    #                 print("[WARNING] Motor config file not found. Motor clipping will be disabled.")
    #                 print(f"[WARNING] Tried: {default_motor_config} and {alt_motor_config}")
    #                 args.enable_motor_clipping = False

    #     if args.enable_motor_clipping and motor_config_path:
    #         try:
    #             motor_configs = load_motor_config(motor_config_path)
    #             print(f"[INFO] Loaded {len(motor_configs)} motor configurations from: {motor_config_path}")

    #             # Get actuator names from MuJoCo model
    #             for i in range(mj_model.nu):
    #                 actuator_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    #                 actuator_names.append(actuator_name if actuator_name else f"actuator_{i}")

    #             print(f"[INFO] Motor clipping enabled for {len(actuator_names)} actuators")
    #         except Exception as e:
    #             print(f"[WARNING] Failed to load motor config: {e}. Motor clipping will be disabled.")
    #             args.enable_motor_clipping = False

    # Setup default joint positions, stiffness, damping, and friction
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    dof_friction = np.zeros(mj_model.nu, dtype=np.float32)

    for i in range(mj_model.nu):
        # Set default joint positions
        found = False
        actuator_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for name in robot_preset["default_joint_angles"].keys():
            if name in actuator_name:
                default_dof_pos[i] = robot_preset["default_joint_angles"][name]
                found = True
                break
        if not found:
            default_dof_pos[i] = robot_preset["default_joint_angles"]["default"]

        # Set stiffness, damping, and friction
        found = False
        for name in robot_preset["stiffness"].keys():
            if name in actuator_name:
                dof_stiffness[i] = robot_preset["stiffness"][name]
                dof_damping[i] = robot_preset["damping"][name]
                dof_friction[i] = robot_preset["friction"][name]
                found = True
                break
        if not found:
            raise ValueError(f"PD gain of joint {actuator_name} were not defined")

    print(f"Default dof pos: {default_dof_pos}")
    print(f"Dof stiffness: {dof_stiffness}")
    print(f"Dof damping: {dof_damping}")
    print(f"Dof friction: {dof_friction}")

    # Set initial state
    mj_data.qpos = np.concatenate(
        [
            np.array(robot_preset["init_pos"], dtype=np.float32),
            np.array(robot_preset["init_rot"], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)

    # Initialize buffers
    actions = np.zeros(num_action, dtype=np.float32)
    smoothed_actions = np.zeros(num_action, dtype=np.float32)
    obs_history = np.zeros((num_obs_per_step * robot_preset["actor_obs_history_length"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    lin_vel_x = 0.0
    lin_vel_y = 0.0
    ang_vel_yaw = 0.0
    it = 0
    paused = False
    space_pressed_last = False  # Track space key state to detect press event

    # Isaac <-> MuJoCo permutation (see preset comments; matches env print isaac2urdf / urdf2isaac)
    mujoco_to_isaac_idx = np.array(robot_preset["mujoco_to_isaac_idx"], dtype=np.int32)
    isaac_to_mujoco_idx = np.array(robot_preset["isaac_to_mujoco_idx"], dtype=np.int32)

    # Setup CSV recording if output path is provided
    csv_writer = None
    csv_file = None
    if args.csv_output:
        csv_file = open(args.csv_output, "w", newline="")
        # Create headers matching simulator_out.csv format
        num_joints = mj_model.nu
        headers = []
        # IMU acceleration (3)
        headers.extend([f"simulator_out/imu_acc/{i+1}" for i in range(3)])
        # IMU Euler angles (3)
        headers.extend([f"simulator_out/imu_eul/{i+1}" for i in range(3)])
        # IMU rotational velocity (3)
        headers.extend([f"simulator_out/imu_rotvel/{i+1}" for i in range(3)])
        # Joint feedback positions (num_joints)
        headers.extend([f"simulator_out/joint_fb_pos/{i+1}" for i in range(num_joints)])
        # Joint feedback torques (num_joints)
        headers.extend([f"simulator_out/joint_fb_tor/{i+1}" for i in range(num_joints)])
        # Joint feedback velocities (num_joints)
        headers.extend([f"simulator_out/joint_fb_vel/{i+1}" for i in range(num_joints)])
        # Motor error codes (num_joints)
        headers.extend([f"simulator_out/motor_err_code/{i+1}" for i in range(num_joints)])
        # Motor temperatures (num_joints)
        headers.extend([f"simulator_out/motor_temp/{i+1}" for i in range(num_joints)])
        # Time (1)
        headers.append("simulator_out/t")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)
        print(f"[INFO] Recording robot state to CSV: {args.csv_output}")

    # Simulation loop
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print("[INFO] Press SPACE to pause/resume simulation")
        print(f"[DEBUG] GLFW_AVAILABLE: {GLFW_AVAILABLE}")
        print(f"[DEBUG] viewer type: {type(viewer)}")
        print(f"[DEBUG] has _handle: {hasattr(viewer, '_handle')}")
        if hasattr(viewer, "_handle"):
            print(f"[DEBUG] viewer._handle: {viewer._handle}")

        print("Set command (x, y, yaw): ", end="")
        while viewer.is_running():
            time_start = time.time()

            # Check for space bar press (keyboard callback)
            if GLFW_AVAILABLE and hasattr(viewer, "_handle") and viewer._handle:  # noqa
                try:
                    space_pressed = glfw.get_key(viewer._handle._window, glfw.KEY_SPACE) == glfw.PRESS  # noqa
                    # Detect press event (transition from not pressed to pressed)
                    if space_pressed and not space_pressed_last:
                        paused = not paused
                        status = "PAUSED" if paused else "RESUMED"
                        print(f"\n[{status}]")
                        print("Set command (x, y, yaw): ", end="")
                    space_pressed_last = space_pressed
                except Exception as e:
                    print(f"[DEBUG] Exception in pause check: {e}")

            # Handle command input
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        print(f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}")
                        print("Set command (x, y, yaw): ", end="")
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")

            # Skip simulation step if paused
            if not paused:
                # Get current state
                dof_pos = mj_data.qpos.astype(np.float32)[7:]
                dof_vel = mj_data.qvel.astype(np.float32)[6:]
                base_quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                # base_lin_vel = mj_data.sensor("linear-velocity").data.astype(np.float32)
                base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
                projected_gravity = quat_rotate_inverse(base_quat, np.array([0.0, 0.0, -1.0]))

                # Compute observations and actions at decimation rate
                if it % args.decimation == 0:
                    obs = np.zeros(num_obs_per_step, dtype=np.float32)
                    # RobotSkyWQEnv.compute_current_observations + NormalizationCfg.obs_scales
                    obs[0:3] = base_ang_vel * obs_s["ang_vel"]
                    obs[3:6] = projected_gravity * obs_s["projected_gravity"]
                    cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_yaw], dtype=np.float32) * obs_s["commands"]
                    obs[6:9] = cmd

                    # Isaac-order joints: gather with isaac_to_mujoco_idx[i] = MuJoCo index for policy dim i
                    # Structure: obs[9:9+N] joint_pos, [9+N:9+2N] joint_vel, [9+2N:9+3N] last action
                    i2m = isaac_to_mujoco_idx[:num_action]
                    joint_pos = (dof_pos - default_dof_pos)[i2m] * obs_s["joint_pos"]
                    joint_pos[wheel_index] = 0.0
                    obs[9 : 9 + num_action] = joint_pos
                    joint_vel_raw = dof_vel[i2m]
                    joint_vel = np.zeros(num_action, dtype=np.float32)
                    joint_vel[leg_index] = joint_vel_raw[leg_index] * obs_s["joint_vel_leg"]
                    joint_vel[wheel_index] = joint_vel_raw[wheel_index] * obs_s["joint_vel_wheel"]
                    obs[9 + num_action : 9 + 2 * num_action] = joint_vel
                    obs[9 + 2 * num_action : 9 + 3 * num_action] = actions * obs_s["actions"]
                    # cmd_is_zero = abs(lin_vel_x) + abs(lin_vel_y) + abs(ang_vel_yaw) < 0.02
                    # obs[9 + 3 * num_action] = cmd_is_zero

                    # Update observation history
                    obs_history = np.roll(obs_history, shift=-num_obs_per_step)
                    obs_history[-num_obs_per_step:] = obs.copy()
                    obs_history = np.clip(obs_history, -clip_obs, clip_obs)

                    # Get action from policy
                    actions = policy(torch.tensor(obs_history, dtype=torch.float32)).detach().numpy().squeeze(0)
                    actions[:] = np.clip(actions, -clip_act, clip_act)

                # Smooth actions
                smoothed_actions = smoothed_actions * args.smooth_factor + actions * (1.0 - args.smooth_factor)

                # MuJoCo dof_targets: mujoco_to_isaac_idx[j] = policy index for MuJoCo actuator j (RobotSkyWQEnv step)
                dof_targets[:] = default_dof_pos
                policy_on_mujoco = smoothed_actions[mujoco_to_isaac_idx[:num_action]]
                dof_targets[:] = default_dof_pos + action_scale * policy_on_mujoco
                dof_targets[wheel_index] = policy_on_mujoco[wheel_index] * wheel_action_scale

                # Apply PD control
                # Note: Friction can be added with: -dof_friction * sign(dof_vel) * abs(dof_vel)
                ctrl_torque = dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel
                ctrl_torque[wheel_index] = dof_damping[wheel_index] * (dof_targets[wheel_index] - dof_vel[wheel_index])

                # Apply motor torque clipping based on velocity if enabled
                # if args.enable_motor_clipping and "motor_mapping" in robot_preset:
                #     ctrl_torque = clip_torque_by_velocity(
                #         ctrl_torque,
                #         dof_vel,
                #         actuator_names,
                #         robot_preset["motor_mapping"],
                #         motor_configs,
                #     )

                # dof_targets[:] = default_dof_pos
                mj_data.ctrl = dof_targets
                # mj_data.ctrl = np.clip(
                #     ctrl_torque,
                #     mj_model.actuator_ctrlrange[:, 0],
                #     mj_model.actuator_ctrlrange[:, 1],
                # )

                # Step simulation
                mujoco.mj_step(mj_model, mj_data)

                # Record robot state to CSV if enabled
                if csv_writer is not None:
                    # Get current state after step
                    current_dof_pos = mj_data.qpos.astype(np.float32)[7:]
                    current_dof_vel = mj_data.qvel.astype(np.float32)[6:]
                    current_base_quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
                    current_base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
                    current_projected_gravity = quat_rotate_inverse(current_base_quat, np.array([0.0, 0.0, -1.0]))

                    # IMU acceleration: use projected gravity * 9.81 (gravity in m/s^2)
                    # In body frame, gravity appears as negative acceleration
                    # Note: This is the gravity component. For full IMU acceleration including linear acceleration,
                    # you would need to compute the derivative of linear velocity, but for simplicity we use gravity only.
                    imu_acc = -current_projected_gravity * 9.81

                    # IMU Euler angles: convert quaternion to roll, pitch, yaw
                    roll, pitch, yaw = quat_to_rpy(current_base_quat)
                    imu_eul = np.array([roll, pitch, yaw])

                    # IMU rotational velocity
                    imu_rotvel = current_base_ang_vel

                    # Joint torques: use actuator force if available, otherwise use qfrc_actuator
                    if hasattr(mj_data, "actuator_force") and mj_data.actuator_force is not None:
                        joint_torques = mj_data.actuator_force.astype(np.float32)
                    elif hasattr(mj_data, "qfrc_actuator") and mj_data.qfrc_actuator is not None:
                        joint_torques = mj_data.qfrc_actuator.astype(np.float32)
                    else:
                        # Fallback: use control signal as approximation
                        joint_torques = mj_data.ctrl.astype(np.float32)

                    # Motor error codes: zeros (no errors in simulation)
                    motor_err_codes = np.zeros(mj_model.nu, dtype=np.float32)

                    # Motor temperatures: placeholder values (not simulated)
                    motor_temps = np.zeros(mj_model.nu, dtype=np.float32)

                    # Time
                    current_time = it * args.dt

                    # Write row to CSV
                    row = (
                        list(imu_acc)
                        + list(imu_eul)
                        + list(imu_rotvel)
                        + list(current_dof_pos)
                        + list(joint_torques)
                        + list(current_dof_vel)
                        + list(motor_err_codes)
                        + list(motor_temps)
                        + [current_time]
                    )
                    csv_writer.writerow(row)

                it += 1

            # Always update viewer even when paused
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()

            # Sleep to maintain real-time
            time_end = time.time()
            time.sleep(max(0, args.dt - (time_end - time_start)))

    # Close CSV file if opened
    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] CSV recording completed. Saved to: {args.csv_output}")


if __name__ == "__main__":
    main()
