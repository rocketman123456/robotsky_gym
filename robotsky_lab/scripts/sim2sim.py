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


def load_motor_config(config_path: str) -> dict:
    """Load motor configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    motors = {}
    for name, data in config.items():
        # Skip non-motor data fields (like comments)
        if name == "comment" or not isinstance(data, dict):
            continue
        motors[name] = data

    return motors


def compute_max_torque_at_velocity(velocity: float, motor_config: dict) -> float:
    """
    Compute maximum torque at given velocity based on motor envelope.

    Motor envelope:
    - For |v| <= corner_velocity: max_torque available
    - For corner_velocity < |v| <= max_velocity: linear decrease (constant power region)
    - For |v| > max_velocity: 0

    Args:
        velocity: Joint velocity (rad/s)
        motor_config: Motor configuration dict with max_velocity, max_torque, corner_velocity, corner_torque

    Returns:
        Maximum allowed torque (Nm)
    """
    max_velocity = motor_config["max_velocity"]
    max_torque = motor_config["max_torque"]
    corner_velocity = motor_config["corner_velocity"]
    corner_torque = motor_config["corner_torque"]

    abs_vel = abs(velocity)

    # If velocity exceeds max_velocity, torque is zero
    if abs_vel >= max_velocity:
        return 0.0

    # If velocity is within constant torque region
    if abs_vel <= corner_velocity:
        return max_torque

    # Constant power region: linear interpolation from (corner_velocity, corner_torque) to (max_velocity, 0)
    # Slope = -corner_torque / (max_velocity - corner_velocity)
    slope = -corner_torque / (max_velocity - corner_velocity)
    max_torque_at_vel = corner_torque + slope * (abs_vel - corner_velocity)

    return max_torque_at_vel


def clip_torque_by_velocity(torque: np.ndarray, velocity: np.ndarray, actuator_names: list, motor_mapping: dict, motor_configs: dict) -> np.ndarray:
    """
    Clip torque based on joint velocity and motor limits.

    Args:
        torque: Torque array (Nm)
        velocity: Velocity array (rad/s)
        actuator_names: List of actuator names (same length as torque/velocity)
        motor_mapping: Dict mapping actuator name patterns to motor model names
        motor_configs: Dict of motor configurations

    Returns:
        Clipped torque array
    """
    clipped_torque = torque.copy()

    for i in range(len(torque)):
        vel = velocity[i]
        torq = torque[i]
        actuator_name = actuator_names[i] if i < len(actuator_names) else None

        # Find motor for this actuator by matching name patterns
        motor_name = None
        if actuator_name:
            for pattern, motor in motor_mapping.items():
                if pattern == "default":
                    continue
                if pattern in actuator_name:
                    motor_name = motor
                    break

        # Use default motor if no pattern matched
        if not motor_name and "default" in motor_mapping:
            motor_name = motor_mapping["default"]

        if motor_name and motor_name in motor_configs:
            motor_config = motor_configs[motor_name]
            max_torque = compute_max_torque_at_velocity(vel, motor_config)

            # # Check if torque and velocity have the same sign (both positive or both negative)
            # # If they have opposite signs or velocity is zero, motor is braking/stationary
            # # and we don't apply velocity-based clipping
            # same_direction = (vel > 0 and torq > 0) or (vel < 0 and torq < 0)

            # if same_direction:
            #     # Motor is accelerating: apply velocity-based torque limit
            #     max_torque = compute_max_torque_at_velocity(vel, motor_config)
            #     # Clip torque to motor envelope (respecting sign)
            #     if torq > 0:
            #         clipped_torque[i] = np.clip(torq, 0.0, max_torque)
            #     else:
            #         clipped_torque[i] = np.clip(torq, -max_torque, 0.0)
            # else:
            #     # Motor is braking or stationary: only clip to max_torque, ignore velocity limit
            #     # This allows full braking torque even at high velocities
            #     max_torque = motor_config["max_torque"]
            #     clipped_torque[i] = np.clip(torq, -max_torque, max_torque)

            # Clip torque to motor envelope
            clipped_torque[i] = np.clip(torq, -max_torque, max_torque)
        # If no motor found, don't clip (use original torque)

    return clipped_torque


def get_robot_preset(robot_type):
    """Get preset configuration for different robot types."""
    presets = {
        "k1": {
            "model_path": "legged_lab/assets/booster_k1/K1_serial.xml",
            "num_action": 20,
            "num_obs_per_step": 69,  # 70 # 69
            "actor_obs_history_length": 10,
            "init_pos": [0.0, 0.0, 0.6],
            "init_rot": [1.0, 0.0, 0.0, 0.0],  # w, x, y, z
            "default_joint_angles": {
                "Shoulder_Pitch": 0.2,
                "Left_Shoulder_Roll": -1.25,
                "Right_Shoulder_Roll": 1.25,
                "Left_Elbow_Yaw": -0.5,
                "Right_Elbow_Yaw": 0.5,
                "Hip_Pitch": -0.15,
                "Knee_Pitch": 0.3,
                "Ankle_Pitch": -0.15,
                "default": 0.0,
            },
            "stiffness": {
                "Shoulder_Pitch": 28.4,
                "Shoulder_Roll": 28.4,
                "Elbow_Pitch": 19.7,
                "Elbow_Yaw": 12.6,
                "_Hip_Pitch": 30.3,
                "_Hip_Roll": 21.5,
                "_Hip_Yaw": 32.0,
                "_Knee_": 50.5,
                "_Ankle_Pitch": 20.3,
                "_Ankle_Roll": 6.5,
            },
            "damping": {
                "Shoulder_Pitch": 1.5,
                "Shoulder_Roll": 1.5,
                "Elbow_Pitch": 1.25,
                "Elbow_Yaw": 1.0,
                "_Hip_Pitch": 2.4,
                "_Hip_Roll": 1.7,
                "_Hip_Yaw": 2.3,
                "_Knee_": 4.0,
                "_Ankle_Pitch": 0.9,
                "_Ankle_Roll": 0.3,
            },
            "friction": {
                "Shoulder_Pitch": 1e-4,
                "Shoulder_Roll": 1e-4,
                "Elbow_Pitch": 1e-4,
                "Elbow_Yaw": 1e-4,
                "_Hip_Pitch": 1e-4,
                "_Hip_Roll": 1e-4,
                "_Hip_Yaw": 1e-4,
                "_Knee_": 1e-4,
                "_Ankle_Pitch": 1e-4,
                "_Ankle_Roll": 1e-4,
            },
            "motor_mapping": {
                "Shoulder_Pitch": "4310",
                "Shoulder_Roll": "4310",
                "Elbow_Pitch": "4310",
                "Elbow_Yaw": "4310",
                "_Hip_Pitch": "6408",
                "_Hip_Roll": "4310",
                "_Hip_Yaw": "4315",
                "_Knee_": "6416",
                "_Ankle_Pitch": "4310",
                "_Ankle_Roll": "4310",
                "default": "4310",
            },
            # fmt:off
            "mujoco_to_isaac_idx": [
                0, 4, 8, 14, 1, 5, 9, 15, 2, 6, 10, 16, 3, 7, 11, 17, 12, 18, 13, 19,
            ],
            "isaac_to_mujoco_idx": [
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 16, 18, 3, 7, 11, 15, 17, 19,
            ],
            # fmt:on
        },
        "k1_leg": {
            "model_path": "legged_lab/assets/booster_k1/K1_serial_leg.xml",
            "num_action": 12,  # 6 joints per leg * 2 legs
            "num_obs_per_step": 45,  # 3 (ang_vel) + 3 (projected_gravity) + 3 (command) + 12 (joint_pos) + 12 (joint_vel) + 12 (action)
            "actor_obs_history_length": 10,
            "init_pos": [0.0, 0.0, 0.6],
            "init_rot": [1.0, 0.0, 0.0, 0.0],
            "default_joint_angles": {
                "Hip_Pitch": -0.15,
                "Knee_Pitch": 0.3,
                "Ankle_Pitch": -0.15,
                "default": 0.0,
            },
            "stiffness": {
                "_Hip_": 100.0,
                "_Knee_": 100.0,
                "_Ankle_": 50.0,
            },
            "damping": {
                "_Hip_": 2.0,
                "_Knee_": 2.0,
                "_Ankle_": 1.0,
            },
            "friction": {
                "_Hip_": 0.2,
                "_Knee_": 0.2,
                "_Ankle_": 0.1,
            },
            "motor_mapping": {
                "_Hip_Pitch": "6408",
                "_Hip_Yaw": "6408",
                "_Knee_": "6416",
                "_Ankle_": "4310",
                "default": "4310",
            },
            "mujoco_to_isaac_idx": [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11],  # Placeholder, adjust based on actual mapping
            "isaac_to_mujoco_idx": [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],  # Placeholder, adjust based on actual mapping
        },
        "t1": {
            "model_path": "legged_lab/assets/booster_t1/T1_serial_collision.xml",
            "num_action": 21,
            "num_obs_per_step": 72,
            "actor_obs_history_length": 10,
            "init_pos": [0.0, 0.0, 0.7],
            "init_rot": [1.0, 0.0, 0.0, 0.0],
            "default_joint_angles": {
                "Shoulder_Pitch": 0.2,
                "Left_Shoulder_Roll": -1.3,
                "Right_Shoulder_Roll": 1.3,
                "Left_Elbow_Yaw": -0.5,
                "Right_Elbow_Yaw": 0.5,
                "Hip_Pitch": -0.2,
                "Knee_Pitch": 0.4,
                "Ankle_Pitch": -0.2,
                "Waist": 0.0,
                "default": 0.0,
            },
            "stiffness": {
                "_Shoulder_": 50.0,
                "_Elbow_": 50.0,
                "Waist": 200.0,
                "_Hip_": 200.0,
                "_Knee_": 200.0,
                "_Ankle_": 50.0,
            },
            "damping": {
                "_Shoulder_": 1.0,
                "_Elbow_": 1.0,
                "Waist": 5.0,
                "_Hip_": 5.0,
                "_Knee_": 5.0,
                "_Ankle_": 1.0,
            },
            "friction": {
                "_Shoulder_": 0.1,
                "_Elbow_": 0.1,
                "Waist": 0.2,
                "_Hip_": 0.2,
                "_Knee_": 0.2,
                "_Ankle_": 0.1,
            },
            "motor_mapping": {
                "_Shoulder_": "4310",
                "_Elbow_": "4310",
                "Waist": "6416",
                "_Hip_": "6416",
                "_Knee_": "6408",
                "_Ankle_": "4315",
                "default": "4310",
            },
            # fmt:off
            "mujoco_to_isaac_idx": [
                0,  4,  8,  1, 5,  9, 15,  2, 6, 10, 16,  3, 7, 11, 17, 12, 18, 13, 19, 14, 20
            ],
            "isaac_to_mujoco_idx": [
                0,  3,  7, 11, 1,  4,  8, 12, 2, 5,  9, 13, 15, 17, 19, 6, 10, 14, 16, 18, 20
            ],
            # fmt:on
        },
        "t1p": {
            "model_path": "legged_lab/assets/booster_t1p/T1P_serial_25_v2.xml",
            "num_action": 21,  # Adjust if different
            "num_obs_per_step": 72,
            "actor_obs_history_length": 10,
            "init_pos": [0.0, 0.0, 0.7],
            "init_rot": [1.0, 0.0, 0.0, 0.0],
            "default_joint_angles": {
                "Shoulder_Pitch": 0.2,
                "Left_Shoulder_Roll": -1.3,
                "Right_Shoulder_Roll": 1.3,
                "Left_Elbow_Yaw": -0.5,
                "Right_Elbow_Yaw": 0.5,
                "Hip_Pitch": -0.2,
                "Knee_Pitch": 0.4,
                "Ankle_Pitch": -0.2,
                "default": 0.0,
            },
            "stiffness": {
                "_Shoulder_": 50.0,
                "_Elbow_": 50.0,
                "_Hip_": 200.0,
                "_Knee_": 200.0,
                "_Ankle_": 50.0,
            },
            "damping": {
                "_Shoulder_": 1.0,
                "_Elbow_": 1.0,
                "_Hip_": 5.0,
                "_Knee_": 5.0,
                "_Ankle_": 1.0,
            },
            "friction": {
                "_Shoulder_": 0.1,
                "_Elbow_": 0.1,
                "_Hip_": 0.2,
                "_Knee_": 0.2,
                "_Ankle_": 0.1,
            },
            "mujoco_to_isaac_idx": list(range(21)),  # Placeholder, adjust based on actual mapping
            "isaac_to_mujoco_idx": list(range(21)),  # Placeholder, adjust based on actual mapping
        },
        "robotsky_wq": {
            "model_path": "robotsky_lab/assets/robotsky_wq/mjcf/robotsky_wq.xml",
            "num_action": 16,
            # 3 (ang_vel) + 3 (projected_gravity) + 3 (command) + 16 (joint_pos) + 16 (joint_vel) + 16 (action)
            "num_obs_per_step": 57,
            "actor_obs_history_length": 10,
            "init_pos": [0.0, 0.0, 0.5],
            "init_rot": [1.0, 0.0, 0.0, 0.0],
            "default_joint_angles": {
                "RF_Roll_Joint": 0.1,
                "RF_Hip_Joint": -0.5,
                "RF_Knee_Joint": 1.0,
                "RF_Wheel_Joint": 0.0,
                "LF_Roll_Joint": -0.1,
                "LF_Hip_Joint": -0.5,
                "LF_Knee_Joint": 1.0,
                "LF_Wheel_Joint": 0.0,
                "RB_Roll_Joint": 0.1,
                "RB_Hip_Joint": 0.5,
                "RB_Knee_Joint": -1.0,
                "RB_Wheel_Joint": 0.0,
                "LB_Roll_Joint": -0.1,
                "LB_Hip_Joint": 0.5,
                "LB_Knee_Joint": -1.0,
                "LB_Wheel_Joint": 0.0,
                "default": 0.0,
            },
            "stiffness": {
                "Roll_Joint": 20.0,
                "Hip_Joint": 20.0,
                "Knee_Joint": 40.0,
                # wheel velocity-controlled in Isaac; in MuJoCo keep it low-stiffness
                "Wheel_Joint": 0.0,
            },
            "damping": {
                "Roll_Joint": 1.0,
                "Hip_Joint": 1.0,
                "Knee_Joint": 1.0,
                "Wheel_Joint": 1.0,
            },
            "friction": {
                "Roll_Joint": 1e-4,
                "Hip_Joint": 1e-4,
                "Knee_Joint": 1e-4,
                "Wheel_Joint": 1e-4,
            },
            "motor_mapping": {
                # If you have a motor config JSON, map by joint name substring.
                # Wheels usually have different envelope; adjust as needed.
                "Roll_Joint": "4310",
                "Hip_Joint": "6408",
                "Knee_Joint": "6416",
                "Wheel_Joint": "4310",
                "default": "4310",
            },
            # From terminal output:
            # ISAAC to URDF indices: [3, 7, 11, 15, 1, 5, 9, 13, 2, 6, 10, 14, 0, 4, 8, 12]
            # URDF to ISAAC indices: [12, 4, 8, 0, 13, 5, 9, 1, 14, 6, 10, 2, 15, 7, 11, 3]
            "isaac_to_mujoco_idx": [3, 7, 11, 15, 1, 5, 9, 13, 2, 6, 10, 14, 0, 4, 8, 12],
            "mujoco_to_isaac_idx": [12, 4, 8, 0, 13, 5, 9, 1, 14, 6, 10, 2, 15, 7, 11, 3],
        },
    }
    return presets.get(robot_type.lower(), None)


def main():
    parser = argparse.ArgumentParser(description="Sim2Sim simulation with support for different robots.")
    parser.add_argument(
        "--robot_type",
        type=str,
        required=True,
        choices=["k1", "k1_leg", "t1", "t1p", "robotsky_wq"],
        help="Robot type (k1, k1_leg, t1, t1p, robotsky_wq)",
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
    parser.add_argument(
        "--motor_config",
        type=str,
        required=False,
        default=None,
        help="Path to motor configuration JSON file. If not specified, will try to use default path.",
    )
    parser.add_argument(
        "--enable_motor_clipping",
        action="store_true",
        help="Enable motor torque clipping based on velocity limits.",
    )
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
    action_scale = args.action_scale if args.action_scale is not None else 0.25

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
    motor_configs = {}
    actuator_names = []
    motor_config_path = None
    if args.enable_motor_clipping:
        # Determine motor config path
        if args.motor_config:
            motor_config_path = args.motor_config
        else:
            # Try default path relative to script location
            script_dir = Path(__file__).parent
            default_motor_config = script_dir.parent / "assets" / "motor" / "motor_config.json"
            if default_motor_config.exists():
                motor_config_path = str(default_motor_config)
            else:
                # Try alternative path from workspace root
                alt_motor_config = Path("legged_lab/assets/motor/motor_config.json")
                if alt_motor_config.exists():
                    motor_config_path = str(alt_motor_config)
                else:
                    print("[WARNING] Motor config file not found. Motor clipping will be disabled.")
                    print(f"[WARNING] Tried: {default_motor_config} and {alt_motor_config}")
                    args.enable_motor_clipping = False

        if args.enable_motor_clipping and motor_config_path:
            try:
                motor_configs = load_motor_config(motor_config_path)
                print(f"[INFO] Loaded {len(motor_configs)} motor configurations from: {motor_config_path}")

                # Get actuator names from MuJoCo model
                for i in range(mj_model.nu):
                    actuator_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    actuator_names.append(actuator_name if actuator_name else f"actuator_{i}")

                print(f"[INFO] Motor clipping enabled for {len(actuator_names)} actuators")
            except Exception as e:
                print(f"[WARNING] Failed to load motor config: {e}. Motor clipping will be disabled.")
                args.enable_motor_clipping = False

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

    # Get joint index mappings
    mujoco_to_isaac_idx = robot_preset["mujoco_to_isaac_idx"]
    isaac_to_mujoco_idx = robot_preset["isaac_to_mujoco_idx"]

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
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
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
                    obs[0:3] = base_ang_vel
                    obs[3:6] = projected_gravity
                    obs[6] = lin_vel_x
                    obs[7] = lin_vel_y
                    obs[8] = ang_vel_yaw

                    # Map joint positions and velocities
                    # Structure: obs[9:9+num_actions] = joint_pos, obs[9+num_actions:9+2*num_actions] = joint_vel, obs[9+2*num_actions:9+3*num_actions] = actions
                    if len(mujoco_to_isaac_idx) >= num_action:
                        obs[9 : 9 + num_action] = (dof_pos - default_dof_pos)[mujoco_to_isaac_idx[:num_action]] * 1.0
                        obs[9 + num_action : 9 + 2 * num_action] = dof_vel[mujoco_to_isaac_idx[:num_action]] * 0.1
                        obs[9 + 2 * num_action : 9 + 3 * num_action] = actions * 1.0
                    else:
                        # Fallback when mapping is shorter than num_action
                        obs[9 : 9 + len(mujoco_to_isaac_idx)] = (dof_pos - default_dof_pos[mujoco_to_isaac_idx]) * 1.0
                        obs[9 + len(mujoco_to_isaac_idx) : 9 + 2 * len(mujoco_to_isaac_idx)] = dof_vel[mujoco_to_isaac_idx] * 0.1
                        obs[9 + 2 * len(mujoco_to_isaac_idx) : 9 + 2 * len(mujoco_to_isaac_idx) + num_action] = actions[:num_action] * 1.0

                    # cmd_is_zero = abs(lin_vel_x) + abs(lin_vel_y) + abs(ang_vel_yaw) < 0.02
                    # obs[9 + 3 * num_action] = cmd_is_zero

                    # Update observation history
                    obs_history = np.roll(obs_history, shift=-num_obs_per_step)
                    obs_history[-num_obs_per_step:] = obs.copy()
                    obs_history = np.clip(obs_history, -100.0, 100.0)

                    # Get action from policy
                    actions = policy(torch.tensor(obs_history, dtype=torch.float32)).detach().numpy().squeeze(0)
                    actions[:] = np.clip(actions, -100.0, 100.0)

                # Smooth actions
                smoothed_actions = smoothed_actions * args.smooth_factor + actions * (1.0 - args.smooth_factor)

                # Compute target positions
                # isaac_to_mujoco_idx maps from Isaac action indices to MuJoCo actuator indices
                dof_targets[:] = default_dof_pos
                # Apply actions to all actuators using the mapping (isaac_to_mujoco_idx should have length num_action)
                if len(isaac_to_mujoco_idx) == num_action:
                    dof_targets[:] += action_scale * smoothed_actions[isaac_to_mujoco_idx]
                elif len(isaac_to_mujoco_idx) > num_action:
                    # Use first num_action elements if mapping is longer
                    dof_targets[:] += action_scale * smoothed_actions[isaac_to_mujoco_idx[:num_action]]
                else:
                    # If mapping is shorter, only apply to mapped indices
                    for i, mujoco_idx in enumerate(isaac_to_mujoco_idx):
                        if i < num_action:
                            dof_targets[mujoco_idx] += action_scale * smoothed_actions[i]

                # Apply PD control
                # Note: Friction can be added with: -dof_friction * sign(dof_vel) * abs(dof_vel)
                ctrl_torque = dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel

                # Apply motor torque clipping based on velocity if enabled
                if args.enable_motor_clipping and "motor_mapping" in robot_preset:
                    ctrl_torque = clip_torque_by_velocity(
                        ctrl_torque,
                        dof_vel,
                        actuator_names,
                        robot_preset["motor_mapping"],
                        motor_configs,
                    )

                mj_data.ctrl = np.clip(
                    ctrl_torque,
                    mj_model.actuator_ctrlrange[:, 0],
                    mj_model.actuator_ctrlrange[:, 1],
                )

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
