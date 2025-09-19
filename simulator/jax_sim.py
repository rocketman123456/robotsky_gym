import os
import sys
import pathlib
import urllib

if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "glfw"
else:
    os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import jaxsim.mujoco
from jaxsim import logging

logging.set_logging_level(logging.LoggingLevel.WARNING)
print(f"Running on {jax.devices()}")


os.path.abspath("")


# Retrieve the file
# url = "https://raw.githubusercontent.com/ami-iit/jaxsim/refs/heads/main/examples/assets/cartpole.urdf"
# model_path, _ = urllib.request.urlretrieve(url)
# model_urdf_string = pathlib.Path(model_path).read_text()

model_urdf_string = pathlib.Path("./resource/cartpole/cartpole.urdf").read_text()


# Create the model from the model description.
model = jaxsim.api.model.JaxSimModel.build_from_model_description(
    model_description=model_urdf_string,
    time_step=0.010,
)

# Create the data storing the simulation state.
data_zero = jaxsim.api.data.JaxSimModelData.zero(model=model)

# Initialize the simulated time.
T = jnp.arange(start=0, stop=5.0, step=model.time_step)

# Create the MJCF resources from the URDF.
mjcf_string, assets = jaxsim.mujoco.UrdfToMjcf.convert(
    urdf=model.built_from,
    # Create the camera used by the recorder.
    cameras=jaxsim.mujoco.loaders.MujocoCamera.build_from_target_view(
        camera_name="cartpole_camera",
        lookat=jaxsim.api.link.com_position(
            model=model,
            data=data_zero,
            link_index=jaxsim.api.link.name_to_idx(model=model, link_name="cart"),
            in_link_frame=False,
        ),
        distance=3,
        azimuth=150,
        elevation=-10,
    ),
)

# Create a helper to operate on the MuJoCo model and data.
mj_model_helper = jaxsim.mujoco.MujocoModelHelper.build_from_xml(mjcf_description=mjcf_string, assets=assets)

# Create the video recorder.
recorder = jaxsim.mujoco.MujocoVideoRecorder(
    model=mj_model_helper.model,
    data=mj_model_helper.data,
    fps=int(1 / model.time_step),
    width=320 * 2,
    height=240 * 2,
)

import mediapy as media


# Create a random joint position.
# For a random full state, you can use jaxsim.api.data.random_model_data.
random_joint_positions = jax.random.uniform(
    minval=-1.0,
    maxval=1.0,
    shape=(model.dofs(),),
    key=jax.random.PRNGKey(0),
)

# Reset the state to the random joint positions.
data = jaxsim.api.data.JaxSimModelData.build(model=model, joint_positions=random_joint_positions)

for _ in T:

    # Step the JaxSim simulation.
    data = jaxsim.api.model.step(
        model=model,
        data=data,
        joint_force_references=None,
        link_forces=None,
    )

    # Update the MuJoCo data.
    mj_model_helper.set_joint_positions(positions=data.joint_positions, joint_names=model.joint_names())

    # Record a new video frame.
    recorder.record_frame(camera_name="cartpole_camera")


# Play the video.
# media.show_video(recorder.frames, fps=recorder.fps)
media.write_video("records/cartpole_random.mp4", recorder.frames, fps=recorder.fps)
recorder.frames = []

# Define the PD gains
kp = 10.0
kd = 6.0


def computed_torque_controller(
    data: jaxsim.api.data.JaxSimModelData,
    s_des: jax.Array,
    s_dot_des: jax.Array,
) -> jax.Array:

    # Compute the gravity compensation term.
    hs = jaxsim.api.model.free_floating_bias_forces(model=model, data=data)[6:]

    # Compute the joint-related portion of the floating-base mass matrix.
    Mss = jaxsim.api.model.free_floating_mass_matrix(model=model, data=data)[6:, 6:]

    # Get the current joint positions and velocities.
    s = data.joint_positions
    ṡ = data.joint_velocities

    # Compute the actuated joint torques.
    s_star = -kp * (s - s_des) - kd * (ṡ - s_dot_des)
    τ = Mss @ s_star + hs

    return τ


# Initialize the data.

# Set the joint positions.
data = jaxsim.api.data.JaxSimModelData.build(
    model=model, joint_positions=jnp.array([-0.25, jnp.deg2rad(160)]), joint_velocities=jnp.array([3.00, jnp.deg2rad(10) / model.time_step])
)

for _ in T:

    # Get the actuated torques from the computed torque controller.
    τ = computed_torque_controller(
        data=data,
        s_des=jnp.array([0.0, 0.0]),
        s_dot_des=jnp.array([0.0, 0.0]),
    )

    # Step the JaxSim simulation.
    data = jaxsim.api.model.step(
        model=model,
        data=data,
        joint_force_references=τ,
    )

    # Update the MuJoCo data.
    mj_model_helper.set_joint_positions(positions=data.joint_positions, joint_names=model.joint_names())

    # Record a new video frame.
    recorder.record_frame(camera_name="cartpole_camera")

# media.show_video(recorder.frames, fps=recorder.fps)
media.write_video("records/cartpole_control.mp4", recorder.frames, fps=recorder.fps)
recorder.frames = []
