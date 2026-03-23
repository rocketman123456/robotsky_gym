import argparse
import os
import torch
import glob


def extract_dimensions_from_state_dict(state_dict, obs_norm_state_dict):
    """Extract dimensions from model state dict by inspecting layer shapes."""
    # Extract actor input dimension (first layer of actor)
    actor_input_dim = None
    actor_output_dim = None
    critic_input_dim = None

    # Find actor layers (Sequential modules use even indices: 0, 2, 4, ...)
    actor_weight_keys = [k for k in state_dict.keys() if "actor" in k and "weight" in k and not "std" in k]
    if actor_weight_keys:
        # Find first layer (actor.0.weight)
        first_layer_key = None
        for key in actor_weight_keys:
            parts = key.split(".")
            if len(parts) >= 2 and parts[1] == "0":
                first_layer_key = key
                break

        if first_layer_key:
            actor_input_dim = state_dict[first_layer_key].shape[1]

        # Find last linear layer (highest even index)
        actor_layer_indices = []
        for key in actor_weight_keys:
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    if idx % 2 == 0:  # Linear layers are at even indices in Sequential
                        actor_layer_indices.append((idx, key))
                except ValueError:
                    pass

        if actor_layer_indices:
            # Get the last layer (highest index)
            last_layer_idx, last_layer_key = max(actor_layer_indices, key=lambda x: x[0])
            actor_output_dim = state_dict[last_layer_key].shape[0]

    # Find critic input dimension from first linear layer
    critic_weight_keys = [k for k in state_dict.keys() if "critic" in k and "weight" in k]
    if critic_weight_keys:
        # Find first layer (critic.0.weight)
        first_critic_key = None
        for key in critic_weight_keys:
            parts = key.split(".")
            if len(parts) >= 2 and parts[1] == "0":
                first_critic_key = key
                break

        if first_critic_key:
            critic_input_dim = state_dict[first_critic_key].shape[1]

    # Extract observation dimension from obs normalizer
    obs_dim = None
    for key in obs_norm_state_dict.keys():
        if "_mean" in key or "_var" in key or "_std" in key:
            param = obs_norm_state_dict[key]
            if len(param.shape) >= 2:
                obs_dim = param.shape[1]
            elif len(param.shape) == 1:
                obs_dim = param.shape[0]
            break

    return actor_input_dim, actor_output_dim, critic_input_dim, obs_dim


def get_robot_preset(robot_type):
    """Get preset dimensions for known robot types."""
    presets = {
        "robotsky_wq": {
            "num_actions": 16,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
        },
    }
    return presets.get(robot_type.lower(), None)


def main():
    parser = argparse.ArgumentParser(description="Save trained model to a specified path with support for different robots.")
    parser.add_argument(
        "--load_run",
        type=str,
        required=True,
        help="Path to the run directory (e.g., 'logs/experiment_1/').",
    )
    parser.add_argument(
        "--step_num",
        type=str,
        required=False,
        default=None,
        help="Step number to load the model from. If not specified, loads the latest model.",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        required=False,
        default=None,
        help="Robot type (k1, k1_leg, t1, t1p). If specified, uses preset dimensions.",
    )
    parser.add_argument(
        "--num_actor_obs",
        type=int,
        required=False,
        default=None,
        help="Number of actor observations. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--num_critic_obs",
        type=int,
        required=False,
        default=None,
        help="Number of critic observations. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--num_actions",
        type=int,
        required=False,
        default=None,
        help="Number of actions. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--actor_hidden_dims",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Actor hidden dimensions (e.g., --actor_hidden_dims 512 256 128).",
    )
    parser.add_argument(
        "--critic_hidden_dims",
        type=int,
        nargs="+",
        required=False,
        default=None,
        help="Critic hidden dimensions (e.g., --critic_hidden_dims 512 256 128).",
    )
    parser.add_argument(
        "--activation",
        type=str,
        required=False,
        default="elu",
        help="Activation function (default: elu).",
    )
    args = parser.parse_args()

    run_folder = args.load_run
    model_files = glob.glob(os.path.join(run_folder, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files found in {run_folder}")

    def extract_step(filename):
        basename = os.path.basename(filename)
        try:
            step_str = basename.split("model_")[1].split(".pt")[0]
            return int(step_str)
        except Exception:
            return -1

    model_files = [(extract_step(f), f) for f in model_files]
    model_files = [item for item in model_files if item[0] != -1]
    if not model_files:
        raise ValueError("No valid model_*.pt files found.")

    if args.step_num is not None:
        step_num = int(args.step_num)
        matching_files = [f for f in model_files if f[0] == step_num]
        if not matching_files:
            raise ValueError(f"No model file found for step {step_num}")
        run_path = matching_files[0][1]
        print(f"Loading model from: {run_path}")
    else:
        max_step_file = max(model_files, key=lambda x: x[0])[1]
        run_path = max_step_file
        print(f"Loading latest model from: {run_path}")

    loaded_dict = torch.load(run_path, weights_only=False, map_location="cpu")

    # Extract dimensions from state dict
    model_state_dict = loaded_dict.get("model_state_dict", {})
    obs_norm_state_dict = loaded_dict.get("obs_norm_state_dict", {})

    actor_input_dim, actor_output_dim, critic_input_dim, obs_dim = extract_dimensions_from_state_dict(model_state_dict, obs_norm_state_dict)

    print(f"[INFO] Auto-detected dimensions:")
    print(f"  Actor input: {actor_input_dim}")
    print(f"  Actor output: {actor_output_dim}")
    print(f"  Critic input: {critic_input_dim}")
    print(f"  Observation dim: {obs_dim}")

    # Override with robot preset if specified
    robot_preset = None
    if args.robot_type:
        robot_preset = get_robot_preset(args.robot_type)
        if robot_preset:
            print(f"[INFO] Using robot preset: {args.robot_type}")
            if args.num_actions is None:
                args.num_actions = robot_preset["num_actions"]
            if args.actor_hidden_dims is None:
                args.actor_hidden_dims = robot_preset["actor_hidden_dims"]
            if args.critic_hidden_dims is None:
                args.critic_hidden_dims = robot_preset["critic_hidden_dims"]
        else:
            print(f"[WARNING] Unknown robot type: {args.robot_type}, ignoring preset")

    # Use provided values or auto-detected values
    num_actor_obs = args.num_actor_obs if args.num_actor_obs is not None else actor_input_dim
    num_critic_obs = args.num_critic_obs if args.num_critic_obs is not None else critic_input_dim
    num_actions = args.num_actions if args.num_actions is not None else actor_output_dim

    # Detect hidden dimensions from state dict if not provided
    if args.actor_hidden_dims is None:
        # Try to infer from state dict
        # Sequential modules: Linear layers at even indices (0, 2, 4, ...)
        actor_weight_keys = [k for k in model_state_dict.keys() if "actor" in k and "weight" in k and not "std" in k]
        actor_layer_indices = []
        for key in actor_weight_keys:
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    if idx % 2 == 0:  # Linear layers are at even indices
                        actor_layer_indices.append((idx, key))
                except ValueError:
                    pass

        if len(actor_layer_indices) >= 2:
            # Sort by index and get all hidden layer dimensions (skip last output layer)
            actor_layer_indices.sort(key=lambda x: x[0])
            args.actor_hidden_dims = []
            for idx, key in actor_layer_indices[:-1]:  # Skip last (output) layer
                dim = model_state_dict[key].shape[0]  # Output dimension of this layer
                args.actor_hidden_dims.append(dim)

            if not args.actor_hidden_dims:
                args.actor_hidden_dims = [512, 256, 128]  # Default fallback
        else:
            args.actor_hidden_dims = [512, 256, 128]  # Default fallback

    if args.critic_hidden_dims is None:
        # Try to infer from state dict
        critic_weight_keys = [k for k in model_state_dict.keys() if "critic" in k and "weight" in k]
        critic_layer_indices = []
        for key in critic_weight_keys:
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    if idx % 2 == 0:  # Linear layers are at even indices
                        critic_layer_indices.append((idx, key))
                except ValueError:
                    pass

        if len(critic_layer_indices) >= 2:
            # Sort by index and get all hidden layer dimensions (skip last output layer)
            critic_layer_indices.sort(key=lambda x: x[0])
            args.critic_hidden_dims = []
            for idx, key in critic_layer_indices[:-1]:  # Skip last (output) layer
                dim = model_state_dict[key].shape[0]  # Output dimension of this layer
                args.critic_hidden_dims.append(dim)

            if not args.critic_hidden_dims:
                args.critic_hidden_dims = [512, 256, 128]  # Default fallback
        else:
            args.critic_hidden_dims = [512, 256, 128]  # Default fallback

    # Validate dimensions
    if num_actor_obs is None:
        raise ValueError("Could not determine num_actor_obs. Please specify --num_actor_obs or ensure model has actor.0.weight")
    if num_critic_obs is None:
        raise ValueError("Could not determine num_critic_obs. Please specify --num_critic_obs or ensure model has critic.0.weight")
    if num_actions is None:
        raise ValueError("Could not determine num_actions. Please specify --num_actions or ensure model has actor output layer")
    if obs_dim is None:
        obs_dim = num_actor_obs  # Fallback to actor obs dim

    print(f"[INFO] Using dimensions:")
    print(f"  Actor input: {num_actor_obs}")
    print(f"  Actor output: {num_actions}")
    print(f"  Critic input: {num_critic_obs}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Actor hidden dims: {args.actor_hidden_dims}")
    print(f"  Critic hidden dims: {args.critic_hidden_dims}")

    # Resolve activation function
    if args.activation.lower() == "elu":
        activation = torch.nn.ELU
    elif args.activation.lower() == "relu":
        activation = torch.nn.ReLU
    elif args.activation.lower() == "tanh":
        activation = torch.nn.Tanh
    elif args.activation.lower() == "mish":
        activation = torch.nn.Mish
    elif args.activation.lower() == "selu":
        activation = torch.nn.SELU
    elif args.activation.lower() == "crelu":
        activation = torch.nn.CELU
    elif args.activation.lower() == "lrelu":
        activation = torch.nn.LeakyReLU
    elif args.activation.lower() == "sigmoid":
        activation = torch.nn.Sigmoid
    elif args.activation.lower() == "identity":
        activation = torch.nn.Identity
    else:
        activation = torch.nn.ELU
        print(f"[WARNING] Unknown activation {args.activation}, using ELU")

    class ActorCritic(torch.nn.Module):
        def __init__(self, num_actor_obs, num_critic_obs, num_actions, actor_hidden_dims, critic_hidden_dims, activation):
            super().__init__()

            # Build actor network
            actor_layers = []
            actor_layers.append(torch.nn.Linear(num_actor_obs, actor_hidden_dims[0]))
            actor_layers.append(activation())
            for i in range(len(actor_hidden_dims)):
                if i == len(actor_hidden_dims) - 1:
                    actor_layers.append(torch.nn.Linear(actor_hidden_dims[i], num_actions))
                else:
                    actor_layers.append(torch.nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
                    actor_layers.append(activation())

            self.actor = torch.nn.Sequential(*actor_layers)

            # Build critic network
            critic_layers = []
            critic_layers.append(torch.nn.Linear(num_critic_obs, critic_hidden_dims[0]))
            critic_layers.append(activation())
            for i in range(len(critic_hidden_dims)):
                if i == len(critic_hidden_dims) - 1:
                    critic_layers.append(torch.nn.Linear(critic_hidden_dims[i], 1))
                else:
                    critic_layers.append(torch.nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
                    critic_layers.append(activation())

            self.critic = torch.nn.Sequential(*critic_layers)
            self.std = torch.nn.Parameter(torch.zeros(num_actions), requires_grad=True)

        def forward(self, x):
            return self.actor(x)

    class ObsNormalizer(torch.nn.Module):
        def __init__(self, obs_dim):
            super().__init__()
            self._mean = torch.nn.Parameter(torch.zeros(1, obs_dim), requires_grad=False)
            self._var = torch.nn.Parameter(torch.ones(1, obs_dim), requires_grad=False)
            self._std = torch.nn.Parameter(torch.ones(1, obs_dim), requires_grad=False)
            self.count = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

        def forward(self, x):
            return (x - self._mean) / (self._std + 1e-8)

    actor_critic = ActorCritic(num_actor_obs, num_critic_obs, num_actions, args.actor_hidden_dims, args.critic_hidden_dims, activation)
    obs_normalizer = ObsNormalizer(obs_dim)

    actor_critic.load_state_dict(loaded_dict["model_state_dict"])

    # Load observation normalizer state
    for name, param in obs_normalizer.named_parameters():
        if name in obs_norm_state_dict:
            param.data.copy_(obs_norm_state_dict[name])

    class PolicyWrapper(torch.nn.Module):
        def __init__(self, actor, obs_normalizer):
            super().__init__()
            self.actor = actor
            self.obs_normalizer = obs_normalizer

        def forward(self, x):
            x_norm = self.obs_normalizer(x)
            return self.actor(x_norm)

    export_model_dir = os.path.join(run_folder, "exported")
    os.makedirs(export_model_dir, exist_ok=True)

    policy_cpu = PolicyWrapper(actor_critic.actor, obs_normalizer).to("cpu")
    policy_cpu.eval()
    script_module = torch.jit.script(policy_cpu)

    robot_suffix = f"_{args.robot_type}" if args.robot_type else ""

    # Save as TorchScript .pt
    export_path = os.path.join(export_model_dir, f"exported_policy{robot_suffix}.pt")
    script_module.save(export_path)
    print(f"[INFO] Exported JIT policy to: {export_path}")
    
    # Save as ONNX
    onnx_path = os.path.join(export_model_dir, f"exported_policy{robot_suffix}.onnx")
    dummy_input = torch.randn(1, num_actor_obs)
    torch.onnx.export(
        policy_cpu,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"}
        }
    )
    print(f"[INFO] Exported ONNX policy to: {onnx_path}")


if __name__ == "__main__":
    main()
