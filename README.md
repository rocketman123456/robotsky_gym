# robotsky_wq_gym
training code for my fast build wheel-legged robot

```bash
python robotsky_lab/scripts/train.py \
    --logger tensorboard \
    --task robotsky_wq_flat \
    --headless \
    --num_envs 4096 

python robotsky_lab/scripts/train.py \
    --logger tensorboard \
    --task robotsky_wq_rough \
    --headless \
    --num_envs 4096 \
    --seed 42 \
    --max_iterations 40000

# Play
python robotsky_lab/scripts/play.py \
    --task robotsky_wq_flat \
    --num_envs 100 \
    --load_run=2026-03-24_18-18-37 \
    --checkpoint=model_18000.pt

# python robotsky_lab/scripts/play.py \
#     --task robotsky_wq_rough \
#     --num_envs 10

# PYGLFW_LIBRARY_VARIANT=x11 python robotsky_lab/scripts/sim2sim.py \
#     --robot_type robotsky_wq \
#     --smooth_factor 0.0 \
#     --dt 0.005 \
#     --decimation 4 \
#     --policy_path logs/robotsky_wq_flat/2026-03-23_11-33-21/exported/exported_policy_robotsky_wq.pt

python robotsky_lab/scripts/save_model.py \
    --load_run logs/robotsky_wq_flat/2026-04-05_10-46-38 \
    --step_num 30000 \
    --robot_type robotsky_wq \
    --actor_hidden_dims 512 256 128 \
    --critic_hidden_dims 512 256 128 \
    --num_actor_obs 570 --num_critic_obs 640 --num_actions 16

python robotsky_lab/scripts/sim2sim.py \
    --robot_type robotsky_wq \
    --smooth_factor 0.0 \
    --action_scale 0.25 \
    --wheel_action_scale 2.5 \
    --dt 0.005 \
    --decimation 4 \
    --policy_path logs/robotsky_wq_flat/2026-04-05_10-46-38/exported/exported_policy_robotsky_wq.pt


```

```bash
python robotsky_lab/scripts/train.py --task <task_name> [选项]

python robotsky_lab/scripts/train.py --task h1_flat --headless
python robotsky_lab/scripts/train.py --task h1_rough --headless

python robotsky_lab/scripts/train.py --task g1_flat --headless
python robotsky_lab/scripts/train.py --task g1_rough --headless

python robotsky_lab/scripts/train.py --task gr2_flat --headless
python robotsky_lab/scripts/train.py --task gr2_rough --headless

# 步行训练
python robotsky_lab/scripts/train.py --task walk --headless

# 跑步训练
python robotsky_lab/scripts/train.py --task run --headless

# 带深度摄像头的步行训练（自动启用摄像头渲染）
python robotsky_lab/scripts/train.py --task walk_with_sensor --headless

# 带深度摄像头的跑步训练
python robotsky_lab/scripts/train.py --task run_with_sensor --headless

# 指定并行环境数量（默认 4096）
python robotsky_lab/scripts/train.py --task walk --headless --num_envs 2048

# 限制训练迭代次数
python robotsky_lab/scripts/train.py --task walk --headless --max_iterations 10000

# 指定随机种子
python robotsky_lab/scripts/train.py --task walk --headless --seed 42

# 自定义实验名称和运行标签
python robotsky_lab/scripts/train.py --task walk --headless \
    --experiment_name tienkung_walk \
    --run_name baseline_v1

# 选择日志后端（默认 wandb）
python robotsky_lab/scripts/train.py --task walk --headless \
    --logger tensorboard

# 从断点继续训练
python robotsky_lab/scripts/train.py --task walk --headless \
    --resume True \
    --load_run "2025-01-01_12-00-00_baseline_v1" \
    --checkpoint "model_10000.pt"

# 多 GPU 分布式训练
python -m torch.distributed.run \
    --nproc_per_node=4 \
    robotsky_lab/scripts/train.py \
    --task walk \
    --headless \
    --distributed

```