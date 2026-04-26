# FOCOPS: First-Order Constrained Optimization in Policy Space

This repository contains a PyTorch implementation of FOCOPS for constrained reinforcement learning, with training/evaluation scripts and logged results for multiple seeds across MuJoCo and Safety-Gymnasium tasks.

## Repository Contents

Core files:

- `main.py`: training entry point (runs multiple seeds in parallel).
- `agent.py`: FOCOPS update logic for policy, reward critic, cost critic, and dual variable `nu`.
- `buffer.py`: rollout buffer with GAE for reward and cost.
- `networks.py`: actor/critic neural networks.
- `plot.py`: aggregates per-seed CSV logs and plots learning curves.
- `visualize.py`: loads saved policies and records rollout videos.
- `env.yml`: Conda environment specification.

Environment result folders (already present):

- `HalfCheetah-v4/`
- `Walker2d-v5/`
- `SafetyCarGoal1-v0/`
- `SafetyCarPush1-v0/`

Each environment folder contains subfolders like `seed_42/`, with files such as:

- `training_log_seed_<seed>.csv`
- `policy_seed_<seed>_end.pth`
- optionally `policy_seed_<seed>_mid.pth`

## Requirements

- Linux (current workspace is Linux)
- Conda or Mamba

## Setup

Create and activate the environment:

```bash
conda env create -f env.yml
conda activate focops_env
```

## Training

Run training for one environment:

```bash
python main.py --env Walker2d-v5
```

Supported environment names are defined in `ENV_CONFIGS` inside `main.py`:

- `Walker2d-v5`
- `HalfCheetah-v4`
- `SafetyCarGoal1-v0`
- `SafetyCarPush1-v0`

Important behavior:

- seeds are hardcoded to `[42, 101, 777, 97, 88]` and run in parallel with `multiprocessing.Pool`.
- logs are written to `<env>/seed_<seed>/training_log_seed_<seed>.csv`.
- policy checkpoints are saved at end of training, and mid-training checkpoint is saved for the last seed.

## Plot Learning Curves

`plot.py` reads all per-seed CSV logs for an environment and draws:

- average returns,
- average discounted cost,
- dual variable `nu`.

Example:

```bash
python plot.py --env Walker2d-v5 --threshold 81.89
```

The figure is saved to:

```text
<env>/focops_<env>_learning_curves.png
```

Suggested thresholds from `main.py`:

- `Walker2d-v5`: `81.89`
- `HalfCheetah-v4`: `151.99`
- `SafetyCarGoal1-v0`: `7.0`
- `SafetyCarPush1-v0`: `5.0`

## Record Policy Videos

`visualize.py` searches recursively for policy checkpoints (`*.pth`) in an environment folder, loads matching normalization stats (`obs_rms_*.pkl`), and exports rollout videos.

Example:

```bash
python visualize.py --env HalfCheetah-v4
```

Generated files are written next to each policy:

- `simulation_seed_<seed>_end.mp4`
- `simulation_seed_<seed>_mid.mp4` (if corresponding checkpoint exists)

## Log Format

Training CSV columns:

- `Seed`
- `Epoch`
- `Dual_Variable_nu`
- `Average_Discounted_Cost`
- `Average_Returns`
