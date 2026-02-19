# Training Summary

## Overview

Training uses Brax PPO on a MuJoCo simulation. The policy maps observations to actions. Default env is `joystick`; task `flat_terrain_backlash` trains on flat terrain with backlash. Optional resume via `--restore_checkpoint_path`.

Backlash: mechanical gear play in joints. The model inserts a small backlash joint between each motor and link (±0.5°, ~0.0087 rad) to mimic dead zone in cheap servos. The policy observes actual link angle (actuator + backlash), not just commanded position.

## Commands

Sampled each episode from `sample_command()` in `joystick.py`. Ranges (from `default_config()`):

| Signal        | Range             | Unit   |
|---------------|-------------------|--------|
| lin_vel_x     | [-0.15, 0.15]     | m/s    |
| lin_vel_y     | [-0.2, 0.2]       | m/s    |
| ang_vel_yaw   | [-1.0, 1.0]       | rad/s  |
| neck_pitch    | [-0.34, 1.1]      | rad    |
| head_pitch    | [-0.78, 0.78]     | rad    |
| head_yaw      | [-1.5, 1.5]       | rad    |
| head_roll     | [-0.5, 0.5]       | rad    |

10% chance of all zeros (stand still). Commands are resampled every 500 steps.

## Episodes

Episode = 1000 policy steps = 10,000 sim steps = 20 s (episode_length=1000, ctrl_dt=0.02, sim_dt=0.002).

| Phase | Policy steps | Sim steps | Time  | Command   |
|-------|---------------|-----------|-------|-----------|
| 1     | 0–500         | 0–5000    | 0–10 s| Command A |
| 2     | 501–1000      | 5010–10000| 10–20 s| Command B |
| End   | —             | —         | 20 s  | Reset     |

Per policy step: 1 policy step = 1 action = 10 sim steps (n_substeps = ctrl_dt/sim_dt). Policy at 50 Hz, sim at 500 Hz.

Command resampling: When step > 500, a new command is sampled (random, same distribution as reset). No env reset; robot state is unchanged.

Two commands per episode: The policy is trained to react to mid-episode command changes, not only at episode start. This improves robustness when the user changes the joystick during deployment.

## Observations

Policy observes `state` (no privileged info). Built in `_get_obs()`:

| Component              | Dims | Notes                                      |
|------------------------|------|--------------------------------------------|
| gyro                   | 3    | Noisy IMU angular velocity                 |
| accelerometer          | 3    | Noisy IMU linear acceleration              |
| command                | 3    | lin_vel_x, lin_vel_y, ang_vel_yaw          |
| joint_angles           | 10   | Noisy, relative to default pose            |
| joint_velocities       | 10   | Noisy, scaled by dof_vel_scale             |
| last_act               | 10   | Previous action                            |
| last_last_act          | 10   | Two steps ago                              |
| last_last_last_act     | 10   | Three steps ago                            |
| motor_targets          | 10   | Last commanded motor positions (rad)       |
| contact                | 2    | Left/right foot contact (binary)           |
| imitation_phase        | 2    | [cos(2π·i/N), sin(2π·i/N)] gait phase      |

IMU and joint values use `noise_config`; contact and imitation_phase are not noised.

## Reward

Per-step reward in `joystick.py` (`_get_reward`). Each term scaled, summed, multiplied by dt (0.02), clipped to [0, 10000]:

| Term              | Scale   | Formula                                                                 |
|-------------------|---------|-------------------------------------------------------------------------|
| tracking_lin_vel  | 2.5     | exp(-error/0.01), error = (cmd_x - vel_x)² + clip(|vel_y - cmd_y| - 0.1, 0)² |
| tracking_ang_vel  | 6.0     | exp(-(cmd_yaw - ang_vel_z)² / 0.01)                                    |
| torques           | -1e-3   | -sum(torque²)                                                          |
| action_rate       | -0.5    | -sum((act - last_act)²)                                                 |
| alive             | 20.0    | 1.0                                                                    |
| imitation         | 1.0     | Match reference motion (lin_vel, ang_vel, joint_pos/vel, contact) when cmd ≠ 0 |
| stand_still       | -0.2    | -(pose_cost + vel_cost) when cmd_norm < 0.01 (stay near default when standing) |

## Joint observations: Python vs CPP (observation_formatter.cpp)

| Aspect              | Python (joystick)                        | CPP (observation_formatter.cpp)          |
|---------------------|------------------------------------------|------------------------------------------|
| joint_angles        | `noisy_joint_angles - _default_actuator` | `joint_positions - default_joint_positions_` |
| joint_velocities    | `noisy_joint_vel * 0.05`                 | `joint_velocities * 0.05`                 |
| default pose source | `keyframe("home").ctrl`                  | `default_joint_positions` param (matches home keyframe) |
| noise               | Yes (noise_config)                       | No                                       |

Same logic: relative to default, velocity scale 0.05. CPP default comes from `open_duck_mini_controllers.yaml` and matches the home keyframe ctrl. CPP does not add noise; policy was trained with noise for robustness.

## motor_targets: Python vs CPP

| Aspect       | Python (joystick)                          | CPP (motion_controller, observation_formatter)     |
|--------------|--------------------------------------------|----------------------------------------------------|
| Definition   | `default_actuator + action * action_scale`| `default_joint_positions + model_outputs * action_scale` |
| Units        | rad (MuJoCo ctrl)                         | rad (joint position commands)                      |
| Timing       | Observed after step (command just applied) | Observed before inference (command from prev cycle)|
| Rate limiting| Yes (USE_MOTOR_SPEED_LIMITS)               | Yes (apply_rate_limiting)                          |

Same formula and units. Both observe last commanded joint positions in rad.

## Contact: Python vs CPP

### Python (training)

- Uses `geoms_colliding()` from mujoco_playground.
- Geom-level: `left_foot_bottom_tpu`, `right_foot_bottom_tpu` vs `floor`.
- MJX (JAX) data, no debouncing.

### CPP (duck_mini_mujoco_system_interface.cpp)

- Uses `mjData->contact[]` with body IDs.
- Body-level: `foot_assembly`, `foot_assembly_2` vs `floor`.
- Optional debouncing: GAIT mode uses `debounce_on_steps=2`, `debounce_off_steps=2`.
- `debounce_on_steps=0` / `debounce_off_steps=0` are clamped to 1.

### Gaps

1. Debouncing: Training uses raw contact. CPP GAIT mode debounces. Use `contact_consumer=collision` for raw contact to match training.
2. Granularity: Geom vs body. Equivalent for this robot (one contact geom per foot).
3. Output: CPP exposes `contact_raw` and `contact`. Use `contact` with `contact_consumer=collision` for training-equivalent behavior.

## Next steps

1. Re-evaluate gyro_deadband for interfence: CPP applies a deadband (default 0.15 rad/s) to gyro before feeding the policy. Training does not use a deadband; it adds noise. Using gyro_deadband at inference changes the gyro input relative to training and can introduce a sim-to-sim gap. Next steps: (1) Run inference with gyro_deadband=0 and compare to deadband=0.15; (2) If needed, add gyro deadband to training and retrain.

2. Use contact_raw for inference: Training uses raw contact (no debouncing). Set contact_consumer=collision in the CPP contact sensor config so contact matches contact_raw and aligns with training.

3. Training, Observation augmentation – Randomly drop or perturb contact and gyro during training to make the policy robust to timing and noise. Changes: (a) In `default_config()`, add `augment_config` with `gyro_deadband` (rad/s), `gyro_deadband_prob`, `contact_flip_prob`, `contact_dropout_prob`. (b) In `_get_obs()`, after `noisy_gyro`: with `gyro_deadband_prob`, zero elements where |val| < `gyro_deadband`. (c) Before `contact` in state hstack: with `contact_flip_prob` flip 0↔1 per foot; with `contact_dropout_prob` set to 0. Use `jax.random.bernoulli` for per-step sampling.

