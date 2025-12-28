# Copyright (C) 2025 Julia Jia
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS2 replay environment for fine-tuning on collected ROS2 data."""

from typing import Any, Dict, Optional, Union
import h5py
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

from . import constants
from . import base as open_duck_mini_v2_base
from playground.common.poly_reference_motion import PolyReferenceMotion
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
    reward_alive,
)
from playground.open_duck_mini_v2.custom_rewards import reward_imitation

USE_IMITATION_REWARD = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,
        noise_config=config_dict.create(
            level=0.0,  # No noise for replay
            action_min_delay=0,
            action_max_delay=0,
            imu_min_delay=0,
            imu_max_delay=0,
            scales=config_dict.create(
                hip_pos=0.0,
                knee_pos=0.0,
                ankle_pos=0.0,
                joint_vel=0.0,
                gravity=0.0,
                linvel=0.0,
                gyro=0.0,
                accelerometer=0.0,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=2.5,
                tracking_ang_vel=6.0,
                torques=-1.0e-3,
                action_rate=-0.5,
                stand_still=-0.2,
                alive=20.0,
                imitation=1.0,
            ),
            tracking_sigma=0.01,
        ),
        push_config=config_dict.create(
            enable=False,  # No push for replay
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        lin_vel_x=[-0.15, 0.15],
        lin_vel_y=[-0.2, 0.2],
        ang_vel_yaw=[-1.0, 1.0],
        neck_pitch_range=[-0.34, 1.1],
        head_pitch_range=[-0.78, 0.78],
        head_yaw_range=[-1.5, 1.5],
        head_roll_range=[-0.5, 0.5],
        head_range_factor=1.0,
        data_path="mujoco_manual_data.h5",
    )


class ROS2Replay(open_duck_mini_v2_base.OpenDuckMiniV2Env):
    """Replay environment that loads ROS2 collected data."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        """Initialize replay-specific data."""
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        self._default_actuator = self._mj_model.keyframe("home").ctrl

        if USE_IMITATION_REWARD:
            self.PRM = PolyReferenceMotion(
                "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
            )

        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        self._weights = jp.array(
            [
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,
                1.0,
                1.0,
                0.01,
                0.01,
                1.0,
            ]
        )

        self._njoints = self._mj_model.njnt
        self._actuators = self._mj_model.nu

        # Extract leg joints only (first 5 left leg, last 5 right leg, skip 4 head joints in middle)
        # This matches the observation format which has 10 joint positions (legs only)
        self._default_actuator_legs_only = jp.concatenate([
            self._default_actuator[:5],   # Left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
            self._default_actuator[9:14],  # Right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
        ])

        self._torso_body_id = self._mj_model.body(constants.ROOT_BODY).id
        self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._site_id = self._mj_model.site("imu").id

        self._feet_site_id = np.array(
            [self._mj_model.site(name).id for name in constants.FEET_SITES]
        )
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array(
            [self._mj_model.geom(name).id for name in constants.FEET_GEOMS]
        )

        foot_linvel_sensor_adr = []
        for site in constants.FEET_SITES:
            sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # Load ROS2 collected data
        data_path = self._config.get("data_path", "mujoco_manual_data.h5")
        print(f"Loading ROS2 data from: {data_path}")
        with h5py.File(data_path, "r") as f:
            self._observations = jp.array(f["observations"][:])
            self._actions = jp.array(f["actions"][:])
            if "velocity_commands" in f:
                self._velocity_commands = jp.array(f["velocity_commands"][:])
            else:
                # Extract velocity commands from observations (elements 6-9 are 7D command, use first 3)
                self._velocity_commands = self._observations[:, 6:9]
            # Load base linear velocities if available (from velocimeter sensor)
            if "base_linear_velocities" in f:
                self._base_linear_velocities = jp.array(f["base_linear_velocities"][:])
                print(f"Loaded {len(self._base_linear_velocities)} base linear velocity measurements")
            else:
                # Fallback: use zeros if not available (backward compatibility)
                self._base_linear_velocities = jp.zeros((len(self._observations), 3))
                print("Warning: base_linear_velocities not found in data file. Using zeros.")
            self._data_length = len(self._observations)
        print(f"Loaded {self._data_length} data points")

    @property
    def observation_size(self) -> Dict[str, tuple]:
        """Override observation_size to match ROS2 data format (101 elements)."""
        # The checkpoint expects 101-element observations
        return {
            "state": (101,),
            "privileged_state": (212,),  # Match checkpoint training size
        }

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """Reset environment to a random point in the dataset."""
        rng, idx_rng = jax.random.split(rng)
        # Start at a random point in the dataset
        episode_start_idx = jax.random.randint(
            idx_rng, (), 0, max(1, self._data_length - self._config.episode_length)
        )
        current_idx = episode_start_idx

        # Initialize with default qpos/qvel
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)
        ctrl = self.get_actuator_joints_qpos(qpos)
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

        # Get velocity command from data
        cmd = self._velocity_commands[current_idx]

        if USE_IMITATION_REWARD:
            current_reference_motion = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], 0
            )
        else:
            current_reference_motion = jp.zeros(0)

        info = {
            "rng": rng,
            "step": 0,
            "command": cmd,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(2),
            "last_contact": jp.zeros(2, dtype=bool),
            "swing_peak": jp.zeros(2),
            "push": jp.array([0.0, 0.0]),
            "push_step": 0,
            "push_interval_steps": 0,
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
            "imu_history": jp.zeros(self._config.noise_config.imu_max_delay * 3),
            "imitation_i": 0,
            "current_reference_motion": current_reference_motion,
            "imitation_phase": jp.zeros(2),
            "data_idx": current_idx,
            "episode_start_idx": episode_start_idx,
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                if v > 0:
                    metrics[f"reward/{k}"] = jp.zeros(())
                else:
                    metrics[f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        # Extract observation from data
        obs_raw = self._observations[current_idx]
        obs = self._parse_observation(obs_raw, info)
        reward, done = jp.zeros(2)  # Both float32, matching joystick/standing pattern
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        """Step environment using next observation from dataset."""
        # Update imitation phase
        if USE_IMITATION_REWARD:
            state.info["imitation_i"] += 1
            state.info["imitation_i"] = (
                state.info["imitation_i"] % self.PRM.nb_steps_in_period
            )
            state.info["imitation_phase"] = jp.array(
                [
                    jp.cos(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                    jp.sin(
                        (state.info["imitation_i"] / self.PRM.nb_steps_in_period)
                        * 2
                        * jp.pi
                    ),
                ]
            )
        else:
            state.info["imitation_i"] = 0

        # Get next data point (update in state.info, not instance variable)
        current_idx = (state.info["data_idx"] + 1) % self._data_length
        state.info["data_idx"] = current_idx

        # Update command from data
        cmd = self._velocity_commands[current_idx]
        state.info["command"] = cmd

        if USE_IMITATION_REWARD:
            state.info["current_reference_motion"] = self.PRM.get_reference_motion(
                cmd[0], cmd[1], cmd[2], state.info["imitation_i"]
            )
        else:
            state.info["current_reference_motion"] = jp.zeros(0)

        # Update action history
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action

        # Get next observation from dataset
        obs_raw = self._observations[state.info["data_idx"]]
        obs = self._parse_observation(obs_raw, state.info)

        # Compute reward based on observation
        # Extract components from observation for reward computation
        done_bool = self._get_termination(obs_raw, state.info)
        rewards = self._get_reward_from_obs(obs_raw, action, state.info, done_bool)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
        
        # Update metrics
        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                if rew_scale > 0:
                    state.metrics[f"reward/{k}"] = v
                else:
                    state.metrics[f"cost/{k}"] = -v

        state.info["step"] += 1
        
        # Match joystick/standing pattern: convert done to reward dtype
        done = done_bool.astype(reward.dtype)
        state = state.replace(data=state.data, obs=obs, reward=reward, done=done)
        
        return state

    def _parse_observation(self, obs_raw: jax.Array, info: dict) -> Dict[str, jax.Array]:
        """Parse raw observation into state and privileged_state format.
        
        ROS2 observation format (101 elements):
        [gyro(3), accelerometer(3), command(7), joint_pos(10), joint_vel(10),
         last_act(10), last_last_act(10), last_last_last_act(10),
         motor_targets(10), contact(2), imitation_phase(2)]
        
        The checkpoint was trained with 101-element observations, so we use
        the full ROS2 observation as the state to match the checkpoint.
        """
        # Use the full 101-element observation directly as state
        # This matches what the checkpoint expects (trained on 101-element obs)
        state = obs_raw[:101]

        # For privileged state, we need to match the training size of 212 elements
        # Pad with zeros for missing privileged components
        # 212 - 101 = 111 padding elements needed
        privileged_state = jp.concatenate([state, jp.zeros(111)])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_termination(self, obs_raw: jax.Array, info: dict) -> jax.Array:
        """Check if episode should terminate."""
        # Check if we've exceeded episode length
        length_done = info["step"] >= self._config.episode_length
        # Check if we've looped back to start (use JAX logical operations)
        loop_done = (
            (info["data_idx"] == info["episode_start_idx"])
            & (info["step"] > 0)
        )
        return length_done | loop_done

    def _get_reward_from_obs(
        self,
        obs_raw: jax.Array,
        action: jax.Array,
        info: dict,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute rewards from observation data."""
        # Extract components from observation
        # obs_raw format: [gyro(3), accel(3), cmd(7D), joint_pos(10), joint_vel(10),
        #                  last_act(10), last_last_act(10), last_last_last_act(10),
        #                  motor_targets(10), contact(2), imitation_phase(2)]

        gyro = obs_raw[0:3]
        cmd_7d = obs_raw[6:13]
        cmd = cmd_7d[:3]  # Use first 3 elements for velocity command
        joint_pos = obs_raw[13:23]
        joint_vel = obs_raw[23:33]
        contact = obs_raw[73:75]

        # Use actual base linear velocity from velocimeter sensor if available
        # This provides the real measured velocity instead of using command as proxy
        current_idx = info.get("data_idx", jp.array(0, dtype=jp.int32))
        # Clip index to valid range to avoid out-of-bounds
        data_len = self._base_linear_velocities.shape[0]
        idx_clipped = jp.clip(current_idx, 0, data_len - 1)
        # Use actual measured velocity from velocimeter sensor
        # If index is out of bounds, use command as fallback
        linvel_estimate = jp.where(
            current_idx < data_len,
            self._base_linear_velocities[idx_clipped],
            jp.array([cmd[0], cmd[1], 0.0])  # Fallback: use command as proxy
        )

        ret = {
            "tracking_lin_vel": reward_tracking_lin_vel(
                cmd,
                linvel_estimate,
                self._config.reward_config.tracking_sigma,
            ),
            "tracking_ang_vel": reward_tracking_ang_vel(
                cmd,
                gyro,
                self._config.reward_config.tracking_sigma,
            ),
            "torques": jp.zeros(()),  # No torque data in replay
            "action_rate": cost_action_rate(action, info["last_act"]),
            "alive": reward_alive(),
            "imitation": jp.zeros(()),  # Simplified for replay
            "stand_still": cost_stand_still(
                cmd,
                joint_pos,
                joint_vel,
                self._default_actuator_legs_only,
                ignore_head=True,  # Observation only has leg joints, so ignore head
            ),
        }

        return ret

