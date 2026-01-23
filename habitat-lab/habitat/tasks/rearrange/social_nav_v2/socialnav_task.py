#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os.path as osp
from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from gym import spaces

from habitat.core.dataset import Episode
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorSuite
from habitat.tasks.nav.nav import NavigationTask
from habitat.tasks.rearrange.rearrange_sim import (
    RearrangeSim,
    add_perf_timing_func,
)
from habitat.tasks.rearrange.utils import (
    CacheHelper,
    CollisionDetails,
    UsesArticulatedAgentInterface,
    rearrange_collision,
    rearrange_logger,
)
from habitat.datasets.rearrange.navmesh_utils import get_largest_island_index
import magnum as mn

def quaternion_to_rad_angle(source_rotation):
    rad_angle = 2 * np.arctan2(np.sqrt(source_rotation[1]**2 + source_rotation[2]**2 + source_rotation[3]**2), source_rotation[0])
    return rad_angle

def set_audio_transform(
    audio_sensor,
    source_position=None,
    listener_position=None,
    listener_rotation=None
) -> None:
    """
    设置音频传感器的声源和监听器变换
    
    Args:
        audio_sensor: 音频传感器对象
        source_position: 声源位置 (mn.Vector3, 可选)
        listener_position: 监听器位置 (mn.Vector3, 可选)
        listener_rotation: 监听器旋转 (quaternion.quaternion, 可选)
    """
    # 设置音频声源位置
    audio_sensor.setAudioSourceTransform(source_position) if source_position is not None else None

    # 仅在两个参数都提供时设置监听器变换
    if listener_position is not None and listener_rotation is not None:
        # 将 quaternion.quaternion 转换为 magnum.Vector4
        # 注意：magnum 的四元数顺序是 (w, x, y, z)
        rotation_vector4 = mn.Vector4(
            listener_rotation.w,
            listener_rotation.x,
            listener_rotation.y,
            listener_rotation.z,
        )
        audio_sensor.setAudioListenerTransform(listener_position, rotation_vector4)

@registry.register_task(name="MultiAgentPointNavTask-v0")
class MultiAgentPointNavTask(NavigationTask):

    """
    Defines additional logic for valid collisions and gripping shared between
    all rearrangement tasks.
    """

    _cur_episode_step: int
    _articulated_agent_pos_start: Dict[str, Tuple[np.ndarray, float]]
    # 盲区人类相关属性
    _blind_spot_humans: List[int]  # 盲区人类索引列表
    _blind_spot_trigger_distance: float  # 触发距离
    _blind_spot_triggered: Dict[int, bool]  # 记录每个盲区人类是否已触发
    _blind_spot_current_waypoint: Dict[int, int]  # 每个盲区人类当前的waypoint索引
    _bound_human_finished: Dict[int, bool]  # 绑定人类是否已完成所有waypoint移动

    def _duplicate_sensor_suite(self, sensor_suite: SensorSuite) -> None:
        """
        Modifies the sensor suite in place to duplicate articulated agent specific sensors
        between the two articulated agents.
        """

        task_new_sensors: Dict[str, Sensor] = {}
        task_obs_spaces = OrderedDict()
        for agent_idx, agent_id in enumerate(self._sim.agents_mgr.agent_names):
            for sensor_name, sensor in sensor_suite.sensors.items():
                # if isinstance(sensor, UsesArticulatedAgentInterface):
                    new_sensor = copy.copy(sensor)
                    new_sensor.agent_id = agent_idx
                    full_name = f"{agent_id}_{sensor_name}"
                    task_new_sensors[full_name] = new_sensor
                    task_obs_spaces[full_name] = new_sensor.observation_space
                # else:
                #     task_new_sensors[sensor_name] = sensor
                #     task_obs_spaces[sensor_name] = sensor.observation_space

        sensor_suite.sensors = task_new_sensors
        sensor_suite.observation_spaces = spaces.Dict(spaces=task_obs_spaces)

    def __init__(
        self,
        *args,
        sim,
        dataset=None,
        should_place_articulated_agent=True,
        **kwargs,
    ) -> None:
        if hasattr(dataset.episodes[0], "targets"):
            self.n_objs = len(dataset.episodes[0].targets)
        else:
            self.n_objs = 0
        self._human_num = 0
        super().__init__(sim=sim, dataset=dataset, **kwargs)
        self.is_gripper_closed = False
        self._sim: RearrangeSim = sim
        self._ignore_collisions: List[Any] = []
        self._desired_resting = np.array(self._config.desired_resting_position)
        # self._need_human = self._config.need_human ## control to debug
        self._sim_reset = True
        self._targ_idx = None # int = 0
        self._episode_id: str = ""
        self._cur_episode_step = 0
        self._should_place_articulated_agent = should_place_articulated_agent
        self._seed = self._sim.habitat_config.seed
        self._min_distance_start_agents = (
            self._config.min_distance_start_agents
        )
        self._min_start_distance = self._config.min_start_distance
        # TODO: this patch supports hab2 benchmark fixed states, but should be refactored w/ state caching for multi-agent
        if (
            hasattr(self._sim.habitat_config.agents, "main_agent")
            and self._sim.habitat_config.agents[
                "main_agent"
            ].is_set_start_state
        ):
            self._should_place_articulated_agent = False
        
        # Get config options
        self._force_regenerate = self._config.force_regenerate
        self._should_save_to_cache = self._config.should_save_to_cache
        self._obj_succ_thresh = self._config.obj_succ_thresh
        self._enable_safe_drop = self._config.enable_safe_drop
        self._constraint_violation_ends_episode = (
            self._config.constraint_violation_ends_episode
        )
        self._constraint_violation_drops_object = (
            self._config.constraint_violation_drops_object
        )
        self._count_obj_collisions = self._config.count_obj_collisions

        if hasattr(dataset,"config"):
            data_path = dataset.config.data_path.format(split=dataset.config.split)
            fname = data_path.split("/")[-1].split(".")[0]
            cache_path = osp.join(
                osp.dirname(data_path),
                f"{fname}_{self._config.type}_robot_start.pickle",
            )

            if self._config.should_save_to_cache or osp.exists(cache_path):
                self._articulated_agent_init_cache = CacheHelper(
                    cache_path,
                    def_val={},
                    verbose=False,
                )
                self._articulated_agent_pos_start = (
                    self._articulated_agent_init_cache.load()
                )
            else:
                self._articulated_agent_pos_start = None
        else: 
            self._articulated_agent_pos_start = None
            
        if len(self._sim.agents_mgr) > 1: # add one agent situation
            # Duplicate sensors that handle articulated agents. One for each articulated agent.
            self._duplicate_sensor_suite(self.sensor_suite)

        self._use_episode_start_goal = True 

        self.agent0_episode_start_position = None
        self.agent0_episode_start_rotation = None
        self.agent1_episode_start_position = None
        self.agent1_episode_start_rotation = None
        self.agent2_episode_start_position = None
        self.agent2_episode_start_rotation = None

        self.agent3_episode_start_position = None
        self.agent3_episode_start_rotation = None
        self.agent4_episode_start_position = None
        self.agent4_episode_start_rotation = None
        self.agent5_episode_start_position = None
        self.agent5_episode_start_rotation = None
        
        self.agent6_episode_start_position = None
        self.agent6_episode_start_rotation = None
        self.agent7_episode_start_position = None
        self.agent7_episode_start_rotation = None
        self.agent8_episode_start_position = None
        self.agent8_episode_start_rotation = None        
        # 盲区人类初始化
        self._blind_spot_humans = []
        self._blind_spot_trigger_distance = 3.0  # 默认触发距离
        self._blind_spot_triggered = {}
        self._blind_spot_current_waypoint = {}
        self._bound_human_finished = {}  # 绑定人类移动完成状态
    def overwrite_sim_config(self, config: Any, episode: Episode) -> Any:
        return config

    @property
    def targ_idx(self):
        if self._targ_idx is None:
            return None
        return self._targ_idx

    @property
    def abs_targ_idx(self):
        if self._targ_idx is None:
            return None
        return self._sim.get_targets()[0][self._targ_idx]

    @property
    def desired_resting(self):
        return self._desired_resting

    def set_args(self, **kwargs):
        raise NotImplementedError("Task cannot dynamically set arguments")

    def set_sim_reset(self, sim_reset):
        self._sim_reset = sim_reset

    def _get_cached_articulated_agent_start(self, agent_idx: int = 0):
        start_ident = self._get_ep_init_ident(agent_idx)
        if (
            self._articulated_agent_pos_start is None
            or start_ident not in self._articulated_agent_pos_start
            or self._force_regenerate
        ):
            return None
        else:
            return self._articulated_agent_pos_start[start_ident]

    def _get_ep_init_ident(self, agent_idx):
        return f"{self._episode_id}_{agent_idx}"

    def _cache_articulated_agent_start(self, cache_data, agent_idx: int = 0):
        if (
            self._articulated_agent_pos_start is not None
            and self._should_save_to_cache
        ):
            start_ident = self._get_ep_init_ident(agent_idx)
            self._articulated_agent_pos_start[start_ident] = cache_data
            self._articulated_agent_init_cache.save(
                self._articulated_agent_pos_start
            )

    def _set_articulated_agent_start(self, agent_idx: int) -> None:
        articulated_agent_pos = None
        articulated_agent_start = self._get_cached_articulated_agent_start(
            agent_idx
        )
        if self._use_episode_start_goal:
            if agent_idx >= 0 and agent_idx <= 8:
                pos_attr = "agent{}_episode_start_position".format(agent_idx)
                rot_attr = "agent{}_episode_start_rotation".format(agent_idx)
                if hasattr(self, pos_attr) and getattr(self, pos_attr) is not None:
                    articulated_agent_pos = mn.Vector3(getattr(self, pos_attr))
                    articulated_agent_rot = getattr(self, rot_attr)
            else:
                # Handle the case when agent_idx is out of range
                print("Agent index out of range")
        elif articulated_agent_start is None:
            filter_agent_position = None
            if self._min_distance_start_agents > 0.0:
                # Force the agents to start a minimum distance apart.
                prev_pose_agents = [
                    np.array(
                        self._sim.get_agent_data(
                            agent_indx_prev
                        ).articulated_agent.base_pos
                    )
                    for agent_indx_prev in range(agent_idx)
                ]

                def _filter_agent_position(start_pos, start_rot):
                    start_pos_2d = start_pos[[0, 2]]
                    prev_pos_2d = [
                        prev_pose_agent[[0, 2]]
                        for prev_pose_agent in prev_pose_agents
                    ]
                    distances = np.array(
                        [
                            np.linalg.norm(start_pos_2d - prev_pos_2d_i)
                            for prev_pos_2d_i in prev_pos_2d
                        ]
                    )
                    return np.all(distances > self._min_distance_start_agents)

                filter_agent_position = _filter_agent_position
            (
                articulated_agent_pos,
                articulated_agent_rot,
            ) = self._sim.set_articulated_agent_base_to_random_point(
                agent_idx=agent_idx, filter_func=filter_agent_position
            )
            self._cache_articulated_agent_start(
                (articulated_agent_pos, articulated_agent_rot), agent_idx
            )
        else:
            (
                articulated_agent_pos,
                articulated_agent_rot,
            ) = articulated_agent_start

        if articulated_agent_pos is not None:
            articulated_agent = self._sim.get_agent_data(
                agent_idx
            ).articulated_agent
            articulated_agent.base_pos = articulated_agent_pos
            articulated_agent.base_rot = articulated_agent_rot
        else:
            articulated_agent = self._sim.get_agent_data(
                agent_idx
            ).articulated_agent
            articulated_agent.base_pos = mn.Vector3(-100+agent_idx,-100+agent_idx,-100+agent_idx)
            articulated_agent.base_rot = 0.0



    def _generate_pointnav_goal(self,n=1):
        if self._use_episode_start_goal: ## remain to improve
            self.goals[0] =  self.agent0_episode_goal
        else:
            max_tries = 10
            # attempt = 0
            max_attempts = 10

            for i in range(n):
                temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                    max_tries,
                    island_index=0 # self._largest_indoor_island_idx,
                )
                if i>0:
                    for attempt in range(max_attempts):
                        if attempt == max_attempts-1:
                            rearrange_logger.error( # try to print
                                f"Could not find a proper start for goal in episode {self.ep_info.episode_id}"
                            )
                        if len(self.goals) >= 1 and np.linalg.norm(temp_coord_nav - self.goals[-1],ord=2,axis=-1) < self._min_start_distance:
                            break
                        else:
                            temp_coord_nav = self._sim.pathfinder.get_random_navigable_point(
                            max_tries,
                            island_index=0 # self._largest_indoor_island_idx,
                        )

                self.goals[i] = temp_coord_nav
        
        self.nav_goal_pos = self.goals[0]

    @add_perf_timing_func()
    def reset(self, episode: Episode, fetch_observations: bool = True):
        self._episode_id = episode.episode_id
        if "human_num" in episode.info:
            self._human_num = episode.info['human_num']
        else:
            self._human_num = 0
        if self._use_episode_start_goal:
            self.agent0_episode_start_position = episode.start_position
            # self.agent0_episode_start_rotation = quaternion_to_rad_angle(episode.start_rotation)
            self.agent0_episode_start_rotation = episode.start_rotation
            self.agent0_episode_goal = episode.goals[0].position
            for i in range(self._human_num):
                position_key = f"human_{i}_waypoint_0_position"
                rotation_key = f"human_{i}_waypoint_0_rotation"
                setattr(self, f"agent{i+1}_episode_start_position", episode.info[position_key])
                setattr(self, f"agent{i+1}_episode_start_rotation", episode.info[rotation_key])
        else:
            num_goals = 1
            self.goals = [np.array([0, 0, 0], dtype=np.float32) for _ in range(num_goals)]
            self._generate_pointnav_goal(num_goals)

        self._ignore_collisions = []

        if self._sim_reset:
            self._sim.reset()
            for action_instance in self.actions.values():
                action_instance.reset(episode=episode, task=self)
            self._is_episode_active = True

            if self._should_place_articulated_agent:
                for agent_idx in range(self._sim.num_articulated_agents):
                    self._set_articulated_agent_start(agent_idx)
        
        self.prev_measures = self.measurements.get_metrics()
        self.coll_accum = CollisionDetails()
        self.prev_coll_accum = CollisionDetails()
        self.should_end = False
        self._done = False
        self._cur_episode_step = 0
        
        # habitat-avh: 初始化盲区人类状态
        self._blind_spot_humans = episode.info.get("blind_spot_humans", [])
        self._blind_spot_trigger_distance = episode.info.get("blind_spot_trigger_distance", 3.0)
        self._blind_spot_triggered = {human_idx: False for human_idx in self._blind_spot_humans}
        self._blind_spot_current_waypoint = {human_idx: 0 for human_idx in self._blind_spot_humans}
        
        # habitat-avh: 初始化绑定人类移动完成状态
        self._bound_human_finished = {}
        if "sounds" in episode.info:
            for sound in episode.info["sounds"]:
                bound_human_idx = sound.get("bound_human_idx")
                if bound_human_idx is not None:
                    self._bound_human_finished[bound_human_idx] = False
        
        # habitat-avh
        if "sounds" in episode.info:
            try:
                agent_state = self._sim.get_agent_state(0)
                for sound in episode.info["sounds"]:
                    audio_sensor_name = f"agent_0_{sound['sensor_name']}"
                    audio_sensor = self._sim.get_agent(0)._sensors[audio_sensor_name]
                    set_audio_transform(
                        audio_sensor=audio_sensor,
                        source_position=sound["position"],
                        listener_position=agent_state.position,
                        listener_rotation=agent_state.rotation
                    )
            except Exception as e:
                rearrange_logger.error(f"Error setting audio transform: {e}")
        if fetch_observations:
            self._sim.maybe_update_articulated_agent()
            return self._get_observations(episode)
        else:
            return None
        
    @add_perf_timing_func()
    def _get_observations(self, episode):
        # Fetch the simulator observations, all visual sensors.
        obs = self._sim.get_sensor_observations()

        if not self._sim.sim_config.enable_batch_renderer:
            # Post-process visual sensor observations
            obs = self._sim._sensor_suite.get_observations(obs)
        else:
            # Keyframes are added so that the simulator state can be reconstituted when batch rendering.
            # The post-processing step above is done after batch rendering.
            self._sim.add_keyframe_to_observations(obs)

        # Task sensors (all non-visual sensors)
        obs.update(
            self.sensor_suite.get_observations(
                observations=obs, episode=episode, task=self, should_time=True
            )
        )
        
        # habitat-avh: 静止盲区人类绑定的音频 RIR 置零
        obs = self._zero_static_blind_spot_audio(obs, episode)
        
        return obs

    def _is_violating_safe_drop(self, action_args):
        idxs, goal_pos = self._sim.get_targets()
        scene_pos = self._sim.get_scene_pos()
        target_pos = scene_pos[idxs]
        min_dist = np.min(
            np.linalg.norm(target_pos - goal_pos, ord=2, axis=-1)
        )
        return (
            self._sim.grasp_mgr.is_grasped
            and action_args.get("grip_action", None) is not None
            and action_args["grip_action"] < 0
            and min_dist < self._obj_succ_thresh
        )

    def _zero_static_blind_spot_audio(self, obs: Dict[str, Any], episode: Episode) -> Dict[str, Any]:
        """
        将静止状态人类绑定的音频传感器 RIR 置零
        
        静止状态包括：
        1. 盲区人类未被触发（等待状态）
        2. 绑定人类已完成所有 waypoint 移动（停止状态）
        
        Args:
            obs: 观测字典
            episode: 当前 episode
            
        Returns:
            处理后的观测字典
        """
        if "sounds" not in episode.info:
            return obs
        
        for sound in episode.info["sounds"]:
            bound_human_idx = sound.get("bound_human_idx")
            if bound_human_idx is None:
                continue
            
            should_zero = False
            
            # 检查该人类是否是盲区人类且未被触发
            if bound_human_idx in self._blind_spot_humans:
                if not self._blind_spot_triggered.get(bound_human_idx, False):
                    should_zero = True
            
            # 检查该人类是否已完成所有 waypoint 移动
            if self._bound_human_finished.get(bound_human_idx, False):
                should_zero = True
            
            if should_zero:
                # 人类静止，将对应音频置零
                sensor_name = sound.get("sensor_name")
                obs_key = f"agent_0_{sensor_name}"
                if obs_key in obs:
                    # 将音频数据置零（保持相同的 shape 和 dtype）
                    obs[obs_key] = np.zeros_like(obs[obs_key])
        
        return obs

    def _update_blind_spot_humans(self, episode: Episode) -> None:
        """
        更新盲区人类状态：当 agent_0 靠近时触发移动
        模拟拐角处突然冲出的人类场景
        """
        if not self._blind_spot_humans:
            return
        
        try:
            agent_state = self._sim.get_agent_state(0)
            agent_pos = np.array(agent_state.position)
            
            for human_idx in self._blind_spot_humans:
                # 获取人类当前位置
                human_agent_idx = human_idx + 1  # episode中0-based，agent中1-based
                try:
                    human_state = self._sim.get_agent_state(human_agent_idx)
                    human_pos = np.array(human_state.position)
                except Exception:
                    continue
                
                # 计算 agent_0 与人类的距离
                distance = np.linalg.norm(agent_pos[[0, 2]] - human_pos[[0, 2]])  # 2D距离
                
                # 如果距离小于触发距离且未触发，则触发移动
                if distance < self._blind_spot_trigger_distance and not self._blind_spot_triggered[human_idx]:
                    self._blind_spot_triggered[human_idx] = True
                    self._blind_spot_current_waypoint[human_idx] = 1  # 开始移动到下一个waypoint
                    rearrange_logger.info(
                        f"Blind spot human {human_idx} triggered! Agent distance: {distance:.2f}m"
                    )
                    
                    # 设置人类移动到下一个waypoint
                    self._move_blind_spot_human_to_next_waypoint(episode, human_idx)
                    
        except Exception as e:
            rearrange_logger.error(f"Error updating blind spot humans: {e}")

    def _move_blind_spot_human_to_next_waypoint(self, episode: Episode, human_idx: int) -> None:
        """
        将盲区人类移动到下一个waypoint
        如果没有下一个waypoint，标记该人类为已完成移动
        """
        current_wp_idx = self._blind_spot_current_waypoint.get(human_idx, 0)
        next_wp_idx = current_wp_idx
        
        position_key = f"human_{human_idx}_waypoint_{next_wp_idx}_position"
        rotation_key = f"human_{human_idx}_waypoint_{next_wp_idx}_rotation"
        
        if position_key in episode.info and rotation_key in episode.info:
            new_position = episode.info[position_key]
            new_rotation = episode.info[rotation_key]
            
            human_agent_idx = human_idx + 1
            try:
                articulated_agent = self._sim.get_agent_data(human_agent_idx).articulated_agent
                articulated_agent.base_pos = mn.Vector3(new_position)
                articulated_agent.base_rot = new_rotation
                rearrange_logger.info(
                    f"Blind spot human {human_idx} moving to waypoint {next_wp_idx}: {new_position}"
                )
                
                # 检查是否还有下一个waypoint
                next_next_wp_idx = next_wp_idx + 1
                next_position_key = f"human_{human_idx}_waypoint_{next_next_wp_idx}_position"
                if next_position_key not in episode.info:
                    # 没有下一个waypoint，标记该人类为已完成移动
                    self._bound_human_finished[human_idx] = True
                    rearrange_logger.info(
                        f"Blind spot human {human_idx} finished all waypoints, marking as static"
                    )
            except Exception as e:
                rearrange_logger.error(f"Error moving blind spot human {human_idx}: {e}")
        else:
            # 当前waypoint不存在，标记为已完成
            self._bound_human_finished[human_idx] = True

    def step(self, action: Dict[str, Any], episode: Episode):
        # habitat-avh: 盲区人类触发逻辑
        self._update_blind_spot_humans(episode)
        
        # habitat-avh: 更新音频位置（包括绑定人类的音频）
        if "sounds" in episode.info:
            try:
                agent_state = self._sim.get_agent_state(0)
                for sound in episode.info["sounds"]:
                    audio_sensor_name = f"agent_0_{sound['sensor_name']}"
                    audio_sensor = self._sim.get_agent(0)._sensors[audio_sensor_name]
                    
                    # 检查是否绑定了人类，如果是则更新音频源位置为人类当前位置
                    source_position = None
                    if "bound_human_idx" in sound:
                        bound_human_idx = sound["bound_human_idx"]
                        try:
                            # human_idx 在 episode 中是 0-based，但 agent 中是 1-based
                            human_state = self._sim.get_agent_state(bound_human_idx + 1)
                            source_position = human_state.position
                        except Exception:
                            pass
                    
                    set_audio_transform(
                        audio_sensor=audio_sensor,
                        source_position=source_position,
                        listener_position=agent_state.position,
                        listener_rotation=agent_state.rotation
                    )
            except Exception as e:
                rearrange_logger.error(f"Error setting audio transform: {e}")
        if "action_args" in action:
            action_args = action["action_args"]
            if self._enable_safe_drop and self._is_violating_safe_drop(
                action_args
            ):
                action_args["grip_action"] = None
        obs = super().step(action=action, episode=episode)

        self.prev_coll_accum = copy.copy(self.coll_accum)
        self._cur_episode_step += 1
        for grasp_mgr in self._sim.agents_mgr.grasp_iter:
            if (
                grasp_mgr.is_violating_hold_constraint()
                and self._constraint_violation_drops_object
            ):
                grasp_mgr.desnap(True)

        # habitat-avh: 静止盲区人类绑定的音频 RIR 置零
        obs = self._zero_static_blind_spot_audio(obs, episode)

        return obs

    def _check_episode_is_active(
        self,
        *args: Any,
        action: Union[int, Dict[str, Any]],
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        done = False
        if self.should_end:
            done = True

        # Check that none of the articulated agents are violating the hold constraint
        for grasp_mgr in self._sim.agents_mgr.grasp_iter:
            if (
                grasp_mgr.is_violating_hold_constraint()
                and self._constraint_violation_ends_episode
            ):
                done = True
                break

        if done:
            rearrange_logger.debug("-" * 10)
            rearrange_logger.debug("------ Episode Over --------")
            rearrange_logger.debug("-" * 10)

        return not done

    def get_coll_forces(self, articulated_agent_id):
        grasp_mgr = self._sim.get_agent_data(articulated_agent_id).grasp_mgr
        articulated_agent = self._sim.get_agent_data(
            articulated_agent_id
        ).articulated_agent
        snapped_obj = grasp_mgr.snap_idx
        articulated_agent_id = articulated_agent.sim_obj.object_id
        contact_points = self._sim.get_physics_contact_points()

        def get_max_force(contact_points, check_id):
            match_contacts = [
                x
                for x in contact_points
                if (check_id in [x.object_id_a, x.object_id_b])
                and (x.object_id_a != x.object_id_b)
            ]

            max_force = 0
            if len(match_contacts) > 0:
                max_force = max([abs(x.normal_force) for x in match_contacts])

            return max_force

        forces = [
            abs(x.normal_force)
            for x in contact_points
            if (
                x.object_id_a not in self._ignore_collisions
                and x.object_id_b not in self._ignore_collisions
            )
        ]
        max_force = max(forces) if len(forces) > 0 else 0

        max_obj_force = get_max_force(contact_points, snapped_obj)
        max_articulated_agent_force = get_max_force(
            contact_points, articulated_agent_id
        )
        return max_articulated_agent_force, max_obj_force, max_force

    def get_cur_collision_info(self, agent_idx) -> CollisionDetails:
        _, coll_details = rearrange_collision(
            self._sim, self._count_obj_collisions, agent_idx=agent_idx
        )
        return coll_details

    def get_n_targets(self) -> int:
        return self.n_objs
    
    @property
    def should_end(self) -> bool:
        return self._should_end

    @should_end.setter
    def should_end(self, new_val: bool):
        self._should_end = new_val
        ##
        # NB: _check_episode_is_active is called after step() but
        # before metrics are updated. Thus if should_end is set
        # by a metric, the episode will end on the _next_
        # step. This makes sure that the episode is ended
        # on the correct step.
        self._is_episode_active = (
            not self._should_end
        ) and self._is_episode_active
        if new_val:
            rearrange_logger.debug("-" * 40)
            rearrange_logger.debug(
                f"-----Episode {self._episode_id} requested to end after {self._cur_episode_step} steps.-----"
            )
            rearrange_logger.debug("-" * 40)