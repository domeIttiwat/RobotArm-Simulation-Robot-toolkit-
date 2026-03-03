#!/usr/bin/env python3
"""
robot_controller.py
====================
ROS2 (Humble / Jazzy) node for a 6-axis robot arm with linear rail.

Bridges the FIBO Robot Cafe Studio web UI (via rosbridge v2) to
MoveIt2 motion planning and hardware controllers.

Quick-start
-----------
1. Source your ROS2 + MoveIt2 workspace:
      source /opt/ros/humble/setup.bash
      source ~/robot_ws/install/setup.bash

2. Launch MoveIt2 for your robot (adjust the launch file):
      ros2 launch arctos_moveit_config move_group.launch.py

3. Launch rosbridge for web UI:
      ros2 launch rosbridge_server rosbridge_websocket_launch.xml

4. Run this node:
      ros2 run robot_pkg robot_controller
   or directly:
      python3 robot_controller.py

Dependencies
------------
  pip install scipy numpy
  apt install ros-humble-moveit ros-humble-moveit-py

Configuration
-------------
  Adjust ARM_GROUP, EE_LINK, BASE_FRAME, JOINT_NAMES to match your SRDF/URDF.
"""

from __future__ import annotations

import json
import math
import threading
import time
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int8, String

# MoveIt2 Python bindings (ros-humble-moveit-py)
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState

# MoveIt2 services / messages
from moveit_msgs.msg import RobotTrajectory
from moveit_msgs.srv import GetCartesianPath

try:
    from scipy.spatial.transform import Rotation as _Rotation
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class RobotStatus(IntEnum):
    IDLE      = 0   # ready for commands
    EXECUTING = 1   # motion in progress
    PAUSED    = 2   # motion on hold, holding current position


class SafetyStatus(IntEnum):
    NORMAL         = 0
    WARNING        = 1   # reduced speed (50 %)
    EMERGENCY_STOP = 2   # halt immediately


# ─────────────────────────────────────────────────────────────────────────────
# RobotController
# ─────────────────────────────────────────────────────────────────────────────

class RobotController(Node):
    """
    ROS2 node that bridges the web UI to MoveIt2 motion planning.

    State Machine
    ~~~~~~~~~~~~~
      IDLE → EXECUTING → IDLE            (goto / trajectory — happy path)
      EXECUTING → PAUSED → EXECUTING     (pause / resume)
      ANY       → IDLE                   (stop / emergency stop)

    Threading Model
    ~~~~~~~~~~~~~~~
      • Subscriber callbacks  — ROS MultiThreadedExecutor threads
      • Motion execution      — one dedicated daemon thread (_motion_thread)
      • Joint-state publisher — ROS timer callback (10 Hz)
      • Thread safety via threading.Lock / Event (no shared mutable state
        touched from two threads simultaneously)
    """

    # ── MoveIt2 configuration — adjust to your URDF / SRDF ───────────────────
    ARM_GROUP   = "arm"         # MoveIt2 planning group for J1–J6
    EE_LINK     = "tool0"       # end-effector link name
    BASE_FRAME  = "base_link"   # planning reference frame
    JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]

    # Cartesian planning
    MAX_CART_SPEED_MS  = 0.25   # m/s — scales with speed_frac
    CART_STEP_M        = 0.01   # 1 cm interpolation resolution

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        super().__init__("robot_controller")

        # ── MoveIt2 ──────────────────────────────────────────────────────────
        self._moveit = MoveItPy(node_name="robot_controller_moveit")
        self._arm    = self._moveit.get_planning_component(self.ARM_GROUP)

        # Service client: straight-line Cartesian path planning
        self._cartesian_cli = self.create_client(
            GetCartesianPath, "/compute_cartesian_path"
        )

        # ── Internal state (all guarded by _status_lock / Events) ────────────
        self._status      : RobotStatus  = RobotStatus.IDLE
        self._safety      : SafetyStatus = SafetyStatus.NORMAL
        self._status_lock = threading.Lock()

        # Shadow joint state — updated after each successful move
        # and read back from MoveIt2 in the 10 Hz timer
        self._q_deg   : List[float] = [0.0] * 6   # J1-J6, degrees
        self._q_vel   : List[float] = [0.0] * 6   # J1-J6 velocities, deg/s
        self._rail_mm : float       = 0.0          # mm
        self._gripper : float       = 0.0          # percent

        # TCP offset (meters) — updated via /tool_config
        self._tcp_x   : float       = 0.0
        self._tcp_y   : float       = 0.0
        self._tcp_z   : float       = 0.0

        # ── Motion thread synchronisation ─────────────────────────────────────
        #   _resume_event  SET → run   CLEAR → paused (motion thread blocks here)
        #   _stop_event    SET → abort current motion
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._stop_event   = threading.Event()
        self._motion_lock  = threading.Lock()   # only one motion at a time
        self._motion_thread: Optional[threading.Thread] = None

        # ── Callback groups ───────────────────────────────────────────────────
        sub_cbg   = MutuallyExclusiveCallbackGroup()
        timer_cbg = MutuallyExclusiveCallbackGroup()

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_status  = self.create_publisher(Int8,       "/robot_status",      10)
        self._pub_safety  = self.create_publisher(Int8,       "/safety_status",     10)
        self._pub_joints  = self.create_publisher(JointState, "/joint_states",      10)
        self._pub_ee_pose = self.create_publisher(String,     "/end_effector_pose", 10)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            String, "/goto_position",
            self.goto_position_cb, 10, callback_group=sub_cbg)
        self.create_subscription(
            String, "/execute_trajectory",
            self.execute_trajectory_cb, 10, callback_group=sub_cbg)
        self.create_subscription(
            Bool, "/pause_execution",
            self.pause_cb, 10, callback_group=sub_cbg)
        self.create_subscription(
            Bool, "/stop_execution",
            self.stop_cb, 10, callback_group=sub_cbg)
        self.create_subscription(
            Bool, "/teach_mode",
            self.teach_mode_cb, 10, callback_group=sub_cbg)
        self.create_subscription(
            Int8, "/safety_status",
            self.safety_status_cb, 10, callback_group=sub_cbg)
        self.create_subscription(
            String, "/tool_config",
            self.tool_config_cb, 10, callback_group=sub_cbg)

        # ── 10 Hz feedback timers ─────────────────────────────────────────────
        self.create_timer(0.1, self.joint_states_timer_cb,  callback_group=timer_cbg)
        self.create_timer(0.1, self.effector_pose_timer_cb, callback_group=timer_cbg)

        self.get_logger().info("RobotController ready.")

    # ─────────────────────────────────────────────────────────────────────────
    # Subscriber callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def goto_position_cb(self, msg: String) -> None:
        """
        /goto_position — std_msgs/String (JSON)

        Execute a single task.  Accepted fields:
          j1–j6 (deg), rail (mm), gripper (%), speed (1–100), controlMode
        """
        try:
            task: Dict[str, Any] = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"goto_position: JSON parse error — {exc}")
            return

        self.get_logger().info(
            f"goto_position  seq={task.get('sequence')}  "
            f"label='{task.get('label')}'  "
            f"mode={task.get('controlMode', 'joint')}"
        )
        self._start_motion(self._run_single_task, task)

    def execute_trajectory_cb(self, msg: String) -> None:
        """
        /execute_trajectory — std_msgs/String (JSON)

        Full job received upfront.  Executes all tasks in sequence order.
        Expected schema: { "id": N, "name": "...", "tasks": [ {...}, ... ] }
        """
        try:
            job: Dict[str, Any] = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"execute_trajectory: JSON parse error — {exc}")
            return

        tasks = sorted(
            job.get("tasks", []),
            key=lambda t: t.get("sequence", 0),
        )
        self.get_logger().info(
            f"execute_trajectory  job='{job.get('name')}'  tasks={len(tasks)}"
        )
        self._start_motion(self._run_task_list, tasks)

    def pause_cb(self, msg: Bool) -> None:
        """
        /pause_execution — std_msgs/Bool
          True  → pause current motion, publish PAUSED (2)
          False → resume motion, publish EXECUTING (1), then IDLE (0) when done
        """
        if msg.data:
            self._resume_event.clear()          # block motion thread at checkpoint
            self._set_status(RobotStatus.PAUSED)
            self.get_logger().info("Motion PAUSED.")
        else:
            self._resume_event.set()            # unblock motion thread
            self.get_logger().info("Motion RESUMED.")
            # The motion thread will re-publish EXECUTING when it unblocks

    def stop_cb(self, msg: Bool) -> None:
        """
        /stop_execution — std_msgs/Bool
          True → emergency stop; decelerate and abort.  Publish IDLE (0).
        """
        if not msg.data:
            return
        self.get_logger().warn("STOP — aborting all motion.")
        self._stop_event.set()
        self._resume_event.set()    # unblock if currently paused
        try:
            # Cancel in-progress MoveIt2 execution (API may vary by version)
            self._moveit.stop_arm()
        except Exception:
            pass
        self._set_status(RobotStatus.IDLE)

    def teach_mode_cb(self, msg: Bool) -> None:
        """
        /teach_mode — std_msgs/Bool
          True  → enable free-drive (release joint brakes)
          False → return to position hold
        """
        if msg.data:
            self.get_logger().info("Teach mode ON — releasing joint brakes.")
            # TODO: call your hardware free-drive service / action
            #   e.g. ros2 service call /ur_driver/set_freedrive_mode ...
        else:
            self.get_logger().info("Teach mode OFF — position hold active.")
            # TODO: re-engage joint brakes / impedance control

    def tool_config_cb(self, msg: String) -> None:
        """
        /tool_config — std_msgs/String (JSON)

        Update the TCP (Tool Centre Point) offset applied to all subsequent
        IK planning and end-effector pose reports.

        Payload:  { "tcp_x": mm, "tcp_y": mm, "tcp_z": mm }
        """
        try:
            cfg = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"tool_config: JSON parse error — {exc}")
            return

        self._tcp_x = float(cfg.get("tcp_x", 0.0)) / 1000.0   # mm → m
        self._tcp_y = float(cfg.get("tcp_y", 0.0)) / 1000.0
        self._tcp_z = float(cfg.get("tcp_z", 0.0)) / 1000.0

        self.get_logger().info(
            f"TCP offset updated: "
            f"({self._tcp_x * 1000:.1f}, {self._tcp_y * 1000:.1f}, "
            f"{self._tcp_z * 1000:.1f}) mm"
        )
        # No /robot_status acknowledgement required per spec.

    def safety_status_cb(self, msg: Int8) -> None:
        """
        /safety_status — std_msgs/Int8
          0 = Normal   1 = Warning (50% speed)   2 = Emergency Stop

        Handles commands from the iOS SafetyPanel or web UI.
        """
        try:
            new_safety = SafetyStatus(int(msg.data))
        except ValueError:
            self.get_logger().warn(f"safety_status: unknown value {msg.data}")
            return

        if new_safety == self._safety:
            return

        self._safety = new_safety
        self.get_logger().warn(f"Safety → {new_safety.name}")

        if new_safety == SafetyStatus.EMERGENCY_STOP:
            self.stop_cb(Bool(data=True))

        # Echo back so other subscribers stay in sync
        self._pub_safety.publish(Int8(data=int(new_safety)))

    # ─────────────────────────────────────────────────────────────────────────
    # 10 Hz joint-state publisher
    # ─────────────────────────────────────────────────────────────────────────

    def joint_states_timer_cb(self) -> None:
        """
        Read back joint positions from MoveIt2 state monitor
        (fed by joint_state_broadcaster → /joint_states) and re-publish
        the extended message that includes rail and gripper axes.
        """
        try:
            # get_current_state() reflects the latest hardware feedback via
            # ros2_control joint_state_broadcaster → MoveIt2 state monitor.
            state: RobotState = self._moveit.get_current_state()
            pos_rad = state.get_joint_group_positions(self.ARM_GROUP)
            if pos_rad is not None and len(pos_rad) >= 6:
                self._q_deg = [math.degrees(r) for r in pos_rad[:6]]
            # Uncomment to read back velocities if available:
            # vel_rad = state.get_joint_group_velocities(self.ARM_GROUP)
            # if vel_rad: self._q_vel = [math.degrees(v) for v in vel_rad[:6]]
        except Exception:
            pass    # fall through to last-known shadow values

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name         = self.JOINT_NAMES + ["rail", "gripper"]
        js.position     = list(self._q_deg) + [self._rail_mm, self._gripper]
        js.velocity     = list(self._q_vel)
        self._pub_joints.publish(js)

    def effector_pose_timer_cb(self) -> None:
        """
        Publish current end-effector Cartesian pose at 10 Hz.

        Pose is computed via MoveIt2 FK from the current robot state,
        with the TCP offset applied.  Published even when idle.

        Topic: /end_effector_pose — std_msgs/String (JSON)
        Units: x, y, z in mm;  roll, pitch, yaw in degrees.
        """
        pose = self._get_current_pose_mm_deg()
        if pose is None:
            return
        msg = String()
        msg.data = json.dumps(pose)
        self._pub_ee_pose.publish(msg)

    # ─────────────────────────────────────────────────────────────────────────
    # Public motion methods
    # ─────────────────────────────────────────────────────────────────────────

    def execute_joint_mode(self, task: Dict[str, Any]) -> bool:
        """
        Joint-space motion: set_joint_value_target → OMPL plan → execute.

        Path is planned entirely in joint space (fast, predictable, avoids
        singularities).  All joints move simultaneously and finish together.

        Parameters
        ----------
        task : dict
            JSON task dict.  Used keys: j1–j6 (deg), rail (mm), speed (%).

        Returns
        -------
        bool  True on success.
        """
        speed_frac = self._clamp_speed(float(task.get("speed", 30)))
        joints_rad = self._task_to_radians(task)
        rail_mm    = float(task.get("rail", self._rail_mm))

        self.get_logger().info(
            f"[JointMode] "
            f"target_deg={[round(math.degrees(r), 1) for r in joints_rad]}  "
            f"rail={rail_mm:.0f} mm  speed={speed_frac * 100:.0f} %"
        )

        # Build a goal RobotState from target joint angles
        goal_state = RobotState(self._moveit.get_robot_model())
        goal_state.set_joint_group_positions(self.ARM_GROUP, joints_rad)
        goal_state.update()

        # Configure OMPL plan request
        params = PlanRequestParameters(self._moveit, "ompl")
        params.planning_time                   = 5.0
        params.max_velocity_scaling_factor     = speed_frac
        params.max_acceleration_scaling_factor = speed_frac * 0.5

        self._arm.set_start_state_to_current_state()
        self._arm.set_goal_state(robot_state=goal_state)
        plan_result = self._arm.plan(single_plan_parameters=params)

        if not plan_result:
            self.get_logger().error("[JointMode] Planning failed.")
            return False

        ok = self._moveit.execute(plan_result.trajectory, blocking=True, controllers=[])
        if not ok:
            self.get_logger().error("[JointMode] Execution failed.")
            return False

        # Update shadow state after successful execution
        self._q_deg = [math.degrees(r) for r in joints_rad]
        self._move_rail(rail_mm, speed_frac)
        return True

    def execute_effector_mode(self, task: Dict[str, Any]) -> bool:
        """
        Cartesian (end-effector) motion:
          1. Use x/y/z/roll/pitch/yaw from task payload (with TCP offset applied)
             Fallback to FK from j1–j6 hint if Cartesian fields are absent.
          2. compute_cartesian_path → straight-line EE trajectory
          3. Execute (fallback to OMPL set_pose_target if coverage < 90 %)

        Parameters
        ----------
        task : dict
            JSON task dict.

        Returns
        -------
        bool  True on success.
        """
        speed_frac = self._clamp_speed(float(task.get("speed", 30)))
        rail_mm    = float(task.get("rail", self._rail_mm))

        # ── Step 1: build Cartesian target from task payload ──────────────────
        # Primary: use x/y/z/roll/pitch/yaw directly (mm → m, deg → rad).
        # Fallback: FK from j1–j6 hint if Cartesian fields are absent.
        x_raw = task.get("x")
        if x_raw is not None:
            x_m   = float(x_raw)             / 1000.0   # mm → m
            y_m   = float(task.get("y", 0.0)) / 1000.0
            z_m   = float(task.get("z", 0.0)) / 1000.0
            roll  = math.radians(float(task.get("roll",  0.0)))
            pitch = math.radians(float(task.get("pitch", 0.0)))
            yaw   = math.radians(float(task.get("yaw",   0.0)))

            # Apply TCP offset (meters)
            x_m += self._tcp_x
            y_m += self._tcp_y
            z_m += self._tcp_z

            quat = self._euler_to_quat(roll, pitch, yaw)
            target_pose = Pose()
            target_pose.position    = Point(x=x_m, y=y_m, z=z_m)
            target_pose.orientation = Quaternion(
                x=quat[0], y=quat[1], z=quat[2], w=quat[3]
            )
        else:
            # Fallback: FK from j1–j6 hint values
            joints_rad  = self._task_to_radians(task)
            target_pose = self._fk(joints_rad)
            if target_pose is None:
                self.get_logger().error(
                    "[EffectorMode] No Cartesian target and FK failed "
                    "— falling back to joint mode."
                )
                return self.execute_joint_mode(task)

        self.get_logger().info(
            f"[EffectorMode] "
            f"xyz=({target_pose.position.x * 1000:.1f}, "
            f"{target_pose.position.y * 1000:.1f}, "
            f"{target_pose.position.z * 1000:.1f}) mm  "
            f"speed={speed_frac * 100:.0f} %"
        )

        # ── Step 2: straight-line Cartesian path ─────────────────────────────
        traj, fraction = self._compute_cartesian_path([target_pose], speed_frac)

        if traj is not None and fraction >= 0.9:
            self.get_logger().info(
                f"[EffectorMode] Cartesian path coverage {fraction:.0%}."
            )
            ok = self._moveit.execute(traj, blocking=True, controllers=[])

        else:
            # Fallback: OMPL pose target (not strictly straight-line)
            self.get_logger().warn(
                f"[EffectorMode] Cartesian coverage {fraction:.0%} < 90 %  "
                "→ falling back to OMPL set_pose_target."
            )
            ps = PoseStamped()
            ps.header.frame_id = self.BASE_FRAME
            ps.pose            = target_pose

            params = PlanRequestParameters(self._moveit, "ompl")
            params.planning_time                   = 5.0
            params.max_velocity_scaling_factor     = speed_frac
            params.max_acceleration_scaling_factor = speed_frac * 0.5

            self._arm.set_start_state_to_current_state()
            self._arm.set_goal_state(pose_stamped_msg=ps, pose_link=self.EE_LINK)
            plan_result = self._arm.plan(single_plan_parameters=params)

            if not plan_result:
                self.get_logger().error("[EffectorMode] OMPL planning failed.")
                return False

            ok = self._moveit.execute(
                plan_result.trajectory, blocking=True, controllers=[]
            )

        if not ok:
            self.get_logger().error("[EffectorMode] Execution failed.")
            return False

        # Shadow joint angles are refreshed from MoveIt2 in the 10 Hz timer
        # after Cartesian execution (no joints_rad available in primary branch).
        self._move_rail(rail_mm, speed_frac)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Private: motion thread management
    # ─────────────────────────────────────────────────────────────────────────

    def _start_motion(self, fn, *args) -> None:
        """
        Launch fn(*args) in a daemon background thread.

        If a motion is already running it is pre-empted (stop signal sent,
        thread joined with 5 s timeout) before the new motion starts.
        """
        # If another motion is in progress, stop it first
        if not self._motion_lock.acquire(blocking=False):
            self.get_logger().warn("Pre-empting existing motion.")
            self._stop_event.set()
            self._resume_event.set()    # unblock if paused
            if self._motion_thread and self._motion_thread.is_alive():
                self._motion_thread.join(timeout=5.0)
            self._motion_lock.acquire()

        # Reset synchronisation events for the new motion
        self._stop_event.clear()
        self._resume_event.set()

        def _run() -> None:
            try:
                fn(*args)
            except Exception as exc:
                self.get_logger().error(f"Motion thread error: {exc}")
            finally:
                self._set_status(RobotStatus.IDLE)
                self._motion_lock.release()

        self._motion_thread = threading.Thread(target=_run, daemon=True)
        self._motion_thread.start()

    def _run_single_task(self, task: Dict[str, Any]) -> None:
        """Execute one task inside the motion thread."""
        self._set_status(RobotStatus.EXECUTING)

        mode    = task.get("controlMode", "joint").lower()
        success = (
            self.execute_joint_mode(task)
            if mode == "joint"
            else self.execute_effector_mode(task)
        )

        if success and not self._stop_event.is_set():
            self._set_gripper(float(task.get("gripper", self._gripper)))

        self._set_status(RobotStatus.IDLE)

    def _run_task_list(self, tasks: List[Dict[str, Any]]) -> None:
        """
        Execute a list of tasks sequentially.

        Pause/resume checkpoint is evaluated before each task, so the robot
        always finishes its current task before pausing.
        """
        self._set_status(RobotStatus.EXECUTING)

        for task in tasks:
            if self._stop_event.is_set():
                break

            # ── Pause checkpoint ─────────────────────────────────────────────
            if not self._resume_event.wait(timeout=300.0):
                self.get_logger().warn("Resume wait timed out — aborting trajectory.")
                break
            if self._stop_event.is_set():
                break

            self._set_status(RobotStatus.EXECUTING)   # re-announce after resume

            mode    = task.get("controlMode", "joint").lower()
            success = (
                self.execute_joint_mode(task)
                if mode == "joint"
                else self.execute_effector_mode(task)
            )

            if not success:
                self.get_logger().error(
                    f"Task seq={task.get('sequence')} failed — stopping trajectory."
                )
                break

            if not self._stop_event.is_set():
                self._set_gripper(float(task.get("gripper", self._gripper)))

        self._set_status(RobotStatus.IDLE)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: MoveIt2 helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _fk(self, joints_rad: List[float]) -> Optional[Pose]:
        """
        Forward kinematics via MoveIt2 RobotState.
        Returns geometry_msgs/Pose or None on failure.
        """
        try:
            state = RobotState(self._moveit.get_robot_model())
            state.set_joint_group_positions(self.ARM_GROUP, joints_rad)
            state.update()
            # Returns a 4×4 numpy homogeneous transform matrix
            transform: np.ndarray = state.get_global_link_transform(self.EE_LINK)
            return self._matrix_to_pose(transform)
        except Exception as exc:
            self.get_logger().error(f"FK error: {exc}")
            return None

    def _compute_cartesian_path(
        self,
        waypoints: List[Pose],
        speed_frac: float,
    ) -> Tuple[Optional[RobotTrajectory], float]:
        """
        Call /compute_cartesian_path service to plan a straight-line EE path.

        Returns (RobotTrajectory, fraction) where fraction ∈ [0, 1].
        Returns (None, 0) on service failure or timeout.
        """
        if not self._cartesian_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "compute_cartesian_path service unavailable — skipping."
            )
            return None, 0.0

        req = GetCartesianPath.Request()
        req.header.frame_id      = self.BASE_FRAME
        req.header.stamp         = self.get_clock().now().to_msg()
        req.group_name           = self.ARM_GROUP
        req.link_name            = self.EE_LINK
        req.waypoints            = waypoints
        req.max_step             = self.CART_STEP_M
        req.jump_threshold       = 0.0     # disable jump detection
        req.avoid_collisions     = True
        req.max_cartesian_speed  = self.MAX_CART_SPEED_MS * speed_frac

        future = self._cartesian_cli.call_async(req)

        # Spin-wait (this runs in the motion thread, not the ROS executor thread)
        deadline = time.monotonic() + 10.0
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().error(
                    "compute_cartesian_path service request timed out."
                )
                return None, 0.0
            time.sleep(0.02)

        resp = future.result()
        if resp is None:
            return None, 0.0
        return resp.solution, float(resp.fraction)

    # ─────────────────────────────────────────────────────────────────────────
    # Private: hardware stubs
    # ─────────────────────────────────────────────────────────────────────────

    def _set_gripper(self, pct: float) -> None:
        """
        Command gripper to pct % (0–100).

        TODO: replace with your gripper controller call, e.g.:
          - GripperCommand action (control_msgs)
          - ros2_control JointGroupPositionController trajectory
        """
        pct = float(max(0.0, min(100.0, pct)))
        self.get_logger().info(f"Gripper → {pct:.1f} %")
        self._gripper = pct

    def _move_rail(self, target_mm: float, speed_frac: float = 1.0) -> None:
        """
        Move linear rail to target_mm (0–600 mm).

        TODO: replace with your rail controller call, e.g.:
          - FollowJointTrajectory action on the rail controller
          - ros2_control position interface
        """
        target_mm = float(max(0.0, min(600.0, target_mm)))
        self.get_logger().info(
            f"Rail → {target_mm:.1f} mm  speed={speed_frac * 100:.0f} %"
        )
        self._rail_mm = target_mm

    # ─────────────────────────────────────────────────────────────────────────
    # Private: utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, status: RobotStatus) -> None:
        """Update robot status and publish /robot_status on every state change."""
        with self._status_lock:
            if self._status == status:
                return
            self._status = status
        self._pub_status.publish(Int8(data=int(status)))
        self.get_logger().info(f"RobotStatus → {status.name} ({int(status)})")

    def _clamp_speed(self, speed_pct: float) -> float:
        """
        Convert speed field (1–100 %) to a MoveIt2 scaling fraction [0.01, 1.0].
        Applies the current safety speed multiplier:
          Normal = 1.0 ×   Warning = 0.5 ×   E-Stop = 0.0 ×
        """
        safety_mult = {
            SafetyStatus.NORMAL:         1.0,
            SafetyStatus.WARNING:        0.5,
            SafetyStatus.EMERGENCY_STOP: 0.0,
        }.get(self._safety, 1.0)
        return min(max(speed_pct / 100.0, 0.01), 1.0) * safety_mult

    def _task_to_radians(self, task: Dict[str, Any]) -> List[float]:
        """Extract J1–J6 from a task dict and convert degrees → radians."""
        return [
            math.radians(float(task.get(j, 0.0)))
            for j in ["j1", "j2", "j3", "j4", "j5", "j6"]
        ]

    @staticmethod
    def _matrix_to_pose(transform: np.ndarray) -> Pose:
        """
        Convert a 4×4 homogeneous transform matrix (numpy) to
        geometry_msgs/Pose.

        Uses scipy for the rotation conversion if available, otherwise falls
        back to Shepperd's numerically stable method.
        """
        pose = Pose()
        pose.position = Point(
            x=float(transform[0, 3]),
            y=float(transform[1, 3]),
            z=float(transform[2, 3]),
        )

        if _HAS_SCIPY:
            rot  = _Rotation.from_matrix(transform[:3, :3])
            quat = rot.as_quat()    # [x, y, z, w]
        else:
            quat = RobotController._shepperd_quat(transform[:3, :3])

        pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]),
            z=float(quat[2]), w=float(quat[3]),
        )
        return pose

    def _get_current_pose_mm_deg(self) -> Optional[Dict[str, float]]:
        """
        Return the current end-effector pose with TCP offset applied.

        Units returned:  x, y, z in mm;  roll, pitch, yaw in degrees.
        Returns None on failure (e.g. MoveIt2 not ready yet).
        """
        try:
            state = self._moveit.get_current_state()
            transform: np.ndarray = state.get_global_link_transform(self.EE_LINK)

            # Translation (m → mm) + TCP offset
            x_mm = float(transform[0, 3] + self._tcp_x) * 1000.0
            y_mm = float(transform[1, 3] + self._tcp_y) * 1000.0
            z_mm = float(transform[2, 3] + self._tcp_z) * 1000.0

            # Rotation matrix → Euler RPY (degrees)
            if _HAS_SCIPY:
                rot = _Rotation.from_matrix(transform[:3, :3])
                rpy = rot.as_euler("xyz", degrees=True)
                roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            else:
                # Shepperd-based fallback: extract roll/pitch/yaw from 3×3 matrix
                m = transform[:3, :3]
                pitch = math.degrees(math.asin(-float(m[2, 0])))
                cy = math.cos(math.radians(pitch))
                if abs(cy) > 1e-6:
                    roll = math.degrees(math.atan2(float(m[2, 1]) / cy, float(m[2, 2]) / cy))
                    yaw  = math.degrees(math.atan2(float(m[1, 0]) / cy, float(m[0, 0]) / cy))
                else:
                    roll = math.degrees(math.atan2(-float(m[1, 2]), float(m[1, 1])))
                    yaw  = 0.0

            return {
                "x":     round(x_mm, 2),
                "y":     round(y_mm, 2),
                "z":     round(z_mm, 2),
                "roll":  round(roll,  2),
                "pitch": round(pitch, 2),
                "yaw":   round(yaw,   2),
            }
        except Exception as exc:
            self.get_logger().debug(f"get_current_pose_mm_deg: {exc}")
            return None

    @staticmethod
    def _euler_to_quat(roll: float, pitch: float, yaw: float) -> List[float]:
        """
        Convert Euler angles (rad, XYZ extrinsic) to quaternion [x, y, z, w].
        Uses scipy if available, otherwise uses closed-form trigonometry.
        """
        if _HAS_SCIPY:
            rot  = _Rotation.from_euler("xyz", [roll, pitch, yaw])
            quat = rot.as_quat()       # [x, y, z, w]
            return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]

        # Closed-form (assumes XYZ intrinsic = ZYX extrinsic reversed)
        cr, sr = math.cos(roll / 2),  math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2),   math.sin(yaw / 2)
        return [
            sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy,   # z
            cr * cp * cy + sr * sp * sy,   # w
        ]

    @staticmethod
    def _shepperd_quat(m: np.ndarray) -> List[float]:
        """
        Shepperd's method — numerically stable quaternion from a 3×3 rotation
        matrix.  Returns [x, y, z, w].
        """
        tr = float(m[0, 0] + m[1, 1] + m[2, 2])
        if tr > 0.0:
            s = 0.5 / math.sqrt(tr + 1.0)
            return [(m[2,1]-m[1,2])*s, (m[0,2]-m[2,0])*s,
                    (m[1,0]-m[0,1])*s, 0.25/s]
        elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
            s = 2.0 * math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            return [0.25*s, (m[0,1]+m[1,0])/s,
                    (m[0,2]+m[2,0])/s, (m[2,1]-m[1,2])/s]
        elif m[1,1] > m[2,2]:
            s = 2.0 * math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            return [(m[0,1]+m[1,0])/s, 0.25*s,
                    (m[1,2]+m[2,1])/s, (m[0,2]-m[2,0])/s]
        else:
            s = 2.0 * math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            return [(m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s,
                    0.25*s, (m[1,0]-m[0,1])/s]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotController()

    # MultiThreadedExecutor lets the 10 Hz timer and subscriber callbacks
    # run concurrently without blocking each other.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
