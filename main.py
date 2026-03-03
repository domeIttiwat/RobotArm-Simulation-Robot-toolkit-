import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
from swift import Swift, Slider, Label, Select
from spatialgeometry import Axes
import os
import json
import time
import threading
import queue
import websocket   # websocket-client package
from datetime import datetime

# ============================================================
# โหลดหุ่นยนต์ Arctos 6 แกนจากไฟล์ URDF
# ============================================================
URDF_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Arctos_description/urdf/Arctos.urdf")
SAVE_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_positions.json")
robot = rtb.Robot.URDF(URDF_PATH)
print(robot)

q_zero = np.zeros(6)
print("\n--- Forward Kinematics (Home) ---")
print(robot.fkine(q_zero))

# ============================================================
# Swift Simulator
# ============================================================
print("\nเปิด Swift simulator... (กด Ctrl+C เพื่อออก)")
env = Swift()
env.launch(realtime=True)

robot.q = q_zero
env.add(robot)

# เก็บสีเดิมจาก URDF (ต้องทำหลัง env.add(robot) เพื่อให้ geometry ถูก populate แล้ว)
original_colors = {}
for link in robot.links:
    original_colors[link.name] = [tuple(shape.color) for shape in link.geometry]

# Tip Axes: offset 70mm ตามแกน -Y ของ EE
TIP_OFFSET     = SE3(0, -0.07, 0)
TIP_OFFSET_INV = TIP_OFFSET.inv()

T_ee_init  = robot.fkine(robot.q)
T_tip_init = T_ee_init * TIP_OFFSET
tip_axes   = Axes(0.05, pose=T_tip_init)
env.add(tip_axes)

# ── Simulator state ───────────────────────────────────────────
joint_sliders = []
ik_sliders    = {}
ik_status     = {'ok': True}
ik_dirty      = {'flag': False}
fk_lock       = {'until': 0.0}
ik_last_solve = {'t': 0.0}
IK_MIN_INTERVAL = 0.08

init_tip = T_tip_init.t * 1000
init_rpy = np.rad2deg(T_tip_init.rpy())
target = {
    'x':  float(init_tip[0]), 'y':  float(init_tip[1]), 'z':  float(init_tip[2]),
    'rx': float(init_rpy[0]), 'ry': float(init_rpy[1]), 'rz': float(init_rpy[2]),
}

sim_rail          = {'v': 0.0}   # mm  (0–600)
sim_gripper       = {'v': 0.0}   # %   (0–100)
teach_mode_active = {'v': True}
sim_safety_status = {'v': 0}     # 0 = Normal, 1 = Warning, 2 = Emergency Stop

# ── Trajectory execution state ────────────────────────────────
#   status: 0 = IDLE, 2 = EXECUTING
traj = {
    'tasks':       [],
    'idx':         0,
    'running':     False,   # True ↔ status == EXECUTING
    'stop':        False,
    'waiting':     False,   # True = inter-task delay
    'delay_until': 0.0,
    'q_start':     None,
    'q_end':       None,
    'rail_start':  0.0,
    'rail_end':    0.0,
    'grip_start':  0.0,
    'grip_end':    0.0,
    'elapsed':     0.0,   # virtual elapsed time (speed-multiplier adjusted)
    'last_step_t': 0.0,   # wall-clock time of previous step_trajectory call
    'duration':    0.0,
}

WS_URL                = "ws://localhost:9090"
JS_PUBLISH_INTERVAL   = 0.1    # seconds
last_js_pub_t         = {'t': 0.0}
MAX_SPEED_DEG_PER_SEC = 90.0   # 100 % speed = 90 deg/sec

# ── Robot status codes (rosbridge v2 /robot_status) ───────────
STATUS_IDLE       = 0   # done / stopped
STATUS_TASK_DONE  = 1   # one task finished; more may follow  (must arrive < 600 ms)
STATUS_EXECUTING  = 2   # currently moving


# ============================================================
# RobotUIBridge — rosbridge v2 WebSocket client
# ============================================================
class RobotUIBridge:
    """
    Encapsulates the rosbridge v2 WebSocket connection.

    Wire format (rosbridge v2):
      subscribe  → {"op":"subscribe","topic":"/foo","type":"pkg/Type"}
      publish    → {"op":"publish",  "topic":"/foo","msg":{...}}
      incoming   → {"op":"publish",  "topic":"/foo","msg":{...}}

    Topics:
      SUBSCRIBE  /execute_trajectory  std_msgs/String  (JSON array of tasks)
                 /goto_position       std_msgs/String  (JSON single task)
                 /stop_execution      std_msgs/Bool
                 /teach_mode          std_msgs/Bool
      PUBLISH    /joint_states        sensor_msgs/JointState (custom, degrees)
                 /robot_status        std_msgs/Int32
                   0 = IDLE, 1 = TASK_DONE (heartbeat), 2 = EXECUTING

    Callback hooks (set before calling start()):
      on_execute_trajectory(tasks: list[dict])
      on_goto_position(task: dict)
      on_stop()
      on_teach_mode(enabled: bool)
    """

    SUBSCRIBE_TOPICS = [
        ("/execute_trajectory", "std_msgs/String"),
        ("/goto_position",      "std_msgs/String"),
        ("/stop_execution",     "std_msgs/Bool"),
        ("/teach_mode",         "std_msgs/Bool"),
        ("/safety_status",      "std_msgs/Int8"),
    ]

    # Topics this node advertises (publishes) to the web service.
    ADVERTISE_TOPICS = [
        ("/safety_status", "std_msgs/Int8"),
    ]

    def __init__(self, url: str = "ws://localhost:9090"):
        self.url       = url
        self.connected = False
        self._ws       = None
        self._cmd_q    = queue.Queue()

        # Callback hooks
        self.on_execute_trajectory = None
        self.on_goto_position      = None
        self.on_stop               = None
        self.on_teach_mode         = None
        self.on_safety_status      = None

    # ── Public API ──────────────────────────────────────────

    def start(self):
        """Create WebSocketApp and launch daemon thread (with auto-reconnect)."""
        self._ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        threading.Thread(
            target=lambda: self._ws.run_forever(reconnect=5),
            daemon=True,
        ).start()

    def drain(self):
        """
        Dispatch all queued incoming messages to callbacks.
        Call once per main-loop iteration (NOT from the WS thread).
        """
        while not self._cmd_q.empty():
            self._dispatch(self._cmd_q.get_nowait())

    def publish(self, topic: str, msg_dict: dict):
        """Send a rosbridge v2 publish frame. Thread-safe."""
        if self._ws:
            try:
                self._ws.send(json.dumps({"op": "publish", "topic": topic, "msg": msg_dict}))
            except Exception:
                pass

    def publish_status(self, code: int):
        """
        Publish /robot_status.
          code 0 = IDLE       (trajectory finished or stopped)
          code 1 = TASK_DONE  (one task complete; more may follow — send < 600 ms)
          code 2 = EXECUTING  (robot is moving)
        """
        self.publish("/robot_status", {"data": code})

    def publish_joint_states(self, q_deg: list, rail: float, gripper: float):
        """Publish /joint_states (degrees, mm, %)."""
        self.publish("/joint_states", {
            "name":     ["j1", "j2", "j3", "j4", "j5", "j6", "rail", "gripper"],
            "position": list(q_deg) + [rail, gripper],
            "velocity": [0.0] * 6,
        })

    @property
    def is_connected(self) -> bool:
        return self.connected and self._ws is not None and self._ws.sock is not None

    # ── Internal ────────────────────────────────────────────

    def _on_open(self, ws):
        self.connected = True
        print("\n[WS] connected to rosbridge")
        for topic, typ in self.ADVERTISE_TOPICS:
            ws.send(json.dumps({"op": "advertise", "topic": topic, "type": typ}))
        for topic, typ in self.SUBSCRIBE_TOPICS:
            ws.send(json.dumps({"op": "subscribe", "topic": topic, "type": typ}))

    def _on_message(self, ws, raw):
        try:
            self._cmd_q.put(json.loads(raw))
        except Exception:
            pass

    def _on_error(self, ws, err):
        print(f"\n[WS] error: {err}")

    def _on_close(self, ws, *_):
        self.connected = False
        print("\n[WS] closed")

    def _dispatch(self, msg: dict):
        """Route one incoming message to the appropriate callback."""
        topic = msg.get("topic", "")

        if topic == "/execute_trajectory":
            tasks = json.loads(msg["msg"]["data"])
            tasks.sort(key=lambda t: t.get("sequence", 0))
            if self.on_execute_trajectory:
                self.on_execute_trajectory(tasks)

        elif topic == "/goto_position":
            task = json.loads(msg["msg"]["data"])
            if self.on_goto_position:
                self.on_goto_position(task)

        elif topic == "/stop_execution":
            if msg["msg"]["data"] and self.on_stop:
                self.on_stop()

        elif topic == "/teach_mode":
            if self.on_teach_mode:
                self.on_teach_mode(bool(msg["msg"]["data"]))

        elif topic == "/safety_status":
            status = int(msg.get("msg", {}).get("data", 0))
            if self.on_safety_status:
                self.on_safety_status(status)


# ── Create bridge instance ────────────────────────────────────
bridge = RobotUIBridge(WS_URL)


# ============================================================
# Core simulator functions
# ============================================================

def update_ik_sliders_from_fk():
    T_tip = robot.fkine(robot.q) * TIP_OFFSET
    tp    = T_tip.t * 1000
    rpy   = np.rad2deg(T_tip.rpy())
    for key, val, mn, mx in [
        ('x',  tp[0],  -400, 400), ('y',  tp[1],  -500, 100), ('z',  tp[2],   50, 700),
        ('rx', rpy[0], -180, 180), ('ry', rpy[1], -180, 180), ('rz', rpy[2], -180, 180),
    ]:
        clipped = float(np.clip(val, mn, mx))
        target[key] = clipped
        ik_sliders[key].value = clipped


def solve_ik():
    T_ee_target = (
        SE3(target['x'] / 1000, target['y'] / 1000, target['z'] / 1000)
        * SE3.RPY(np.deg2rad([target['rx'], target['ry'], target['rz']]))
        * TIP_OFFSET_INV
    )
    q_sol, ok, *_ = robot.ik_LM(T_ee_target, q0=robot.q, ilimit=500)
    if not ok:
        q_sol, ok, *_ = robot.ik_LM(T_ee_target, q0=q_zero, ilimit=1000)
    ik_status['ok'] = bool(ok)
    if ok:
        robot.q = q_sol
        q_deg = np.rad2deg(q_sol)
        for i, s in enumerate(joint_sliders):
            s.value = float(np.clip(q_deg[i], s.min, s.max))
    else:
        update_ik_sliders_from_fk()


def make_joint_cb(j):
    def cb(x):
        if not teach_mode_active['v']:
            return
        desired = np.deg2rad(float(x))
        if abs(desired - robot.q[j]) < np.deg2rad(0.6):
            return
        q = robot.q.copy()
        q[j] = desired
        robot.q = q
        fk_lock['until'] = time.time() + 0.3
        update_ik_sliders_from_fk()
    return cb


def make_target_cb(key):
    def cb(x):
        if not teach_mode_active['v']:
            return
        val = float(x)
        if abs(val - target[key]) < ik_sliders[key].step * 0.6:
            return
        target[key] = val
        ik_dirty['flag'] = True
    return cb


def save_position():
    T_ee  = robot.fkine(robot.q)
    T_tip = T_ee * TIP_OFFSET
    tp    = T_tip.t * 1000
    rpy   = np.rad2deg(T_tip.rpy())
    entry = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "q_deg":       [round(float(np.rad2deg(robot.q[i])), 2) for i in range(6)],
        "tip_mm":      [round(float(tp[0]), 1), round(float(tp[1]), 1), round(float(tp[2]), 1)],
        "tip_rpy_deg": [round(float(rpy[0]), 1), round(float(rpy[1]), 1), round(float(rpy[2]), 1)],
    }
    data = []
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            
            try:
                data = json.load(f)
            except Exception:
                data = []
    data.append(entry)
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(data)


# ============================================================
# Trajectory execution helpers
# ============================================================

def _load_next_task():
    """Set up interpolation parameters for traj['idx']."""
    task      = traj['tasks'][traj['idx']]
    speed_pct = max(float(task.get('speed', 30)), 1.0)
    speed     = (speed_pct / 100.0) * MAX_SPEED_DEG_PER_SEC   # deg/sec

    q_target = np.deg2rad([
        float(task.get('j1', 0)), float(task.get('j2', 0)),
        float(task.get('j3', 0)), float(task.get('j4', 0)),
        float(task.get('j5', 0)), float(task.get('j6', 0)),
    ])
    delta_deg = np.abs(np.rad2deg(q_target - robot.q))
    duration  = max(float(np.max(delta_deg)) / speed, 0.05)

    traj['q_start']    = robot.q.copy()
    traj['q_end']      = q_target
    traj['rail_start'] = sim_rail['v']
    traj['rail_end']   = float(task.get('rail',    sim_rail['v']))
    traj['grip_start'] = sim_gripper['v']
    traj['grip_end']   = float(task.get('gripper', sim_gripper['v']))
    traj['elapsed']     = 0.0
    traj['last_step_t'] = time.time()
    traj['duration']    = duration
    traj['waiting']     = False


def step_trajectory():
    """
    Called every main-loop iteration.

    State machine:
      EXECUTING (status=2) → task done → send status=1 immediately
        └─ more tasks? → delay → EXECUTING next (status=2)
        └─ no more?    → IDLE (status=0), unlock teach mode
    """
    if not traj['running'] or traj['stop']:
        return

    # ── Inter-task delay ──
    if traj['waiting']:
        if time.time() >= traj['delay_until']:
            traj['waiting'] = False
            _load_next_task()
            bridge.publish_status(STATUS_EXECUTING)   # status=2: next task starts
        return

    # ── Advance virtual elapsed time (respects safety speed multiplier) ──
    # Warning (1) → speed_mult=0.5 → each real second counts as 0.5 virtual sec
    # E-Stop  (2) → already halted by handle_stop(); never reaches here
    now        = time.time()
    speed_mult = _SAFETY_SPEED_MULT.get(sim_safety_status['v'], 1.0)
    traj['elapsed']    += (now - traj['last_step_t']) * speed_mult
    traj['last_step_t'] = now
    t = min(traj['elapsed'] / traj['duration'], 1.0)

    q_interp = traj['q_start'] + t * (traj['q_end'] - traj['q_start'])
    robot.q  = q_interp

    sim_rail['v']    = traj['rail_start'] + t * (traj['rail_end'] - traj['rail_start'])
    sim_gripper['v'] = traj['grip_start'] + t * (traj['grip_end'] - traj['grip_start'])

    q_deg = np.rad2deg(q_interp)
    for i, s in enumerate(joint_sliders):
        s.value = float(np.clip(q_deg[i], s.min, s.max))
    update_ik_sliders_from_fk()

    # ── Task complete ──
    if t >= 1.0:
        delay_ms = float(traj['tasks'][traj['idx']].get('delay', 0))
        traj['idx'] += 1

        # ─ CRITICAL: send status=1 immediately (must arrive < 600 ms) ─
        bridge.publish_status(STATUS_TASK_DONE)

        if traj['idx'] >= len(traj['tasks']):
            # All tasks done → IDLE
            traj['running'] = False
            teach_mode_active['v'] = True
            bridge.publish_status(STATUS_IDLE)
        else:
            # More tasks → wait delay, then EXECUTING again
            traj['delay_until'] = time.time() + delay_ms / 1000.0
            traj['waiting']     = True
            # status=2 will be sent after delay in the 'waiting' branch above


# ============================================================
# Bridge callback handlers
# ============================================================

def handle_execute_trajectory(tasks: list):
    """Start full trajectory (sorted by sequence)."""
    traj['tasks']   = tasks
    traj['idx']     = 0
    traj['running'] = True
    traj['stop']    = False
    traj['waiting'] = False
    teach_mode_active['v'] = False
    _load_next_task()
    bridge.publish_status(STATUS_EXECUTING)   # status=2: trajectory started


def handle_goto_position(task: dict):
    """Move to a single position (DryRun / Test)."""
    traj['tasks']   = [task]
    traj['idx']     = 0
    traj['running'] = True
    traj['stop']    = False
    traj['waiting'] = False
    teach_mode_active['v'] = False
    _load_next_task()
    bridge.publish_status(STATUS_EXECUTING)   # status=2


def handle_stop():
    """Emergency / user stop."""
    traj['stop']    = True
    traj['running'] = False
    traj['waiting'] = False
    teach_mode_active['v'] = True
    bridge.publish_status(STATUS_IDLE)        # status=0: stopped


def handle_teach_mode(enabled: bool):
    teach_mode_active['v'] = enabled


# ============================================================
# Safety status helpers
# ============================================================

_SAFETY_LABELS     = {0: "Normal", 1: "Warning", 2: "Emergency Stop"}
_SAFETY_SPEED_MULT = {0: 1.0, 1: 0.5, 2: 0.0}   # fraction of nominal speed

# Robot color per safety state (RGBA).  None = restore URDF original.
_SAFETY_COLORS = {
    0: None,                              # Normal  → original URDF colors
    1: (1.0, 0.584, 0.0,  1.0),          # Warning → orange
    2: (1.0, 0.231, 0.188, 1.0),         # E-Stop  → red
}


def _apply_safety_colors(status: int):
    """Tint robot geometry to reflect the current safety state."""
    color = _SAFETY_COLORS.get(status)
    if color is None:
        # Restore original per-link URDF colors
        for link in robot.links:
            for i, shape in enumerate(link.geometry):
                shape.color = original_colors[link.name][i]
    else:
        for link in robot.links:
            for shape in link.geometry:
                shape.color = color


def apply_safety_status(status: int):
    """
    Apply a safety status change from the simulator UI or an incoming WS command.

    Effects:
      0 Normal        — restore colors, re-enable teach mode
      1 Warning       — orange tint, trajectory continues
      2 Emergency Stop — red tint, halt trajectory immediately
    Publishes /safety_status to rosbridge so the web service (Next.js / iOS app)
    receives the updated state.
    """
    status = max(0, min(2, int(status)))   # clamp to valid range
    sim_safety_status['v'] = status

    _apply_safety_colors(status)

    if status == 2:
        # Halt trajectory — same effect as pressing Stop on the UI
        handle_stop()
    elif status == 0 and not traj['running']:
        # Return to Normal: re-enable manual control
        teach_mode_active['v'] = True

    # Publish to rosbridge → web service (Next.js / SafetyPanel iOS)
    bridge.publish("/safety_status", {"data": status})
    print(f"\n[Safety] {status} — {_SAFETY_LABELS[status]}")


def handle_safety_status(status: int):
    """Called when /safety_status arrives from the web service."""
    apply_safety_status(status)
    # Sync the Swift UI dropdown to match the incoming command
    safety_select.value = status


# Wire callbacks into bridge
bridge.on_execute_trajectory = handle_execute_trajectory
bridge.on_goto_position      = handle_goto_position
bridge.on_stop               = handle_stop
bridge.on_teach_mode         = handle_teach_mode
bridge.on_safety_status      = handle_safety_status


# ============================================================
# Swift UI
# ============================================================

# ── JOINT CONTROL ─────────────────────────────────────────────
env.add(Label("-- JOINT CONTROL (FK) --"))

joint_names = ["J1 (Z)", "J2 (X)", "J3 (-X)", "J4 (Y)", "J5 (X)", "J6 (-Y)"]
for i in range(robot.n):
    s = Slider(
        make_joint_cb(i),
        min=round(np.rad2deg(robot.qlim[0, i]), 1),
        max=round(np.rad2deg(robot.qlim[1, i]), 1),
        step=1, value=0,
        desc=joint_names[i], unit="&#176;"
    )
    joint_sliders.append(s)
    env.add(s)

# ── IK TARGET ─────────────────────────────────────────────────
env.add(Label("-- IK TARGET (Tip Point) --"))

ik_defs = [
    ('x',  "Tip X",        -400, 400, 5, "mm"),
    ('y',  "Tip Y",        -500, 100, 5, "mm"),
    ('z',  "Tip Z",          50, 700, 5, "mm"),
    ('rx', "Tip Rx Roll",  -180, 180, 5, "&#176;"),
    ('ry', "Tip Ry Pitch", -180, 180, 5, "&#176;"),
    ('rz', "Tip Rz Yaw",   -180, 180, 5, "&#176;"),
]
for key, desc, mn, mx, step, unit in ik_defs:
    s = Slider(
        make_target_cb(key),
        min=mn, max=mx, step=step,
        value=round(float(np.clip(target[key], mn, mx))),
        desc=desc, unit=unit
    )
    ik_sliders[key] = s
    env.add(s)

# ── ACTION SELECT ─────────────────────────────────────────────
def action_cb(idx):
    idx = int(idx)
    if idx == 1:    # Reset to Home
        robot.q = q_zero.copy()
        for s in joint_sliders:
            s.value = 0.0
        T_tip0 = robot.fkine(q_zero) * TIP_OFFSET
        tp0    = T_tip0.t * 1000
        rpy0   = np.rad2deg(T_tip0.rpy())
        for key, val, mn, mx in [
            ('x',  tp0[0],  -400, 400), ('y',  tp0[1],  -500, 100), ('z',  tp0[2],   50, 700),
            ('rx', rpy0[0], -180, 180), ('ry', rpy0[1], -180, 180), ('rz', rpy0[2], -180, 180),
        ]:
            clipped = float(np.clip(val, mn, mx))
            target[key] = clipped
            ik_sliders[key].value = clipped
        ik_status['ok'] = True
    elif idx == 2:  # Save Position
        count = save_position()
        lbl_save.desc = f"Saved {count} pos → saved_positions.json"
    action_select.value = 0

action_select = Select(action_cb, desc="Action",
                       options=["-- คำสั่ง --", "Reset to Home", "Save Position"], value=0)
env.add(action_select)

# ── APPEARANCE ────────────────────────────────────────────────
PRESET_COLORS = [
    None,
    (0.85, 0.15, 0.15, 1.0),
    (0.15, 0.35, 0.85, 1.0),
    (0.15, 0.75, 0.25, 1.0),
    (1.0,  0.50, 0.0,  1.0),
    (0.75, 0.75, 0.80, 1.0),
    (1.0,  0.80, 0.0,  1.0),
    "rainbow",
]

def change_robot_color(idx):
    chosen = PRESET_COLORS[int(idx)]
    if chosen is None:
        for link in robot.links:
            for i, shape in enumerate(link.geometry):
                shape.color = original_colors[link.name][i]
    elif chosen == "rainbow":
        cmap = robot.linkcolormap("viridis")
        n = max(len(robot.links) - 1, 1)
        for j, link in enumerate(robot.links):
            c = cmap(j / n)
            for shape in link.geometry:
                shape.color = c
    else:
        for link in robot.links:
            for shape in link.geometry:
                shape.color = chosen

env.add(Label("-- APPEARANCE --"))
color_select = Select(
    change_robot_color,
    desc="Robot Color",
    options=["Default (URDF)", "Red", "Blue", "Green", "Orange", "Silver", "Gold", "Rainbow"],
    value=0,
)
env.add(color_select)

# ── SAFETY CONTROL ────────────────────────────────────────────
env.add(Label("-- SAFETY CONTROL --"))

def safety_select_cb(idx):
    """
    Called when the operator changes the safety dropdown in the Swift viewer.
    Applies the state locally and publishes to the web service via rosbridge.
    """
    apply_safety_status(int(idx))

# The dropdown stays on the selected value — it always shows the active state.
safety_select = Select(
    safety_select_cb,
    desc="Safety Status",
    options=[
        "0  Normal  (reset)",
        "1  Warning  (reduced speed)",
        "2  Emergency Stop",
    ],
    value=0,
)
env.add(safety_select)

lbl_safety = Label("Safety: Normal")
env.add(lbl_safety)

# ── PARAMETERS ────────────────────────────────────────────────
env.add(Label("-- PARAMETERS (real-time) --"))

lbl_q    = Label("q: –")
lbl_ee   = Label("EE: –")
lbl_tip  = Label("Tip: –")
lbl_tgt  = Label("Target: –")
lbl_stat = Label("IK: –")
lbl_save = Label("Saved: 0 pos")
lbl_ws   = Label("rosbridge: connecting...")
for lbl in [lbl_q, lbl_ee, lbl_tip, lbl_tgt, lbl_stat, lbl_save, lbl_ws]:
    env.add(lbl)

# ============================================================
# Start WebSocket (after UI setup so callbacks can touch sliders)
# ============================================================
bridge.start()

_t0 = time.time()
while not bridge.connected and time.time() - _t0 < 3.0:
    time.sleep(0.1)
if not bridge.connected:
    print(f"[WS] WARNING: ไม่สามารถ connect {WS_URL} — รัน Swift ต่อโดยไม่มี rosbridge")

# ============================================================
# Main loop
# ============================================================
print(f"\n{'Tip X':>10} {'Tip Y':>10} {'Tip Z':>10}  mm")
print("-" * 36)

while True:
    env.step(0.05)
    now = time.time()

    # ── Dispatch WS commands → callback handlers ──
    bridge.drain()

    # ── Trajectory interpolation ──
    step_trajectory()

    # ── Deferred IK (skip while trajectory is running) ──
    if not traj['running']:
        if ik_dirty['flag'] and now > fk_lock['until']:
            if now - ik_last_solve['t'] >= IK_MIN_INTERVAL:
                ik_dirty['flag'] = False
                ik_last_solve['t'] = now
                solve_ik()

    # ── FK / display ──
    T_ee  = robot.fkine(robot.q)
    T_tip = T_ee * TIP_OFFSET
    tip_axes.T = T_tip

    q_deg = np.rad2deg(robot.q)
    ep    = T_ee.t * 1000
    tp    = T_tip.t * 1000

    lbl_q.desc    = f"q(deg): J1={q_deg[0]:+.1f} J2={q_deg[1]:+.1f} J3={q_deg[2]:+.1f} J4={q_deg[3]:+.1f} J5={q_deg[4]:+.1f} J6={q_deg[5]:+.1f}"
    lbl_ee.desc   = f"EE  (mm): X={ep[0]:+.1f}  Y={ep[1]:+.1f}  Z={ep[2]:+.1f}"
    lbl_tip.desc  = f"Tip (mm): X={tp[0]:+.1f}  Y={tp[1]:+.1f}  Z={tp[2]:+.1f}"
    lbl_tgt.desc  = f"Target: X={target['x']:+.0f} Y={target['y']:+.0f} Z={target['z']:+.0f}mm  Rx={target['rx']:+.0f} Ry={target['ry']:+.0f} Rz={target['rz']:+.0f}deg"
    lbl_stat.desc   = f"IK: {'OK' if ik_status['ok'] else 'FAIL (out of reach)'}"
    lbl_ws.desc     = f"rosbridge: {'connected' if bridge.is_connected else 'disconnected'}"
    lbl_safety.desc = f"Safety: {_SAFETY_LABELS[sim_safety_status['v']]}"

    # ── Publish joint states every 100 ms ──
    if now - last_js_pub_t['t'] >= JS_PUBLISH_INTERVAL:
        last_js_pub_t['t'] = now
        bridge.publish_joint_states(
            q_deg=list(q_deg.astype(float)),
            rail=sim_rail['v'],
            gripper=sim_gripper['v'],
        )

    print(f"\r{tp[0]:>+10.1f} {tp[1]:>+10.1f} {tp[2]:>+10.1f}  mm", end="", flush=True)
