#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — 3D-отрисовщик руки + MQTT-сервер (эмулятор манипулятора)
для управления через библиотеку pm_python_sdk.

Программа делает две вещи в одном процессе:
  1. Поднимает MQTT-брокер (amqtt) на порту 1883 и отвечает на команды
     так, как это делает настоящий манипулятор Promobot:
       /management     <- {"get_management": true, "id": N}
       /management     -> {"management_client_id": "<id клиента>", ...}
       /command        <- {"action":"execute","id":N,"command":"move_joints","data":{...}}
       /command_result -> {"id":N,"result":true,"data":{...}}
       /joint_states   -> {"<joint>": {"position": рад, "velocity": рад/с}, ...}  (10 Гц)
       /coordinates    -> {"position":{x,y,z},"orientation":{x,y,z,w},...}       (10 Гц)
       /stream         <- {"stream":"joint"|"pose"|"twist","data":{...}}
  2. Рисует 3D-модель руки (URDF + STL) в окне pyglet/pyrender и плавно
     анимирует суставы к заданным целевым углам (имитация движения).

Запуск (окружение .venv-app, пакеты ставятся из requirements.txt):
    python -m venv .venv-app
    .venv-app\\Scripts\\python.exe -m pip install -r requirements.txt
    .venv-app\\Scripts\\python.exe main.py            # окно + сервер
    .venv-app\\Scripts\\python.exe main.py --headless # только сервер, без окна

Клиент (папка client/, отдельное окружение client\\.venv с pm_python_sdk):
    client\\.venv\\Scripts\\python.exe client\\test_client.py
    from sdk.manipulators.medu import MEdu
    m = MEdu('127.0.0.1', '12', 'login', 'password')
    m.connect()
    m.get_control()
    m.move_to_angles(0.5, -0.7, -0.7)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import threading
import time
import xml.etree.ElementTree as ET

BASE_DIR = pathlib.Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

import numpy as np
import trimesh
import yourdfpy
from scipy.optimize import least_squares

import paho.mqtt.client as mqtt
from amqtt.broker import Broker
from amqtt.contexts import default_broker_plugins

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

MODEL_FILES = {
    "medu": ASSETS_DIR / "meDu.urdf",
    "m13": ASSETS_DIR / "m13.urdf",
}

STATE_RATE_HZ = 10.0        # частота публикации /joint_states и /coordinates
RENDER_FPS = 60.0           # частота кадров окна
JOINT_EPS = 2e-3            # радиан, допуск «сустав доехал»
BASE_JOINT_SPEED = 3.0      # рад/с при velocity_factor = 1.0
MIN_JOINT_SPEED = 0.05      # рад/с, чтобы движение всегда было заметно
COMMAND_TIMEOUT = 55.0      # сек, принудительный ответ по команде (у SDK таймаут 60 с)

TCP_CANDIDATES = ["flange", "tool0", "tool", "tcp", "ee_link", "link6"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Математика
# ---------------------------------------------------------------------------

def rpy_to_matrix(rpy) -> np.ndarray:
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    fwd = target - eye
    norm = np.linalg.norm(fwd)
    fwd = fwd / norm if norm > 1e-12 else np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    n = np.linalg.norm(right)
    right = right / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
    upv = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = upv
    T[:3, 2] = -fwd
    T[:3, 3] = eye
    return T


def quat_to_matrix(x, y, z, w) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> tuple:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def rotation_vector(R) -> np.ndarray:
    """Вектор-угол (ось * угол) матрицы поворота — ошибка ориентации."""
    cos_a = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(cos_a)
    if ang < 1e-9:
        return np.zeros(3)
    if abs(ang - math.pi) < 1e-6:
        # вырожденный случай: берём симметричную часть
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return w * ang


# ---------------------------------------------------------------------------
# Модель руки: URDF + меш + симуляция движения + КЗ/ОДК
# ---------------------------------------------------------------------------

class ArmModel:
    """Хранит текущие углы суставов, целевые углы и умеет считать кинематику."""

    # линки-«домик» рабочей зоны: закрывают руку, по умолчанию не рисуем
    HIDDEN_LINKS = {"workspace_shell"}

    def __init__(self, urdf_path: pathlib.Path, mesh_root: pathlib.Path,
                 hidden_links: set | None = None):
        self.urdf_path = pathlib.Path(urdf_path)
        self.mesh_root = pathlib.Path(mesh_root)
        self.hidden_links = set(self.HIDDEN_LINKS if hidden_links is None else hidden_links)
        self.lock = threading.RLock()

        def filename_handler(fname, mesh_dir=None, **kwargs):
            f = fname
            if f.startswith("package://"):
                parts = f[len("package://"):].split("/", 1)
                f = parts[1] if len(parts) > 1 else ""
            return str(self.mesh_root / f.replace("/", os.sep))

        self.robot = yourdfpy.URDF.load(str(self.urdf_path), filename_handler=filename_handler)
        self.base_link = self.robot.base_link
        self.actuated = list(self.robot.actuated_joint_names)

        # лимиты суставов из URDF
        self.limits = {}
        for name in self.actuated:
            joint = self.robot.joint_map[name]
            if joint.limit is not None and joint.type != "continuous":
                self.limits[name] = (float(joint.limit.lower), float(joint.limit.upper))
            else:
                self.limits[name] = (-math.pi, math.pi)

        self.pos = {n: 0.0 for n in self.actuated}     # текущие углы, рад
        self.target = dict(self.pos)                   # целевые углы, рад
        self.vel = {n: 0.0 for n in self.actuated}     # текущие скорости, рад/с
        self.cmd_vel = {}                              # заданные скорости, рад/с
        self.velocity_factor = 0.1
        self.acceleration_factor = 0.1
        self.version = 0                               # счётчик изменения целей

        # состояние «железа», которое не влияет на геометрию
        self.gripper = {"rotation": 0, "gripper": 0}
        self.nozzle_power = False
        self.vacuum_power = False

        # --- цепь до TCP и суставы для ОДК ---
        child_to_joint = {}
        for jn, j in self.robot.joint_map.items():
            child_to_joint[j.child] = jn
        self.tcp_link = next((c for c in TCP_CANDIDATES if c in self.robot.link_map), self.base_link)

        chain = []
        cur = self.tcp_link
        seen = set()
        while cur in child_to_joint and cur not in seen:
            seen.add(cur)
            jn = child_to_joint[cur]
            chain.append(jn)
            cur = self.robot.joint_map[jn].parent
        chain.reverse()
        # для позиционной ОДК берём только revolute-суставы цепи
        self.ik_joints = [j for j in chain if self.robot.joint_map[j].type == "revolute"]

        # --- визуальные меша ---
        self.actors = self._load_visuals()

    # -- загрузка геометрии -------------------------------------------------
    def _load_visuals(self):
        root = ET.fromstring(self.urdf_path.read_text(encoding="utf-8"))
        from pyrender import Mesh, Node  # импорт здесь: оконные зависимости нужны только для рендера

        mesh_cache = {}
        actors = []
        for link in root.findall("link"):
            link_name = link.get("name")
            if link_name in self.hidden_links:
                continue
            for vis in link.findall("visual"):
                mesh_el = vis.find("geometry/mesh")
                if mesh_el is None:
                    continue
                fname = mesh_el.get("filename")
                if fname.startswith("package://"):
                    parts = fname[len("package://"):].split("/", 1)
                    fname = parts[1] if len(parts) > 1 else ""
                path = str(self.mesh_root / fname.replace("/", os.sep))
                if not os.path.exists(path):
                    log(f"ВНИМАНИЕ: не найден меш {path}")
                    continue
                if path not in mesh_cache:
                    m = trimesh.load(path, force="mesh")
                    scale = np.array([float(v) for v in (mesh_el.get("scale") or "1 1 1").split()])
                    m.apply_scale(scale)
                    mesh_cache[path] = m
                org = vis.find("origin")
                xyz = [float(v) for v in (org.get("xyz") if org is not None else "0 0 0").split()]
                rpy = [float(v) for v in (org.get("rpy") if org is not None else "0 0 0").split()]
                T = np.eye(4)
                T[:3, :3] = rpy_to_matrix(rpy)
                T[:3, 3] = xyz
                actors.append({
                    "node": Node(mesh=Mesh.from_trimesh(mesh_cache[path], smooth=True)),
                    "link": link_name,
                    "origin": T,
                })
        return actors

    def view_bounds(self) -> tuple:
        """Габариты руки в нулевой позе: (центр, радиус) в метрах."""
        with self.lock:
            self.robot.update_cfg(self.cfg_array())
            pts = []
            for actor in self.actors:
                T = self.robot.get_transform(actor["link"], self.base_link) @ actor["origin"]
                lo, hi = actor["node"].mesh.primitives[0].bounds
                grid = np.array([[box[i, 0], box[j, 1], box[k, 2]]
                                 for box in (np.vstack([lo, hi]),)
                                 for i in range(2) for j in range(2) for k in range(2)])
                pts.append((T[:3, :3] @ grid.T).T + T[:3, 3])
        all_pts = np.vstack(pts)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo) / 2.0)
        return center, max(radius, 0.1)

    # -- доступ к состоянию --------------------------------------------------
    def cfg_array(self, overrides: dict | None = None) -> np.ndarray:
        vals = []
        for n in self.actuated:
            if overrides and n in overrides:
                vals.append(overrides[n])
            else:
                vals.append(self.pos[n])
        return np.array(vals, dtype=float)

    def clamp(self, name: str, value: float) -> float:
        lo, hi = self.limits.get(name, (-math.pi, math.pi))
        return max(lo, min(hi, value))

    def set_targets(self, positions: dict, velocities: dict | None = None,
                    velocity_factor: float | None = None) -> list:
        """Ставит целевые углы. Возвращает список суставов, которых коснулось изменение."""
        touched = []
        if velocity_factor is not None:
            self.velocity_factor = float(velocity_factor)
        with self.lock:
            for name, value in positions.items():
                if name not in self.pos:
                    continue
                self.target[name] = self.clamp(name, float(value))
                touched.append(name)
            if velocities:
                for name, value in velocities.items():
                    if name in self.pos:
                        self.cmd_vel[name] = float(value)
            if touched:
                self.version += 1
        return touched

    def stop(self) -> None:
        with self.lock:
            for n in self.actuated:
                self.target[n] = self.pos[n]
                self.vel[n] = 0.0
            self.version += 1

    def zero(self) -> None:
        with self.lock:
            for n in self.actuated:
                self.pos[n] = 0.0
                self.target[n] = 0.0
                self.vel[n] = 0.0
            self.version += 1

    def _speed(self, name: str) -> float:
        """Скорость сустава, рад/с.

        Явно заданная скорость (из данных команды/стрима) — абсолютная,
        а общая velocity_factor применяется только к скорости по умолчанию.
        """
        v = self.cmd_vel.get(name, 0.0)
        if v > 0:
            return max(MIN_JOINT_SPEED, v)
        return max(MIN_JOINT_SPEED, BASE_JOINT_SPEED * max(0.01, self.velocity_factor))

    def step(self, dt: float) -> bool:
        """Один шаг симуляции. Возвращает True, если всё движение завершено."""
        moving = False
        with self.lock:
            for n in self.actuated:
                cur, tgt = self.pos[n], self.target[n]
                delta = tgt - cur
                if abs(delta) > JOINT_EPS:
                    w = self._speed(n)
                    advance = min(w * dt, abs(delta))
                    self.pos[n] = cur + math.copysign(advance, delta)
                    self.vel[n] = math.copysign(w, delta)
                    moving = True
                else:
                    if abs(cur - tgt) > 1e-12:
                        self.pos[n] = tgt
                    self.vel[n] = 0.0
        return not moving

    # -- кинематика ----------------------------------------------------------
    def fk(self, link: str | None = None) -> np.ndarray:
        link = link or self.tcp_link
        with self.lock:
            self.robot.update_cfg(self.cfg_array())
            return self.robot.get_transform(link, self.base_link)

    def solve_ik(self, xyz, quat=None) -> dict | None:
        """Численная ОДК: возвращает углы суставов или None, если точка недостижима."""
        xyz = np.asarray(xyz, dtype=float)
        R_target = quat_to_matrix(*quat) if quat is not None else None
        joints = self.ik_joints
        if not joints:
            return None

        with self.lock:
            base_cfg = self.cfg_array()
            current = {n: self.pos[n] for n in self.actuated}
        idx = [self.actuated.index(j) for j in joints]
        x0 = np.array([current[j] for j in joints])
        lo = np.array([self.limits[j][0] for j in joints])
        hi = np.array([self.limits[j][1] for j in joints])
        use_orientation = R_target is not None and len(joints) >= 6

        def residual(q):
            cfg = base_cfg.copy()
            for k, i in enumerate(idx):
                cfg[i] = q[k]
            with self.lock:
                self.robot.update_cfg(cfg)
                T = self.robot.get_transform(self.tcp_link, self.base_link)
            res = list(T[:3, 3] - xyz)
            if use_orientation:
                res.extend(rotation_vector(R_target @ T[:3, :3].T))
            return np.array(res)

        try:
            sol = least_squares(residual, x0, bounds=(lo, hi),
                                xtol=1e-10, ftol=1e-10, max_nfev=400)
        except Exception as exc:  # noqa: BLE001
            log(f"ОДК: ошибка вычисления ({exc})")
            return None
        finally:
            with self.lock:
                self.robot.update_cfg(self.cfg_array())

        err = residual(sol.x)
        pos_err = float(np.linalg.norm(err[:3]))
        ori_err = float(np.linalg.norm(err[3:])) if len(err) > 3 else 0.0
        if pos_err > 0.01 or ori_err > 0.1:
            return None
        return {j: float(v) for j, v in zip(joints, sol.x)}

    # -- публикуемые состояния ------------------------------------------------
    def joint_states_payload(self) -> dict:
        with self.lock:
            return {n: {"position": round(self.pos[n], 6), "velocity": round(self.vel[n], 6)}
                    for n in self.actuated}

    def coordinates_payload(self) -> dict:
        T = self.fk()
        x, y, z = (float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        qx, qy, qz, qw = matrix_to_quat(T[:3, :3])
        return {
            "frame_id": "base_link",
            "position": {"x": x, "y": y, "z": z},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "euler_angles": {"rx": 0.0, "ry": 0.0, "rz": 0.0},
        }


# ---------------------------------------------------------------------------
# Брокер: запоминает, какой клиент опубликовал сообщение
# ---------------------------------------------------------------------------

class RobotBroker(Broker):
    """amqtt Broker, который знает MQTT client_id автора каждого сообщения."""

    def __init__(self, config=None, loop=None):
        super().__init__(config=config, loop=loop)
        self.last_publisher: dict = {}
        self.robot_client_id: str | None = None

    async def _handle_message_delivery(self, client_session, handler, wait_deliver):
        try:
            app_message = wait_deliver.result()
            topic = getattr(app_message, "topic", None)
            if topic and client_session.client_id != self.robot_client_id:
                self.last_publisher[topic] = client_session.client_id
        except Exception:  # noqa: BLE001
            pass
        return await super()._handle_message_delivery(client_session, handler, wait_deliver)


def run_broker(holder: dict, bind_host: str, port: int, ready: threading.Event) -> None:
    """Поднимает брокер в отдельном потоке с собственным asyncio-циклом."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    config = {
        "listeners": {"default": {"type": "tcp", "bind": f"{bind_host}:{port}"}},
        "plugins": default_broker_plugins(),
    }
    broker = RobotBroker(config=config, loop=loop)
    holder["broker"] = broker
    loop.run_until_complete(broker.start())
    ready.set()
    log(f"MQTT-брокер запущен на {bind_host}:{port}")
    try:
        loop.run_forever()
    finally:
        try:
            loop.run_until_complete(broker.shutdown())
        except Exception:  # noqa: BLE001
            pass
        loop.close()


# ---------------------------------------------------------------------------
# Сервер: обработка MQTT-команд от pm_python_sdk
# ---------------------------------------------------------------------------

class RobotServer:
    def __init__(self, broker: RobotBroker, model: ArmModel, host: str, port: int,
                 title: str = "MEdu"):
        self.broker = broker
        self.model = model
        self.host = host
        self.port = port
        self.title = title
        self.client_id = "medu-server"
        self.pending: list = []
        self.connected_clients: set = set()
        self._lock = threading.Lock()

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=self.client_id, protocol=mqtt.MQTTv311)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.username_pw_set("server", "server")

    # -- жизненный цикл -------------------------------------------------------
    def start(self) -> None:
        self.broker.robot_client_id = self.client_id
        self.client.connect(self.host, self.port, 60)
        self.client.loop_start()
        for topic in ("/command", "/command_result", "/management", "/stream"):
            self.client.subscribe(topic)
        log(f"Сервер-манипулятор подключен к брокеру как '{self.client_id}'")

    def stop(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:  # noqa: BLE001
            pass

    # -- callbacks paho --------------------------------------------------------
    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        log(f"MQTT connect rc={reason_code}")

    def _on_disconnect(self, client, userdata, *args, **kwargs):
        log("MQTT disconnect")

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage):
        try:
            payload = msg.payload.decode("utf-8")
            data = json.loads(payload)
        except Exception:  # noqa: BLE001
            log(f"Некорректный JSON в {msg.topic}: {msg.payload[:200]!r}")
            return
        try:
            if msg.topic == "/command":
                self._handle_command(data)
            elif msg.topic == "/management":
                self._handle_management(data)
            elif msg.topic == "/stream":
                self._handle_stream(data)
        except Exception as exc:  # noqa: BLE001
            log(f"Ошибка обработки {msg.topic}: {exc}")
            import traceback
            traceback.print_exc()

    # -- /management ------------------------------------------------------------
    def _handle_management(self, data: dict) -> None:
        if "management_client_id" in data:
            return  # это наш собственный ответ
        requester = self.broker.last_publisher.get("/management")
        cmd_id = data.get("id")
        log(f"Запрос управления от клиента '{requester}' (id={cmd_id})")
        reply = {
            "get_management": bool(data.get("get_management", True)),
            "id": cmd_id,
            "management_client_id": requester,
        }
        # сначала /command_result (на случай, если ответ управления не дойдёт),
        # затем сам ответ в /management — его ждёт GetManageCommand
        self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
        self._publish("/management", reply)
        if requester:
            self.connected_clients.add(requester)

    # -- /command ----------------------------------------------------------------
    def _handle_command(self, data: dict) -> None:
        action = data.get("action", "execute")
        cmd_id = data.get("id")
        command = data.get("command", "")
        body = data.get("data")
        if not isinstance(body, dict):   # SDK присылает строку "None", если данных нет
            body = {}

        if action == "cancel":
            log(f"Отмена команды id={cmd_id} ({command})")
            self.model.stop()
            self._finish_pending()
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
            return

        if action != "execute":
            log(f"Неизвестное действие '{action}' (id={cmd_id})")
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
            return

        log(f"Команда {command} id={cmd_id}: {json.dumps(body, ensure_ascii=False)}")

        if command == "move_joints":
            positions = body.get("positions")
            velocities = body.get("velocities")
            if not isinstance(positions, dict):
                positions = {}
            if not isinstance(velocities, dict):
                velocities = {}
            touched = self.model.set_targets(
                positions, velocities, body.get("velocity_factor"))
            if body.get("acceleration_factor") is not None:
                self.model.acceleration_factor = float(body["acceleration_factor"])
            if not touched:
                self._publish("/command_result",
                              {"id": cmd_id, "result": False,
                               "data": {"error": "Неизвестные суставы: "
                                                 + ", ".join(positions)}})
                return
            self._add_pending(cmd_id, command, touched)
            return

        if command == "set_coordinates":
            pos = body.get("position")
            orient = body.get("orientation")
            euler = body.get("euler_angles")
            if not isinstance(pos, dict):
                pos = {}
            if not isinstance(orient, dict):
                orient = {}
            if not isinstance(euler, dict):
                euler = {}
            target_xyz = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]
            quat = None
            if body.get("use_euler"):
                rx, ry, rz = float(euler.get("rx", 0)), float(euler.get("ry", 0)), float(euler.get("rz", 0))
                quat = euler_to_quat(rx, ry, rz)
            else:
                quat = (float(orient.get("x", 0.0)), float(orient.get("y", 0.0)),
                        float(orient.get("z", 0.0)), float(orient.get("w", 1.0)))
            sol = self.model.solve_ik(target_xyz, quat)
            if sol is None:
                log(f"ОДК: точка {target_xyz} недостижима")
                self._publish("/command_result",
                              {"id": cmd_id, "result": False,
                               "data": {"error": f"Точка {target_xyz} вне рабочей зоны"}})
                return
            if body.get("velocity_factor") is not None:
                self.model.velocity_factor = float(body["velocity_factor"])
            touched = self.model.set_targets(sol)
            self._add_pending(cmd_id, command, touched)
            return

        if command == "gripper_control":
            if body.get("rotation") is not None:
                self._apply_percent("povorot_instrumenta", float(body["rotation"]))
                self.model.gripper["rotation"] = int(body["rotation"])
            if body.get("gripper") is not None:
                self._apply_percent("servo_zahvata", float(body["gripper"]))
                self.model.gripper["gripper"] = int(body["gripper"])
            self._add_pending(cmd_id, command,
                              [j for j in ("povorot_instrumenta", "servo_zahvata")
                               if j in self.model.pos])
            return

        if command == "vacuum_control":
            if body.get("rotation") is not None:
                self._apply_percent("povorot_instrumenta", float(body["rotation"]))
            self.model.vacuum_power = bool(body.get("power_supply", False))
            log(f"Вакуум: {self.model.vacuum_power}")
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
            return

        if command == "nozzle_power":
            self.model.nozzle_power = bool(body.get("power_supply", False))
            log(f"Питание насадки: {self.model.nozzle_power}")
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
            return

        if command == "stop_moving":
            self.model.stop()
            self._finish_pending()
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
            return

        if command == "set_zero":
            self.model.zero()
            self._finish_pending()
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})
            return

        if command == "get_current_errors":
            self._publish("/command_result",
                          {"id": cmd_id, "result": True, "data": {"errors": []}})
            return

        if command in ("get_i2c", "get_gpio", "get_gpio_bit", "get_gpio_mask"):
            self._publish("/command_result",
                          {"id": cmd_id, "result": True, "data": {"value": 0}})
            return

        # прочие команды: подтверждаем, чтобы SDK не ждал таймаута
        log(f"Команда {command} принята (эмуляция без побочного эффекта)")
        self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})

    def _apply_percent(self, joint: str, percent: float) -> None:
        """Значение в процентах (-100..100) переводим в угол по лимиту сустава."""
        if joint not in self.model.pos:
            return
        lo, hi = self.model.limits.get(joint, (-1.0, 1.0))
        p = max(-100.0, min(100.0, percent)) / 100.0
        angle = hi * p if p >= 0 else lo * p
        self.model.set_targets({joint: angle})

    # -- /stream ------------------------------------------------------------------
    def _handle_stream(self, data: dict) -> None:
        kind = data.get("stream")
        body = data.get("data")
        if not isinstance(body, dict):
            body = {}
        if kind == "joint":
            positions = body.get("positions")
            velocities = body.get("velocities")
            self.model.set_targets(positions if isinstance(positions, dict) else {},
                                   velocities if isinstance(velocities, dict) else {})
        elif kind == "pose":
            pos = body.get("position")
            o = body.get("orientation")
            if not isinstance(pos, dict):
                pos = {}
            if not isinstance(o, dict):
                o = {}
            target_xyz = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]
            if o.get("use_euler"):
                quat = euler_to_quat(float(o.get("rx", 0)), float(o.get("ry", 0)), float(o.get("rz", 0)))
            else:
                quat = (float(o.get("x", 0.0)), float(o.get("y", 0.0)),
                        float(o.get("z", 0.0)), float(o.get("w", 1.0)))
            sol = self.model.solve_ik(target_xyz, quat)
            if sol:
                self.model.set_targets(sol)
        elif kind == "twist":
            log("stream: twist (скорости по осям) — игнорируется в эмуляции")

    # -- служебное -----------------------------------------------------------------
    def _publish(self, topic: str, payload: dict) -> None:
        self.client.publish(topic, json.dumps(payload))

    def _add_pending(self, cmd_id, command: str, joints: list) -> None:
        with self._lock:
            # новая команда движения завершает предыдущие, чтобы SDK не висел
            self._finish_pending_locked()
            self.pending.append({
                "id": cmd_id,
                "command": command,
                "joints": set(joints),
                "version": self.model.version,
                "deadline": time.time() + COMMAND_TIMEOUT,
            })

    def _finish_pending(self) -> None:
        with self._lock:
            self._finish_pending_locked()

    def _finish_pending_locked(self) -> None:
        for cmd in self.pending:
            self._publish("/command_result", {"id": cmd["id"], "result": True, "data": {}})
        self.pending.clear()

    def update(self, dt: float) -> None:
        """Вызывается из главного цикла (окно или headless): двигает руку и
        отвечает на команды, дождавшиеся окончания движения."""
        version_at_start = self.model.version
        finished = self.model.step(dt)

        with self._lock:
            if not self.pending:
                return
            now = time.time()

            expired = [c for c in self.pending if now >= c["deadline"]]
            if expired:
                for cmd in expired:
                    log(f"Команда {cmd['command']} id={cmd['id']} завершена по таймауту")
                self._publish_results([c["id"] for c in expired])
                expired_ids = {c["id"] for c in expired}
                self.pending = [c for c in self.pending if c["id"] not in expired_ids]
                return

            if not finished:
                return

            # команды, появившиеся во время этого кадра, не завершаем раньше времени
            ready = [c for c in self.pending if c["version"] <= version_at_start]
            if ready:
                for cmd in ready:
                    log(f"Команда {cmd['command']} id={cmd['id']} выполнена")
                self._publish_results([c["id"] for c in ready])
                ready_ids = {c["id"] for c in ready}
                self.pending = [c for c in self.pending if c["id"] not in ready_ids]

    def _publish_results(self, ids) -> None:
        for cmd_id in ids:
            self._publish("/command_result", {"id": cmd_id, "result": True, "data": {}})


def euler_to_quat(rx: float, ry: float, rz: float) -> tuple:
    R = rpy_to_matrix((rx, ry, rz))
    return matrix_to_quat(R)


# ---------------------------------------------------------------------------
# Поток публикации состояния (/joint_states, /coordinates)
# ---------------------------------------------------------------------------

class StatePublisher(threading.Thread):
    def __init__(self, server: RobotServer):
        super().__init__(daemon=True, name="state-publisher")
        self.server = server
        self._stop = threading.Event()

    def run(self) -> None:
        period = 1.0 / STATE_RATE_HZ
        while not self._stop.is_set():
            start = time.time()
            try:
                self.server._publish("/joint_states", self.server.model.joint_states_payload())
                self.server._publish("/coordinates", self.server.model.coordinates_payload())
            except Exception as exc:  # noqa: BLE001
                log(f"Ошибка публикации состояния: {exc}")
            elapsed = time.time() - start
            self._stop.wait(max(0.0, period - elapsed))

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Окно с 3D-моделью
# ---------------------------------------------------------------------------

class Viewer:
    def __init__(self, model: ArmModel, server: RobotServer, title: str, msaa: bool = False):
        import pyglet
        from pyrender import Renderer, Scene, PerspectiveCamera, DirectionalLight, RenderFlags

        self.pyglet = pyglet
        self.RenderFlags = RenderFlags
        pyglet.options["shadow_window"] = False

        self.model = model
        self.server = server
        self.width, self.height = 1100, 760

        # орбитальная камера: центр и дистанция — по габаритам руки
        self.center, radius = model.view_bounds()
        self.yaw, self.pitch = math.radians(-35), math.radians(22)
        self.dist = max(0.25, radius * 2.6)
        self.home = (self.yaw, self.pitch, self.dist, self.center.copy())
        self._drag = None

        # перерисовываем только когда что-то изменилось (рука/камера/размер)
        self._need_draw = True
        self._last_state = None
        self._last_draw = 0.0

        self.scene = Scene(bg_color=np.array([0.10, 0.11, 0.13, 1.0]))
        for actor in model.actors:
            self.scene.add_node(actor["node"])

        # освещение
        self._add_light((0.5, 0.6, 1.0), 3.0)
        self._add_light((-0.6, -0.5, 0.7), 1.3)
        self._add_light((0.1, -0.4, -0.8), 0.6)

        # камера
        from pyrender import Node
        cam = PerspectiveCamera(yfov=0.85)
        cam_node = Node(camera=cam)
        self.scene.add_node(cam_node)
        self.camera_node = cam_node

        # MSAA по умолчанию выключен (сильная нагрузка на GPU); включается --msaa
        if msaa:
            config = pyglet.gl.Config(major_version=4, minor_version=1, depth_size=24,
                                      double_buffer=True, sample_buffers=1, samples=4)
        else:
            config = pyglet.gl.Config(major_version=4, minor_version=1, depth_size=24,
                                      double_buffer=True)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, visible=True, resizable=True,
            caption=title, config=config)
        self.renderer = Renderer(viewport_width=self.width, viewport_height=self.height)
        self._bind_events()
        self.frames = 0
        self._fps_time = time.time()
        self.fps = 0.0
        self._caption_time = 0.0

    def _add_light(self, eye, intensity) -> None:
        from pyrender import DirectionalLight, Node
        light = DirectionalLight(color=np.ones(3), intensity=intensity)
        self.scene.add_node(Node(light=light, matrix=look_at(eye, self.center)))

    def _bind_events(self) -> None:
        pyglet = self.pyglet
        window = self.window
        self._last_caption = 0.0

        @window.event
        def on_draw():
            # flip() НЕ вызываем: pyglet делает это сам в Window.draw()
            window.switch_to()
            self._update_camera()
            self.renderer.render(self.scene, self.RenderFlags.NONE)

        @window.event
        def on_resize(w, h):
            self.renderer.viewport_width = w
            self.renderer.viewport_height = h
            self._need_draw = True

        @window.event
        def on_expose():
            self._need_draw = True

        @window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            if buttons & pyglet.window.mouse.LEFT:
                self.yaw -= dx * 0.008
                self.pitch = max(-1.45, min(1.45, self.pitch + dy * 0.008))
                self._need_draw = True

        @window.event
        def on_mouse_scroll(x, y, sx, sy):
            self.dist = max(0.25, min(4.0, self.dist * (0.9 if sy > 0 else 1.1)))
            self._need_draw = True

        @window.event
        def on_key_press(symbol, modifiers):
            if symbol == pyglet.window.key.R:
                self.yaw, self.pitch, self.dist, self.center = (
                    self.home[0], self.home[1], self.home[2], self.home[3].copy())
                self._need_draw = True
            elif symbol == pyglet.window.key.ESCAPE:
                window.close()

        @window.event
        def on_close():
            pyglet.app.exit()

    def _update_camera(self) -> None:
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        offset = self.dist * np.array([cp * cy, cp * sy, sp])
        eye = self.center + offset
        self.scene.set_pose(self.camera_node, look_at(eye, self.center))

    def tick(self, dt: float) -> None:
        """Вызывается pyglet-часами: движение руки и перерисовка по необходимости."""
        self.server.update(dt)

        # если углы изменились — обновляем позы мешей и просим кадр
        with self.model.lock:
            state = tuple(round(self.model.pos[n], 5) for n in self.model.actuated)
        if state != self._last_state:
            self._last_state = state
            self._need_draw = True
            with self.model.lock:
                self.model.robot.update_cfg(self.model.cfg_array())
                for actor in self.model.actors:
                    try:
                        T_link = self.model.robot.get_transform(actor["link"], self.model.base_link)
                    except Exception:  # noqa: BLE001
                        T_link = np.eye(4)
                    self.scene.set_pose(actor["node"], T_link @ actor["origin"])

        if self._need_draw:
            self._need_draw = False
            self.window.draw(dt)   # сам делает switch_to / on_draw / flip
            self.frames += 1

        now = time.time()
        if now - self._fps_time >= 1.0:
            self.fps = self.frames / (now - self._fps_time)
            self.frames = 0
            self._fps_time = now
        if now - self._caption_time >= 0.5:
            self._caption_time = now
            self.window.set_caption(self._caption())

    def _caption(self) -> str:
        m = self.model
        with m.lock:
            a = m.pos.get("povorot_osnovaniya", 0.0)
            b = m.pos.get("privod_plecha", 0.0)
            c = m.pos.get("privod_strely", 0.0)
        return (f"{self.server.title} | MQTT :{self.server.port} | "
                f"клиентов: {len(self.server.connected_clients)} | "
                f"углы: {math.degrees(a):.1f}°, {math.degrees(b):.1f}°, {math.degrees(c):.1f}° | "
                f"{self.fps:.0f} FPS | ЛКМ — вращение, колесо — зум, R — сброс")

    def run(self) -> None:
        pyglet = self.pyglet
        pyglet.clock.schedule_interval(self.tick, 1.0 / RENDER_FPS)
        # interval=None: pyglet сам кадры не рисует, делаем это в tick по флагу
        pyglet.app.run(None)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="3D-модель руки + MQTT-сервер для pm_python_sdk")
    p.add_argument("--model", choices=sorted(MODEL_FILES), default="medu",
                   help="какую руку рисовать (по умолчанию medu)")
    p.add_argument("--bind", default="0.0.0.0", help="адрес бинднга MQTT-брокера")
    p.add_argument("--port", type=int, default=1883, help="порт MQTT-брокера")
    p.add_argument("--client-host", default="127.0.0.1",
                   help="куда подключается «серверная» сторона MQTT (обычно 127.0.0.1)")
    p.add_argument("--headless", action="store_true", help="без 3D-окна, только сервер")
    p.add_argument("--show-shell", action="store_true",
                   help="рисовать купол рабочей зоны (workspace_shell), по умолчанию скрыт")
    p.add_argument("--msaa", action="store_true",
                   help="включить сглаживание MSAA 4x (больше нагрузка на GPU)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    args = parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("amqtt").setLevel(logging.WARNING)

    urdf_path = MODEL_FILES[args.model]
    if not urdf_path.exists():
        log(f"Не найден URDF: {urdf_path}")
        return 1

    log(f"Загрузка модели {urdf_path.name} ...")
    hidden = set() if args.show_shell else None  # None -> HIDDEN_LINKS по умолчанию
    model = ArmModel(urdf_path, ASSETS_DIR, hidden_links=hidden)
    log(f"Суставы ({len(model.actuated)}): {', '.join(model.actuated)}")
    log(f"Суставы для ОДК: {', '.join(model.ik_joints)}")
    log(f"TCP: {model.tcp_link} | визуальных мешей: {len(model.actors)}")

    broker_holder: dict = {}
    ready = threading.Event()
    broker_thread = threading.Thread(
        target=run_broker, args=(broker_holder, args.bind, args.port, ready),
        daemon=True, name="mqtt-broker")
    broker_thread.start()
    if not ready.wait(timeout=15):
        log("Брокер не поднялся за 15 секунд")
        return 1
    broker = broker_holder["broker"]

    server = RobotServer(broker, model, args.client_host, args.port,
                         title=f"MEdu — сервер + 3D ({urdf_path.stem})")
    server.start()

    publisher = StatePublisher(server)
    publisher.start()

    title = server.title

    try:
        if args.headless:
            log("Режим headless: сервер работает без окна, Ctrl+C для выхода")
            last = time.time()
            while True:
                time.sleep(1.0 / RENDER_FPS)
                now = time.time()
                server.update(min(0.25, now - last))
                last = now
        else:
            viewer = Viewer(model, server, title, msaa=args.msaa)
            log("3D-окно открыто. Управлять можно из pm_python_sdk (host=127.0.0.1).")
            viewer.run()
    except KeyboardInterrupt:
        log("Останов по Ctrl+C")
    finally:
        publisher.stop()
        server.stop()
        log("Сервер остановлен")
    return 0


if __name__ == "__main__":
    sys.exit(main())
