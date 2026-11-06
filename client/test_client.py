"""Сквозной тест: pm_python_sdk (папка client/, окружение client\\.venv) -> main.py (окружение .venv-app).

Запуск из корня проекта:
    client\\.venv\\Scripts\\python.exe client\\test_client.py
(сервер main.py при этом должен быть запущен).
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')

from sdk.manipulators.medu import MEdu

HOST = "127.0.0.1"
CLIENT_ID = "test-client-1"
LOGIN = "user"
PASSWORD = "pass"

results = []


def check(name, ok, extra=""):
    results.append((name, bool(ok)))
    print(("OK   " if ok else "FAIL ") + name + (" | " + str(extra) if extra else ""), flush=True)


m = MEdu(HOST, CLIENT_ID, LOGIN, PASSWORD)

print("1) connect()", flush=True)
m.connect()
check("connect", True)

print("2) get_joint_state()", flush=True)
try:
    js = m.get_joint_state(timeout_seconds=10)
    check("get_joint_state", js is not None and "povorot_osnovaniya" in json.loads(js), js[:160])
except Exception as e:
    check("get_joint_state", False, e)

print("3) get_cartesian_coordinates()", flush=True)
try:
    cc = m.get_cartesian_coordinates(timeout_seconds=10)
    data = json.loads(cc)
    check("get_cartesian_coordinates", "position" in data, str(data)[:160])
except Exception as e:
    check("get_cartesian_coordinates", False, e)

print("4) get_control()", flush=True)
try:
    m.get_control(timeout_seconds=10)
    check("get_control", True)
except Exception as e:
    check("get_control", False, e)
    print("остальные проверки без управления невозможны", flush=True)

print("5) move_to_angles(0.5, -0.7, -0.7)", flush=True)
try:
    t0 = time.time()
    m.move_to_angles(0.5, -0.7, -0.7, timeout_seconds=45)
    check("move_to_angles", True, "%.1f сек" % (time.time() - t0))
except Exception as e:
    check("move_to_angles", False, e)

print("6) повторный get_joint_state() — должны увидеть новые углы", flush=True)
try:
    js = m.get_joint_state(timeout_seconds=10)
    data = json.loads(js)
    got = [round(data[j]["position"], 3) for j in ("povorot_osnovaniya", "privod_plecha", "privod_strely")]
    ok = all(abs(a - b) < 0.05 for a, b in zip(got, [0.5, -0.7, -0.7]))
    check("углы доехали", ok, got)
except Exception as e:
    check("углы доехали", False, e)

print("7) stream_joint_positions()", flush=True)
try:
    m.stream_joint_positions({"povorot_osnovaniya": -0.3, "privod_plecha": -1.0, "privod_strely": -1.0},
                             {"povorot_osnovaniya": 0.5, "privod_plecha": 0.5, "privod_strely": 0.5})
    time.sleep(4.0)
    js = m.get_joint_state(timeout_seconds=10)
    data = json.loads(js)
    got = round(data["povorot_osnovaniya"]["position"], 3)
    check("stream двигает руку", abs(got - (-0.3)) < 0.05, got)
except Exception as e:
    check("stream двигает руку", False, e)

print("8) move_to_coordinates(0.25, 0.0, 0.20)", flush=True)
try:
    from sdk.commands.move_coordinates_command import (
        MoveCoordinatesParamsPosition, MoveCoordinatesParamsOrientation)
    t0 = time.time()
    m.move_to_coordinates(MoveCoordinatesParamsPosition(0.25, 0.0, 0.20),
                          MoveCoordinatesParamsOrientation(0, 0, 0, 1),
                          0.2, 0.2, timeout_seconds=45)
    check("move_to_coordinates", True, "%.1f сек" % (time.time() - t0))
except Exception as e:
    check("move_to_coordinates", False, e)

print("9) manage_gripper(40, 40)", flush=True)
try:
    m.manage_gripper(40, 40, timeout_seconds=30)
    check("manage_gripper", True)
except Exception as e:
    check("manage_gripper", False, e)

print("10) nozzle_power(True/False)", flush=True)
try:
    m.nozzle_power(True, timeout_seconds=15)
    m.nozzle_power(False, timeout_seconds=15)
    check("nozzle_power", True)
except Exception as e:
    check("nozzle_power", False, e)

print("11) get_current_errors()", flush=True)
try:
    errs = m.get_current_errors(timeout_seconds=15)
    check("get_current_errors", True, str(errs)[:120])
except Exception as e:
    check("get_current_errors", False, e)

print()
passed = sum(1 for _, ok in results if ok)
print(f"ИТОГО: {passed}/{len(results)} проверок пройдено", flush=True)
sys.exit(0 if passed == len(results) else 1)
