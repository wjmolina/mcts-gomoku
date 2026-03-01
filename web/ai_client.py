#!/usr/bin/env python3
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

SERVER = os.environ.get("SERVER", "http://127.0.0.1:8000")
ENGINE_BIN = os.environ.get("ENGINE_BIN", str(Path(__file__).resolve().parent.parent / "target/release/mcts-gomoku"))
ENGINE_SECONDS = int(os.environ.get("ENGINE_SECONDS", "60"))
POLL = float(os.environ.get("POLL_SECS", "0.5"))


def post(path: str, payload: dict):
    req = urllib.request.Request(
        SERVER + path,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def get(path: str):
    req = urllib.request.Request(SERVER + path, method="GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def parse_best_move(stdout: str):
    best = None
    for line in stdout.splitlines():
        line = line.strip()
        if "," in line and not line.startswith("best="):
            parts = line.split(",")
            if len(parts) == 2:
                try:
                    best = (int(parts[0]), int(parts[1]))
                    continue
                except ValueError:
                    pass
        if line.startswith("best="):
            left = line.split(",", 1)[0]
            parts = left.replace("best=", "").split()
            if len(parts) == 2:
                try:
                    best = (int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
    return best


def think(moves: list[list[int]]):
    args = [f"{x},{y}" for x, y in moves]
    env = os.environ.copy()
    env["MCTS_SECONDS"] = str(ENGINE_SECONDS)
    p = subprocess.run([ENGINE_BIN, *args], text=True, capture_output=True, env=env, timeout=max(ENGINE_SECONDS + 10, 15))
    if p.stdout:
        print(p.stdout, end="")
    if p.stderr:
        print(p.stderr, end="")
    if p.returncode != 0:
        raise RuntimeError((p.stderr or "engine failed").strip())
    mv = parse_best_move(p.stdout)
    if mv is None:
        raise RuntimeError("could not parse engine move")
    return mv


def main():
    pid = post("/connect", {})["player_id"]
    st = post("/join", {"player_id": pid})

    while True:
        if st.get("status") != "matched":
            time.sleep(POLL)
            st = get(f"/state?player_id={pid}")
            continue

        if st.get("over"):
            st = post("/join", {"player_id": pid})
            continue

        if st.get("turn") == st.get("side"):
            x, y = think(st.get("moves", []))
            st = post("/play", {"player_id": pid, "x": x, "y": y})
        else:
            time.sleep(POLL)
            st = get(f"/state?player_id={pid}")


if __name__ == "__main__":
    main()
