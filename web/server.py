#!/usr/bin/env python3
import json
import os
import random
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
INDEX_PATH = WEB_DIR / "index.html"

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

N = 15
CELLS = N * N
E, BLACK, WHITE = 0, 1, 2
DIRS = ((1, 0), (0, 1), (1, 1), (1, -1))
PLAYER_TTL_SECS = 3600

PLAYERS = {}  # player_id -> {game_id, side, last_seen}
GAMES = {}    # game_id -> state
QUEUE = deque()


def idx(x: int, y: int) -> int:
    return y * N + x


def now() -> float:
    return time.time()


def touch(player_id: str) -> None:
    if player_id in PLAYERS:
        PLAYERS[player_id]["last_seen"] = now()


def prune_waiting_players() -> None:
    cutoff = now() - PLAYER_TTL_SECS
    fresh = deque()
    seen = set()
    for pid in QUEUE:
        if pid in seen:
            continue
        seen.add(pid)
        p = PLAYERS.get(pid)
        if not p:
            continue
        if p["game_id"] is not None:
            continue
        if p["last_seen"] < cutoff:
            del PLAYERS[pid]
            continue
        fresh.append(pid)
    QUEUE.clear()
    QUEUE.extend(fresh)


def empty_game(black_id: str, white_id: str) -> dict:
    return {
        "board": [E] * CELLS,
        "moves": [],
        "turn": BLACK,
        "over": False,
        "win_marks": [],
        "last_move": None,
        "players": {str(BLACK): black_id, str(WHITE): white_id},
    }


def win_line(board: list[int], x: int, y: int, side: int):
    for dx, dy in DIRS:
        line = [(x, y)]
        for k in range(1, 5):
            nx, ny = x + dx * k, y + dy * k
            if not (0 <= nx < N and 0 <= ny < N) or board[idx(nx, ny)] != side:
                break
            line.append((nx, ny))
        for k in range(1, 5):
            nx, ny = x - dx * k, y - dy * k
            if not (0 <= nx < N and 0 <= ny < N) or board[idx(nx, ny)] != side:
                break
            line.insert(0, (nx, ny))
        if len(line) >= 5:
            return [[a, b] for a, b in line]
    return None


def apply_move(game: dict, x: int, y: int, side: int) -> bool:
    if game["over"]:
        return False
    if not (0 <= x < N and 0 <= y < N):
        return False
    i = idx(x, y)
    if game["board"][i] != E:
        return False

    game["board"][i] = side
    game["moves"].append([x, y])
    game["last_move"] = [x, y]

    line = win_line(game["board"], x, y, side)
    if line:
        game["over"] = True
        game["win_marks"] = line
    elif len(game["moves"]) == CELLS:
        game["over"] = True
        game["win_marks"] = []
    else:
        game["turn"] = WHITE if side == BLACK else BLACK
    return True


def player_state(player_id: str) -> dict:
    p = PLAYERS[player_id]
    gid = p["game_id"]
    if gid is None:
        return {
            "status": "waiting",
            "player_id": player_id,
        }

    g = GAMES.get(gid)
    if g is None:
        p["game_id"] = None
        p["side"] = None
        return {
            "status": "waiting",
            "player_id": player_id,
        }
    return {
        "status": "matched",
        "player_id": player_id,
        "game_id": gid,
        "side": p["side"],
        "board": g["board"],
        "moves": g["moves"],
        "turn": g["turn"],
        "over": g["over"],
        "win_marks": g["win_marks"],
        "last_move": g["last_move"],
    }


def ensure_player(player_id: Optional[str]) -> str:
    if player_id and player_id in PLAYERS:
        touch(player_id)
        return player_id
    pid = uuid4().hex
    PLAYERS[pid] = {"game_id": None, "side": None, "last_seen": now()}
    return pid


def maybe_match(player_id: str) -> None:
    p = PLAYERS[player_id]
    if p["game_id"] is not None:
        return

    prune_waiting_players()

    opponent = None
    while QUEUE:
        cand = QUEUE.popleft()
        if cand == player_id:
            continue
        cp = PLAYERS.get(cand)
        if not cp:
            continue
        if cp["game_id"] is not None:
            continue
        opponent = cand
        break

    if opponent is None:
        QUEUE.append(player_id)
        return

    black_id, white_id = (opponent, player_id) if random.getrandbits(1) == 0 else (player_id, opponent)
    gid = uuid4().hex
    GAMES[gid] = empty_game(black_id, white_id)
    PLAYERS[black_id]["game_id"] = gid
    PLAYERS[black_id]["side"] = BLACK
    PLAYERS[white_id]["game_id"] = gid
    PLAYERS[white_id]["side"] = WHITE


def restart_player(player_id: str) -> None:
    p = PLAYERS[player_id]
    gid = p["game_id"]
    if gid is None:
        maybe_match(player_id)
        return

    g = GAMES.pop(gid, None)
    participants = [player_id]
    if g is not None:
        participants = list(g["players"].values())

    for pid in participants:
        q = PLAYERS.get(pid)
        if not q:
            continue
        if q["game_id"] != gid:
            continue
        q["game_id"] = None
        q["side"] = None
        if pid not in QUEUE:
            QUEUE.append(pid)

    maybe_match(player_id)


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_html(INDEX_PATH)
            return

        if parsed.path == "/state":
            q = parse_qs(parsed.query)
            pid = (q.get("player_id") or [None])[0]
            if not pid or pid not in PLAYERS:
                self._send_json(404, {"error": "player not found"})
                return
            touch(pid)
            self._send_json(200, player_state(pid))
            return

        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/connect":
            payload = self._read_json()
            pid = ensure_player(payload.get("player_id"))
            self._send_json(200, {"player_id": pid})
            return

        if parsed.path == "/join":
            payload = self._read_json()
            pid = payload.get("player_id")
            if not pid or pid not in PLAYERS:
                self._send_json(404, {"error": "player not found"})
                return
            touch(pid)
            maybe_match(pid)
            self._send_json(200, player_state(pid))
            return

        if parsed.path == "/restart":
            payload = self._read_json()
            pid = payload.get("player_id")
            if not pid or pid not in PLAYERS:
                self._send_json(404, {"error": "player not found"})
                return
            touch(pid)
            restart_player(pid)
            self._send_json(200, player_state(pid))
            return

        if parsed.path == "/play":
            payload = self._read_json()
            pid = payload.get("player_id")
            if not pid or pid not in PLAYERS:
                self._send_json(404, {"error": "player not found"})
                return
            touch(pid)

            p = PLAYERS[pid]
            gid = p["game_id"]
            if gid is None:
                self._send_json(409, {"error": "player not matched"})
                return
            g = GAMES[gid]
            if g["over"]:
                self._send_json(200, player_state(pid))
                return

            side = p["side"]
            if g["turn"] != side:
                self._send_json(409, {"error": "not your turn"})
                return

            try:
                x = int(payload["x"])
                y = int(payload["y"])
            except Exception:
                self._send_json(400, {"error": "invalid payload"})
                return

            if not apply_move(g, x, y, side):
                self._send_json(400, {"error": "illegal move"})
                return

            self._send_json(200, player_state(pid))
            return

        self.send_error(404)

    def log_message(self, fmt, *args):
        return


def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"missing frontend: {INDEX_PATH}")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
