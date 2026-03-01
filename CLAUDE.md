# mcts-gomoku Project Log

This file summarizes what was implemented during this project across engine, frontend, backend, testing, and EC2 runtime/deployment operations.

## 1) Core Goal

Build an intentionally minimal Rust MCTS Gomoku (15x15, 5-in-a-row) engine, then add enough surrounding tooling/UI to:
- play vs engine locally,
- run matches/experiments,
- expose play over HTTP with persistent game state,
- and deploy/run on EC2.

## 2) Engine (Rust, `src/main.rs`)

### Initial direction
- Built a single-file engine (`src/main.rs`) per request (no multi-file split).
- Used MCTS with:
  - UCT selection,
  - expansion,
  - rollout,
  - backpropagation.
- Shared tree across threads (single arena + synchronized node structures).
- Added transposition table support and canonical hashing under board symmetries.

### Parallelism and resource usage
- Engine uses `available_parallelism()` for worker thread count.
- Shared arena + atomics + mutex-protected child/untried vectors.
- Virtual loss added in selection to reduce thread collisions.

### Hyperparameters/env configs
- `MCTS_LOCAL_RADIUS` (default: 2)
- `MCTS_SECONDS` (default: 1)
- `MCTS_ITERS` (default: unset)
- `MCTS_SEED` (default: unset)
- `MCTS_EARLY_STOP_RATIO` (default: 0.99)
- `MCTS_EARLY_STOP_MIN_VISITS` (default: 10000)
- `MCTS_TACTICAL_DEPTH` (default: 2)
- `MCTS_TACTICAL_TOPK` (default: 225)
- `MCTS_DEBUG` (default: off; set to `1` to enable)

### Tactical layer
- Shallow alpha-beta tactical pass at root (depth/top-k configurable).
- Tactical can override pure MCTS root choice.
- Merges candidates from all legal root moves (not just expanded children) to avoid missing critical moves.

### Output/logging
- Engine prints minimal final move as `x,y`.
- Verbose debug output gated behind `MCTS_DEBUG=1`.

## 3) DRY/minimality refactors (engine)

Multiple rounds of DRY audit across 5 commits (`b408628` → `347a2d5`):
- `make_node()` / `make_root_node()` — canonical Node constructor
- `zero_candidate()` — canonical zero-visit Candidate constructor
- `resolve_mv()` — fallback move unwrap helper
- `opp()` — opponent color helper (eliminated 3 inline expressions)
- `Board::on_board` used consistently in `wins_at`
- `idx()` used consistently in `parse_move` and `transform_index`
- Tests: `make_root_node` replaces manual `Node{...}` literals; `let cfg` deduplicated in worker tests

## 4) Tests and Coverage

- 57 tests total (55 unit + 2 CLI integration in `tests/cli.rs`).
- Coverage: **100% regions / 100% functions / 100% lines** (verified via `cargo llvm-cov`).
- Tests cover: board rules, win detection, parsing, worker branches, rollout/selection/backprop, tactical selection, run loop stop conditions, env parsing, transposition/symmetry hashing, root candidate merge.

## 5) Backend/Service (`server.py`, `ai.py`)

### Server
- Minimal HTTP server for game lifecycle and stateful play.
- Supports connect/join/state/play/restart flow and queue-based pairing.
- Added `/restart` endpoint: tears down current game, re-queues both players.
- Fixed `GAMES.get()` guard in `player_state` for missing game robustness.
- Maintains game state server-side (frontend stays thin).
- Serves `web/index.html`.

### AI client bridge
- `ai.py` polls server and plays as one queued participant.
- Invokes engine binary each AI turn with move history args.
- Uses environment-configured think time (`ENGINE_SECONDS`, default 60).
- Parses both legacy `best=...` lines and new minimal `x,y` output.
- Emits captured engine stdout/stderr into `ai.log`.

## 6) Frontend (`web/index.html`)

- Centered board layout, responsive sizing, wood theme.
- Hover marker, last-move indicator, win marking.
- Restart button tile: mini board-cell canvas positioned top-left of board.
  - Hot/down visual states, hit-test via fractional bounds.
  - Calls `/restart` endpoint; re-queues player without full reconnect.
- `lastMouse` tracked to recompute hover correctly on state change.
- `restartSize()` / `redrawRestart()` helpers to avoid repeated `getBoundingClientRect` calls.
- `mousePos()` result destructured explicitly instead of spread trick.

## 7) EC2 operations

- `ssh -i /tmp/wmolina-tmp.pem ec2-user@54.235.29.11`
- Project path on EC2: `/home/ec2-user/mcts`
- Synced via `rsync`, built release binary, started/restarted `server.py` and `ai.py`.
- Bound to `HOST=0.0.0.0`, port 8000 for public access.
- Notable EC2 runtime settings: `MCTS_DEBUG=1`, `MCTS_EARLY_STOP_MIN_VISITS=100000`.

## 8) Debug log keys (when `MCTS_DEBUG=1`)

Visible in `ai.log`:
- `root_expanded`, `root_unexpanded`
- `tactical_depth`, `tactical_topk`, `tactical_chosen`
- `mcts_chosen`, `total_root_visits`

## 9) Current architecture

- Engine: Rust binary `target/release/mcts-gomoku`
- Server: Python HTTP service (`server.py`)
- AI bridge: Python polling client (`ai.py`)
- Frontend: single HTML app (`web/index.html`)
- Deployment: single EC2 instance running all components
- Local directory renamed: `mcts` → `mcts-gomoku`

## 10) Important behavior notes

- This is **free Gomoku** (no swap rules, no forbidden moves). Black has a theoretical forced win from center — not used in competitive tournament play (which uses Swap2).
- `MCTS_LOCAL_RADIUS=2` is the main strength limiter — moves >2 cells from any stone are invisible to the engine.
- Tactical alpha-beta is shallow (depth 2) and root-level only.
- `MCTS_ITERS` acts per-thread, not as a strict global iteration cap.
- Early stop controls compute efficiency, not correctness.

## 11) Recent commits

- `b408628` — extract make_node, zero_candidate, resolve_mv
- `760dac6` — extract opp(), unify bounds check in wins_at
- `c9f7e54` — use idx() in parse_move; use make_node in tests
- `604f07e` — use idx() in transform_index
- `347a2d5` — deduplicate test_cfg calls in worker tests
- `8045e55` — add restart button tile to frontend; add CLAUDE.md
- `edde4dd` — add /restart endpoint and wire up restart button
- `947d37d` — track lastMouse to recompute hover on state change
