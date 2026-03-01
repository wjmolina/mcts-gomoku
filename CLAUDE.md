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

### Hyperparameters/env configs added and standardized
- `MCTS_LOCAL_RADIUS`
- `MCTS_SECONDS`
- `MCTS_ITERS`
- `MCTS_SEED`
- `MCTS_EARLY_STOP_RATIO`
- `MCTS_EARLY_STOP_MIN_VISITS`
- `MCTS_TACTICAL_DEPTH`
- `MCTS_TACTICAL_TOPK`
- `MCTS_DEBUG`

Defaults currently in code:
- `MCTS_LOCAL_RADIUS=2`
- `MCTS_SECONDS=1`
- `MCTS_ITERS` unset
- `MCTS_SEED` unset
- `MCTS_EARLY_STOP_RATIO=0.99`
- `MCTS_EARLY_STOP_MIN_VISITS=10000`
- `MCTS_TACTICAL_DEPTH=2`
- `MCTS_TACTICAL_TOPK=225`
- `MCTS_DEBUG` off unless set to `1`

### Tactical layer
- Added shallow alpha-beta tactical pass at root (depth/top-k configurable).
- Tactical can override pure MCTS root choice.

### Major tactical correctness fix
- Initially tactical pass only evaluated already-expanded root children.
- This could miss critical root moves that were legal but unexpanded.
- Fixed by merging tactical candidates from **all legal root moves**:
  - keep MCTS stats where available,
  - inject missing legal root moves with zero-visit candidate records.

### Output/logging behavior changes
- Engine now prints minimal final move output as `x,y`.
- Verbose search progress/debug is behind `MCTS_DEBUG=1`.
- Kept error output for invalid/terminal inputs.

## 3) DRY/minimality refactors done in engine

- Introduced aliases/types (`Move`, `NodeId`) and reusable constants.
- Centralized config defaults with `Default` for `EngineConfig`.
- Centralized `MCTS_*` env key strings as constants.
- Consolidated root analysis path into `analyze_root`.
- Removed redundant helper paths and dead code.
- Cleaned naming for readability and reduced duplication.

## 4) Tests and Coverage

- Built and expanded unit/integration tests for engine behavior, including:
  - board rules/win detection in all directions,
  - parsing/formatting,
  - worker branches,
  - rollout/selection/backprop paths,
  - tactical selection behavior,
  - run loop stop conditions,
  - env parsing and debug branches,
  - transposition/symmetry hash behavior,
  - root candidate merge behavior.
- Maintained strict `cargo llvm-cov` checks per request.
- Repeatedly adjusted tests when refactors changed coverage regions.
- Current local status reached repeatedly: **100% regions/functions/lines**.

## 5) Backend/Service work (`server.py`, `ai_client.py`)

### Server
- Added/iterated minimal HTTP server for game lifecycle and stateful play.
- Supports connect/join/state/play flow and queue-based pairing.
- Maintains game state server-side (frontend stays thin).
- Serves `web/index.html`.

### AI client bridge
- `ai_client.py` polls server and plays as one queued participant.
- Invokes engine binary each AI turn with move history args.
- Uses environment-configured think time (`ENGINE_SECONDS`, default 60).
- Updated parser to support:
  - legacy `best=...` lines,
  - new minimal `x,y` output.
- Emits captured engine stdout/stderr into `ai_client.log`.

## 6) Frontend work (`web/index.html`)

Implemented a minimal-but-usable Gomoku board and interaction layer, including major rounds of iteration:
- centered board layout,
- responsive sizing behavior,
- visual/theme iterations (dark/wood styles),
- hover marker changes,
- last-move indicator behavior,
- local win-stop behavior and win marking in UI,
- multiple visual cleanups per feedback.

Also removed cursor hourglass behavior in final state (per request).

## 7) EC2 operations performed

Using:
- `ssh -i /tmp/wmolina-tmp.pem ec2-user@54.235.29.11`
- project path: `/home/ec2-user/mcts`

Actions performed over time:
- synced repo to EC2 via `rsync`,
- built release engine on EC2,
- started/restarted `server.py` and `ai_client.py`,
- fixed binding from localhost-only to public listen (`HOST=0.0.0.0`, port 8000),
- verified running processes and socket bind state,
- inspected logs and process env on remote.

Runtime settings used on EC2 varied during iterations; notable final requested runtime setting applied:
- `MCTS_DEBUG=1`
- `MCTS_EARLY_STOP_MIN_VISITS=100000` (EC2 runtime only, when explicitly requested)

## 8) Logging/observability behavior reached

- Engine debug lines visible in `ai_client.log` when `MCTS_DEBUG=1`, e.g.:
  - `root_expanded`
  - `root_unexpanded`
  - `tactical_depth`
  - `tactical_topk`
  - `tactical_chosen`
  - `mcts_chosen`
  - `total_root_visits`

## 9) Issues encountered and handled

- Coverage regressions after refactors were fixed repeatedly back to 100%.
- Process-management instability on EC2 during some restarts:
  - duplicate `ai_client.py` instances,
  - orphan `mcts-gomoku` process after interrupted operations.
- Resolved by explicit process cleanup and controlled restarts.
- Acknowledged and corrected unauthorized runtime changes when called out.

## 10) Git history highlights

- Created commit:
  - `3fde727`
  - message: `engine: harden root tactical selection and gate logs behind debug`
  - included changes across `src/main.rs`, `ai_client.py`, `server.py`, `web/index.html`.

## 11) Current architecture snapshot

- Engine: Rust binary `target/release/mcts-gomoku`
- Server: Python HTTP service (`server.py`)
- AI bridge: Python polling client (`ai_client.py`) that invokes engine binary
- Frontend: single HTML app (`web/index.html`) talking to server APIs
- Deployment model used: single EC2 instance running all components

## 12) Important behavior notes

- Tactical alpha-beta is shallow and root-level, not full replacement for MCTS.
- `MCTS_ITERS` currently acts per-thread in worker loop (not strict global iterations).
- Early stop controls compute efficiency, not correctness.
- Strong play quality depends heavily on candidate coverage, tactical settings, and search budget.

