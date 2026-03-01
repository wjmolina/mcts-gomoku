# mcts-gomoku

## What this is

A from-scratch MCTS Gomoku engine for a 15×15 board (5-in-a-row, free play). Written in Rust with zero external dependencies. Includes a training-free engine, a matchmaking server, and a browser UI.

## Files

- `src/main.rs` - everything: MCTS, tactical alpha-beta, board, CLI entry point
- `tests/cli.rs` - integration tests: spawn the binary, check exit code and output
- `web/index.html` - browser UI
- `web/server.py` - stdlib HTTP server; manages player sessions, matchmaking queue, and game state
- `web/ai_client.py` - bot player; joins the queue like any other player and calls the engine for moves
- `Cargo.toml` - no external dependencies

## Commands

Build:

```
cargo build --release
```

Test (55 unit + 2 integration, 100% coverage via `cargo llvm-cov`):

```
cargo test
cargo llvm-cov
```

Run UI (two terminals):

```
python3 web/server.py
python3 web/ai_client.py
```

## Engine protocol

Input: move history as positional args, each `x,y` (0-indexed, column,row).

```
./target/release/mcts-gomoku 7,7 7,8 6,7
```

Output: one line on stdout, `x,y` — the chosen move.

Exit code 0 on success, non-zero on invalid input (out-of-bounds move, already-terminal position).

## Engine env vars

| var | default | meaning |
| --- | --- | --- |
| `MCTS_SECONDS` | 1 | think time per move |
| `MCTS_ITERS` | unset | iteration cap (overrides time if set) |
| `MCTS_LOCAL_RADIUS` | 2 | only consider moves within N cells of existing stones |
| `MCTS_EARLY_STOP_RATIO` | 0.99 | stop early if best child has this fraction of root visits |
| `MCTS_EARLY_STOP_MIN_VISITS` | 10000 | minimum root visits before early stop applies |
| `MCTS_TACTICAL_DEPTH` | 2 | alpha-beta depth for tactical pre-search |
| `MCTS_TACTICAL_TOPK` | 225 | candidate moves considered in tactical search |
| `MCTS_SEED` | unset | RNG seed for reproducibility |
| `MCTS_DEBUG` | unset | set to `1` for verbose stdout output |

## AI client env vars

| var | default | meaning |
| --- | --- | --- |
| `SERVER` | `http://127.0.0.1:8000` | server URL |
| `ENGINE_BIN` | `target/release/mcts-gomoku` | path to engine binary |
| `MCTS_SECONDS` | 60 | passed through to the engine |
| `POLL_SECS` | 0.5 | how often to poll for opponent's move |

## Architecture

The server is a pure game state manager — it has no knowledge of the engine. The ai_client joins the matchmaking queue as a player (identical to a browser tab), calls the engine subprocess on its turn, and submits moves via `/play`. Two ai_client instances will play each other.

## Network architecture

Parallel MCTS with shared tree across threads using a single arena. Virtual loss reduces thread collisions. Tactical alpha-beta pre-search at root can override MCTS choice.

- `MCTS_LOCAL_RADIUS=2` is the main strength limiter — moves more than 2 cells from any stone are invisible to the engine
- Tactical search is shallow (depth 2) and root-level only
- `MCTS_ITERS` acts per-thread, not as a global cap

## THREADS and the thermal/power issue

Uses `available_parallelism()` for worker thread count — no manual tuning needed.

## Code style

- No comments in production code
- Zero external dependencies
- cargo fmt for formatting
