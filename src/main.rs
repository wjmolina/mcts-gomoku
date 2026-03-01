use std::collections::HashMap;
use std::env;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const N: usize = 15;
const CELLS: usize = N * N;
const BLACK: u8 = 1;
const WHITE: u8 = 2;
const DIRS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];
const UCT_C: f32 = std::f32::consts::SQRT_2;
const BACKPROP_WIN: u64 = 2;
const BACKPROP_DRAW: u64 = 1;
const BACKPROP_LOSS: u64 = 0;
const PATH_CAPACITY: usize = 256;

type Move = u16;
type NodeId = usize;

#[derive(Clone)]
struct Board {
    cells: [u8; CELLS],
    side: u8,
    moves_played: usize,
    last: Option<usize>,
}

impl Board {
    fn new() -> Self {
        Self {
            cells: [0; CELLS],
            side: BLACK,
            moves_played: 0,
            last: None,
        }
    }

    fn legal_moves(&self) -> Vec<Move> {
        let mut out = Vec::with_capacity(CELLS - self.moves_played);
        for i in 0..CELLS {
            if self.cells[i] == 0 {
                out.push(i as Move);
            }
        }
        out
    }

    fn local_moves(&self, radius: usize) -> Vec<Move> {
        if self.moves_played == 0 || radius == 0 {
            return self.legal_moves();
        }

        let mut out = Vec::with_capacity(CELLS - self.moves_played);
        for y in 0..N {
            for x in 0..N {
                let i = idx(x, y);
                if self.cells[i] != 0 {
                    continue;
                }
                let x0 = x.saturating_sub(radius);
                let y0 = y.saturating_sub(radius);
                let x1 = (x + radius).min(N - 1);
                let y1 = (y + radius).min(N - 1);
                let mut near = false;
                'scan: for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        if self.cells[idx(nx, ny)] != 0 {
                            near = true;
                            break 'scan;
                        }
                    }
                }
                if near {
                    out.push(i as Move);
                }
            }
        }

        if out.is_empty() {
            self.legal_moves()
        } else {
            out
        }
    }

    fn play(&mut self, mv: usize) {
        self.cells[mv] = self.side;
        self.moves_played += 1;
        self.last = Some(mv);
        self.side = opp(self.side);
    }

    fn tt_key(&self) -> u64 {
        canonical_hash(&self.cells, self.side)
    }

    fn winner(&self) -> Option<u8> {
        let last = self.last?;
        let x = (last % N) as isize;
        let y = (last / N) as isize;
        let p = self.cells[last];
        for (dx, dy) in DIRS {
            let mut count = 1;
            let mut i = 1;
            while Self::on_board(x + dx * i, y + dy * i) {
                let nx = (x + dx * i) as usize;
                let ny = (y + dy * i) as usize;
                if self.cells[idx(nx, ny)] != p {
                    break;
                }
                count += 1;
                i += 1;
            }
            i = 1;
            while Self::on_board(x - dx * i, y - dy * i) {
                let nx = (x - dx * i) as usize;
                let ny = (y - dy * i) as usize;
                if self.cells[idx(nx, ny)] != p {
                    break;
                }
                count += 1;
                i += 1;
            }
            if count >= 5 {
                return Some(p);
            }
        }
        None
    }

    fn terminal(&self) -> Option<u8> {
        if let Some(w) = self.winner() {
            return Some(w);
        }
        if self.moves_played == CELLS {
            return Some(0);
        }
        None
    }

    fn on_board(x: isize, y: isize) -> bool {
        x >= 0 && y >= 0 && x < N as isize && y < N as isize
    }
}

struct Node {
    visits: AtomicU64,
    win_halves: AtomicU64,
    virtual_loss: AtomicU64,
    terminal: Option<u8>,
    untried: Mutex<Vec<Move>>,
    children: Mutex<Vec<(Move, NodeId)>>,
}

type Arena = Arc<RwLock<Vec<Arc<Node>>>>;
type Tt = Arc<TtSharded>;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    mv: usize,
    visits: u64,
    winrate: f64,
}

const SIDE_HASH: u64 = 0x9E37_79B9_7F4A_7C15;
const TT_SHARDS: usize = 64;
const ENV_LOCAL_RADIUS: &str = "MCTS_LOCAL_RADIUS";
const ENV_SECONDS: &str = "MCTS_SECONDS";
const ENV_ITERS: &str = "MCTS_ITERS";
const ENV_SEED: &str = "MCTS_SEED";
const ENV_EARLY_STOP_RATIO: &str = "MCTS_EARLY_STOP_RATIO";
const ENV_EARLY_STOP_MIN_VISITS: &str = "MCTS_EARLY_STOP_MIN_VISITS";
const ENV_TACTICAL_DEPTH: &str = "MCTS_TACTICAL_DEPTH";
const ENV_TACTICAL_TOPK: &str = "MCTS_TACTICAL_TOPK";
const ENV_DEBUG: &str = "MCTS_DEBUG";

#[derive(Clone, Copy)]
struct EngineConfig {
    local_radius: usize,
    seconds: u64,
    max_iters: Option<u64>,
    seed_base: Option<u64>,
    early_stop_ratio: f64,
    early_stop_min_visits: u64,
    tactical_depth: usize,
    tactical_topk: usize,
    debug: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            local_radius: 2,
            seconds: 1,
            max_iters: None,
            seed_base: None,
            early_stop_ratio: 0.99,
            early_stop_min_visits: 10_000,
            tactical_depth: 2,
            tactical_topk: 225,
            debug: false,
        }
    }
}

impl EngineConfig {
    fn parse_or<T: std::str::FromStr>(key: &str, default: T) -> T {
        env::var(key)
            .ok()
            .and_then(|v| v.parse::<T>().ok())
            .unwrap_or(default)
    }

    fn parse_opt<T: std::str::FromStr>(key: &str) -> Option<T> {
        env::var(key).ok().and_then(|v| v.parse::<T>().ok())
    }

    /// Parses all engine hyperparameters from environment with defaults.
    fn from_env() -> Self {
        let mut cfg = Self::default();
        cfg.local_radius = Self::parse_or(ENV_LOCAL_RADIUS, cfg.local_radius);
        cfg.seconds = Self::parse_or(ENV_SECONDS, cfg.seconds);
        cfg.max_iters = Self::parse_opt(ENV_ITERS);
        cfg.seed_base = Self::parse_opt(ENV_SEED);
        cfg.early_stop_ratio = Self::parse_or(ENV_EARLY_STOP_RATIO, cfg.early_stop_ratio);
        cfg.early_stop_min_visits =
            Self::parse_or(ENV_EARLY_STOP_MIN_VISITS, cfg.early_stop_min_visits);
        cfg.tactical_depth = Self::parse_or(ENV_TACTICAL_DEPTH, cfg.tactical_depth);
        cfg.tactical_topk = Self::parse_or(ENV_TACTICAL_TOPK, cfg.tactical_topk);
        cfg.debug = matches!(env::var(ENV_DEBUG).ok().as_deref(), Some("1"));
        cfg
    }
}

struct TtSharded {
    shards: Vec<Mutex<HashMap<u64, NodeId>>>,
}

#[derive(Clone, Copy)]
struct SearchSnapshot {
    best_mv_mcts: Option<usize>,
    best_visits: u64,
    node_count: usize,
    total_visits: u64,
}

struct RootAnalysis {
    snapshot: SearchSnapshot,
    ranked: Vec<Candidate>,
    root_untried: usize,
}

impl TtSharded {
    fn new() -> Self {
        let mut shards = Vec::with_capacity(TT_SHARDS);
        for _ in 0..TT_SHARDS {
            shards.push(Mutex::new(HashMap::new()));
        }
        Self { shards }
    }

    fn shard_idx(hash: u64) -> usize {
        (hash as usize) & (TT_SHARDS - 1)
    }

    fn get_or_insert_with<F: FnOnce() -> NodeId>(&self, hash: u64, create: F) -> NodeId {
        let mut shard = self.shards[Self::shard_idx(hash)].lock().unwrap();
        if let Some(&idx) = shard.get(&hash) {
            idx
        } else {
            let idx = create();
            shard.insert(hash, idx);
            idx
        }
    }

    fn insert(&self, hash: u64, idx: NodeId) {
        self.shards[Self::shard_idx(hash)].lock().unwrap().insert(hash, idx);
    }
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn piece_hash(cell: usize, player: u8) -> u64 {
    splitmix64(((cell as u64) << 2) ^ (player as u64) ^ 0xA5A5_5A5A_DEAD_BEEF)
}

fn transform_index(sym: usize, cell: usize) -> usize {
    let x = cell % N;
    let y = cell / N;
    let (tx, ty) = match sym {
        0 => (x, y),
        1 => (N - 1 - y, x),
        2 => (N - 1 - x, N - 1 - y),
        3 => (y, N - 1 - x),
        4 => (N - 1 - x, y),
        5 => (x, N - 1 - y),
        6 => (y, x),
        7 => (N - 1 - y, N - 1 - x),
        _ => unreachable!(),
    };
    idx(tx, ty)
}

/// Canonical transposition key under all D4 board symmetries plus side-to-move.
fn canonical_hash(cells: &[u8; CELLS], side: u8) -> u64 {
    let mut hashes = [0u64; 8];
    for (cell, p) in cells.iter().copied().enumerate() {
        if p == 0 {
            continue;
        }
        for (s, h) in hashes.iter_mut().enumerate() {
            *h ^= piece_hash(transform_index(s, cell), p);
        }
    }
    if side == WHITE {
        for h in &mut hashes {
            *h ^= SIDE_HASH;
        }
    }
    hashes.into_iter().min().unwrap()
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn gen_usize(&mut self, n: usize) -> usize {
        (self.next_u64() as usize) % n
    }
}

fn uct(child_win_halves: u64, child_visits: u64, child_vl: u64, parent_visits: u64) -> f32 {
    let v = child_visits + child_vl;
    if v == 0 {
        return f32::INFINITY;
    }
    let q = (child_win_halves as f32) / (2.0 * v as f32);
    let u = UCT_C * ((parent_visits.max(1) as f32).ln() / v as f32).sqrt();
    q + u
}

fn parse_move(s: &str) -> Option<usize> {
    let (a, b) = s.split_once(',')?;
    let x: usize = a.parse().ok()?;
    let y: usize = b.parse().ok()?;
    if x < N && y < N { Some(idx(x, y)) } else { None }
}

fn move_to_xy(mv: usize) -> (usize, usize) {
    (mv % N, mv / N)
}

fn opp(player: u8) -> u8 {
    if player == BLACK { WHITE } else { BLACK }
}

fn opening_move(board: &Board) -> Option<usize> {
    match board.moves_played {
        // Black always opens at center.
        0 => Some(idx(N / 2, N / 2)),
        // If Black opened at center, White responds with a strong diagonal offset.
        1 if board.cells[idx(N / 2, N / 2)] == BLACK => Some(idx(N / 2 + 1, N / 2 + 1)),
        _ => None,
    }
}

fn idx(x: usize, y: usize) -> usize {
    y * N + x
}

fn node_at(arena: &Arena, idx: NodeId) -> Arc<Node> {
    arena.read().unwrap()[idx].clone()
}

fn add_child(arena: &Arena, node: Node) -> NodeId {
    let mut a = arena.write().unwrap();
    a.push(Arc::new(node));
    a.len() - 1
}

fn tt_with_root(hash: u64, idx: NodeId) -> Tt {
    let tt = Arc::new(TtSharded::new());
    tt.insert(hash, idx);
    tt
}

/// Returns true if placing `player` at `pos` on `cells` would create 5-in-a-row.
/// `cells[pos]` must be empty; no board clone needed.
fn wins_at(cells: &[u8; CELLS], player: u8, pos: usize) -> bool {
    let x = (pos % N) as isize;
    let y = (pos / N) as isize;
    for &(dx, dy) in &DIRS {
        let mut count = 1;
        for &sign in &[1isize, -1isize] {
            let mut i = 1isize;
            loop {
                let nx = x + sign * dx * i;
                let ny = y + sign * dy * i;
                if !Board::on_board(nx, ny) {
                    break;
                }
                if cells[idx(nx as usize, ny as usize)] != player {
                    break;
                }
                count += 1;
                i += 1;
            }
        }
        if count >= 5 {
            return true;
        }
    }
    false
}

/// Heuristic threat score for `player`.
/// Scans every 5-cell window in all four directions. Windows that contain no
/// opponent stones are scored by how many player stones they already hold.
/// Handles broken/split patterns naturally (XX_XX = one k=4 window = 1 000).
/// A non-linear fork bonus rewards threats in multiple directions simultaneously:
///   two fours   → +50 000   (double-four: unblockable)
///   four + three → +10 000  (four-three fork: very strong)
///   two threes  → +5 000    (double-three: usually unblockable)
fn threat_score(cells: &[u8; CELLS], player: u8) -> i32 {
    let opp = opp(player);
    let mut score = 0i32;
    let mut four_dirs = [false; 4];
    let mut three_dirs = [false; 4];
    for (d, &(dx, dy)) in DIRS.iter().enumerate() {
        for sy in 0..N as isize {
            for sx in 0..N as isize {
                // Skip if the 5-cell window doesn't fit on the board.
                if !Board::on_board(sx + dx * 4, sy + dy * 4) {
                    continue;
                }
                let mut p_count = 0u8;
                let mut has_opp = false;
                for k in 0..5 {
                    let cell = cells[((sy + dy * k) * N as isize + sx + dx * k) as usize];
                    if cell == player {
                        p_count += 1;
                    } else if cell == opp {
                        has_opp = true;
                        break;
                    }
                }
                if has_opp {
                    continue;
                }
                if p_count == 4 {
                    four_dirs[d] = true;
                }
                if p_count == 3 {
                    three_dirs[d] = true;
                }
                score += match p_count {
                    4 => 1_000,
                    3 => 100,
                    2 => 10,
                    1 => 1,
                    _ => 0,
                };
            }
        }
    }
    let n_four = four_dirs.iter().filter(|&&x| x).count();
    let n_three = three_dirs.iter().filter(|&&x| x).count();
    score += match (n_four, n_three) {
        (f, _) if f >= 2 => 50_000,
        (1, t) if t >= 1 => 10_000,
        (_, t) if t >= 2 => 5_000,
        _ => 0,
    };
    score
}

/// Static board evaluation: returns the player favoured by position, or 0 for equal.
/// Replaces random rollouts with a fast pattern-based assessment.
fn evaluate(cells: &[u8; CELLS]) -> u8 {
    let bs = threat_score(cells, BLACK);
    let ws = threat_score(cells, WHITE);
    if bs > ws {
        BLACK
    } else if ws > bs {
        WHITE
    } else {
        0
    }
}

fn analyze_root(arena: &Arena) -> RootAnalysis {
    let root_node = node_at(arena, 0);
    let kids = root_node.children.lock().unwrap().clone();
    let root_untried = root_node.untried.lock().unwrap().len();
    let node_count = arena.read().unwrap().len();
    let total_visits = root_node.visits.load(Ordering::Relaxed);

    let mut ranked = Vec::<Candidate>::with_capacity(kids.len());
    let mut best_mv_mcts = None;
    let mut best_visits = 0u64;

    for (mv_u16, c) in kids {
        let cn = node_at(arena, c);
        let visits = cn.visits.load(Ordering::Relaxed);
        let win_halves = cn.win_halves.load(Ordering::Relaxed);
        let cand = Candidate {
            mv: mv_u16 as usize,
            visits,
            winrate: win_halves as f64 / (2.0 * visits.max(1) as f64),
        };
        if visits > best_visits {
            best_visits = visits;
            best_mv_mcts = Some(cand.mv);
        }
        ranked.push(cand);
    }
    ranked.sort_by(|a, b| b.visits.cmp(&a.visits));

    RootAnalysis {
        snapshot: SearchSnapshot {
            best_mv_mcts,
            best_visits,
            node_count,
            total_visits,
        },
        ranked,
        root_untried,
    }
}

fn alpha_beta(board: &Board, depth: usize, mut alpha: i8, beta: i8, radius: usize) -> i8 {
    if let Some(t) = board.terminal() {
        return if t == 0 { 0 } else { -1 };
    }
    if depth == 0 {
        return 0;
    }

    let mut best = -1i8;
    for mv in board.local_moves(radius) {
        let mut b2 = board.clone();
        b2.play(mv as usize);
        let score = -alpha_beta(&b2, depth - 1, -beta, -alpha, radius);
        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Picks forced win first, otherwise first non-losing move from ranked candidates.
fn tactical_pick(
    board: &Board,
    candidates: &[Candidate],
    depth: usize,
    radius: usize,
) -> Option<usize> {
    if depth == 0 || candidates.is_empty() {
        return None;
    }

    let mut first_non_losing = None::<usize>;
    for c in candidates {
        let mv = c.mv;
        let mut b2 = board.clone();
        b2.play(mv);
        let score_for_root = -alpha_beta(&b2, depth.saturating_sub(1), -1, 1, radius);
        if score_for_root == 1 {
            return Some(mv);
        }
        if score_for_root >= 0 && first_non_losing.is_none() {
            first_non_losing = Some(mv);
        }
    }
    first_non_losing
}

fn apply_moves(board: &mut Board, args: &[String]) -> Result<(), String> {
    for a in args {
        match parse_move(a) {
            Some(mv) if board.cells[mv] == 0 => board.play(mv),
            _ => {
                return Err(format!(
                    "invalid move: {a} (expected x,y in 0..14 and legal)"
                ))
            }
        }
    }
    Ok(())
}

fn make_node(terminal: Option<u8>, untried: Vec<Move>) -> Node {
    Node {
        visits: AtomicU64::new(0),
        win_halves: AtomicU64::new(0),
        virtual_loss: AtomicU64::new(0),
        terminal,
        untried: Mutex::new(untried),
        children: Mutex::new(Vec::new()),
    }
}

fn make_root_node(board: &Board, cfg: EngineConfig) -> Node {
    make_node(board.terminal(), board.local_moves(cfg.local_radius))
}

fn spawn_workers(
    threads: usize,
    board: &Board,
    root_player: u8,
    stop: &Arc<AtomicBool>,
    arena: &Arena,
    tt: &Tt,
    cfg: EngineConfig,
) -> Vec<thread::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(threads);
    for i in 0..threads {
        let stop_c = stop.clone();
        let arena_c = arena.clone();
        let tt_c = tt.clone();
        let root_c = board.clone();
        handles.push(thread::spawn(move || {
            worker(i, root_c, root_player, stop_c, arena_c, tt_c, cfg);
        }));
    }
    handles
}

fn print_opening_choice(mv: usize) {
    let (x, y) = move_to_xy(mv);
    println!(
        "best={} {}, visits=0, winrate=0.0000, elapsed=0s, threads=0, nodes=0 (opening)",
        x, y
    );
}

fn print_search_choice(c: Candidate, elapsed_s: u64, threads: usize, node_count: usize) {
    let (x, y) = move_to_xy(c.mv);
    println!(
        "best={} {}, visits={}, winrate={:.4}, elapsed={}s, threads={}, nodes={} (shared tree, unbounded growth)",
        x, y, c.visits, c.winrate, elapsed_s, threads, node_count
    );
}

fn print_move_result(mv: usize) {
    let (x, y) = move_to_xy(mv);
    println!("{x},{y}");
}

fn zero_candidate(mv: usize) -> Candidate {
    Candidate { mv, visits: 0, winrate: 0.0 }
}

fn candidate_for_mv(ranked: &[Candidate], mv: usize) -> Candidate {
    ranked.iter().find(|c| c.mv == mv).copied().unwrap_or_else(|| zero_candidate(mv))
}

fn merged_root_candidates(board: &Board, ranked: &[Candidate]) -> Vec<Candidate> {
    let mut by_move = HashMap::<usize, Candidate>::with_capacity(ranked.len());
    for c in ranked {
        by_move.insert(c.mv, *c);
    }

    let legal = board.legal_moves();
    let mut out = Vec::<Candidate>::with_capacity(legal.len());
    for mv in legal {
        let m = mv as usize;
        out.push(by_move.get(&m).copied().unwrap_or_else(|| zero_candidate(m)));
    }
    out.sort_by(|a, b| b.visits.cmp(&a.visits));
    out
}

fn maybe_print_debug(
    cfg: EngineConfig,
    root_expanded: usize,
    root_unexpanded: usize,
    tactical_chosen: bool,
    mcts_chosen: bool,
    total_root_visits: u64,
) {
    if cfg.debug {
        eprintln!(
            "debug: root_expanded={}, root_unexpanded={}, root_total={}, tactical_depth={}, tactical_topk={}, tactical_chosen={}, mcts_chosen={}, total_root_visits={}",
            root_expanded,
            root_unexpanded,
            root_expanded + root_unexpanded,
            cfg.tactical_depth,
            cfg.tactical_topk,
            tactical_chosen,
            mcts_chosen,
            total_root_visits
        );
    }
}

/// One MCTS worker operating on the shared arena/TT.
fn worker(
    id: usize,
    root: Board,
    root_player: u8,
    stop: Arc<AtomicBool>,
    arena: Arena,
    tt: Tt,
    cfg: EngineConfig,
) {
    let seed = cfg.seed_base.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
    }) ^ ((id as u64) << 32)
        ^ 0x9E3779B97F4A7C15;
    let mut rng = Rng::new(seed);
    let mut iters = 0u64;

    while !stop.load(Ordering::Relaxed) {
        if let Some(limit) = cfg.max_iters {
            if iters >= limit {
                break;
            }
        }
        iters += 1;
        let mut board = root.clone();
        let mut path = Vec::<NodeId>::with_capacity(PATH_CAPACITY);
        let mut vl_path = Vec::<NodeId>::with_capacity(PATH_CAPACITY);
        let mut node_idx = 0usize;
        path.push(0);

        let outcome = loop {
            let node = node_at(&arena, node_idx);

            if let Some(t) = node.terminal {
                break t;
            }

            let maybe_expand = {
                let mut untried = node.untried.lock().unwrap();
                if untried.is_empty() {
                    None
                } else {
                    let i = rng.gen_usize(untried.len());
                    Some(untried.swap_remove(i) as usize)
                }
            };

            if let Some(mv) = maybe_expand {
                board.play(mv);
                let child_terminal = board.terminal();
                let child_hash = board.tt_key();
                let child_idx = tt.get_or_insert_with(child_hash, || {
                    add_child(&arena, make_node(child_terminal, board.local_moves(cfg.local_radius)))
                });
                {
                    let mut kids = node.children.lock().unwrap();
                    if !kids.iter().any(|&(_, id)| id == child_idx) {
                        kids.push((mv as Move, child_idx));
                    }
                }
                path.push(child_idx);

                if let Some(t) = child_terminal {
                    break t;
                }
                break evaluate(&board.cells);
            }

            let kids = node.children.lock().unwrap().clone();
            if kids.is_empty() {
                break evaluate(&board.cells);
            }

            let parent_visits = node.visits.load(Ordering::Relaxed).max(1);
            let mut best = kids[0];
            let mut best_score = f32::NEG_INFINITY;

            for &(edge_mv, c) in &kids {
                let cn = node_at(&arena, c);
                let score = uct(
                    cn.win_halves.load(Ordering::Relaxed),
                    cn.visits.load(Ordering::Relaxed),
                    cn.virtual_loss.load(Ordering::Relaxed),
                    parent_visits,
                );
                if score > best_score {
                    best_score = score;
                    best = (edge_mv, c);
                }
            }

            let chosen = node_at(&arena, best.1);
            chosen.virtual_loss.fetch_add(1, Ordering::Relaxed);
            vl_path.push(best.1);
            let mv = best.0 as usize;
            board.play(mv);
            node_idx = best.1;
            path.push(best.1);
        };

        for (depth, idx) in path.iter().enumerate() {
            let n = node_at(&arena, *idx);
            // Each node stores win-halves from the perspective of the player
            // who *chose* this node (the parent).  Depth 0 = root (root_player
            // aggregate); depth 1 = chosen by root_player; depth 2 = chosen by
            // opponent; etc.  Flip at every other level so UCT selection
            // correctly maximises each side's own win-rate.
            let reward = if depth > 0 && depth % 2 == 0 {
                // Opponent chose this node — store from opponent's perspective.
                if outcome != root_player && outcome != 0 {
                    BACKPROP_WIN
                } else if outcome == 0 {
                    BACKPROP_DRAW
                } else {
                    BACKPROP_LOSS
                }
            } else {
                // Root player's perspective (root itself, or node chosen by root_player).
                if outcome == root_player {
                    BACKPROP_WIN
                } else if outcome == 0 {
                    BACKPROP_DRAW
                } else {
                    BACKPROP_LOSS
                }
            };
            n.visits.fetch_add(1, Ordering::Relaxed);
            n.win_halves.fetch_add(reward, Ordering::Relaxed);
        }

        for idx in vl_path {
            let n = node_at(&arena, idx);
            n.virtual_loss.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

/// Runs analysis + tactical selection on the current search state.
/// Returns (chosen move, tactical_override flag, full analysis).
fn analyse_and_pick(board: &Board, arena: &Arena, cfg: EngineConfig) -> (Option<usize>, bool, RootAnalysis) {
    let analysis = analyze_root(arena);
    let candidates = merged_root_candidates(board, &analysis.ranked);
    let tactical = if cfg.tactical_depth > 0 && cfg.tactical_topk > 0 {
        let k = cfg.tactical_topk.min(candidates.len());
        tactical_pick(board, &candidates[..k], cfg.tactical_depth, cfg.local_radius)
    } else {
        None
    };
    let mv = tactical.or(analysis.snapshot.best_mv_mcts);
    (mv, tactical.is_some(), analysis)
}

fn resolve_mv(chosen_mv: Option<usize>, board: &Board, cfg: EngineConfig) -> usize {
    chosen_mv.unwrap_or_else(|| board.local_moves(cfg.local_radius)[0] as usize)
}

fn run(args: &[String]) -> i32 {
    let mut cfg = EngineConfig::from_env();
    // Guard: seconds=0 with no iteration cap has no termination condition → fall back to default.
    if cfg.seconds == 0 && cfg.max_iters.is_none() {
        cfg.seconds = EngineConfig::default().seconds;
    }
    let mut board = Board::new();

    if let Err(msg) = apply_moves(&mut board, args) {
        eprintln!("{msg}");
        return 1;
    }

    if board.terminal().is_some() {
        eprintln!("position is already terminal");
        return 1;
    }

    if let Some(mv) = opening_move(&board) {
        if cfg.debug {
            print_opening_choice(mv);
        }
        print_move_result(mv);
        return 0;
    }

    // Pre-search forced-move check: play an immediate win or block an immediate
    // opponent win without spinning up workers or sleeping.
    {
        let legal = board.legal_moves();
        let opp = opp(board.side);
        let forced = legal
            .iter()
            .find(|&&m| wins_at(&board.cells, board.side, m as usize))
            .or_else(|| legal.iter().find(|&&m| wins_at(&board.cells, opp, m as usize)));
        if let Some(&mv) = forced {
            print_move_result(mv as usize);
            return 0;
        }
    }

    let root = make_root_node(&board, cfg);
    let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(root)]));
    let tt: Tt = tt_with_root(board.tt_key(), 0usize);
    let root_player = board.side;
    let threads = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let stop = Arc::new(AtomicBool::new(false));
    let handles = spawn_workers(threads, &board, root_player, &stop, &arena, &tt, cfg);

    let start = Instant::now();
    loop {
        thread::sleep(Duration::from_secs(1));

        let (chosen_mv, tactical_override, analysis) = analyse_and_pick(&board, &arena, cfg);
        let snap = analysis.snapshot;
        maybe_print_debug(
            cfg,
            analysis.ranked.len(),
            analysis.root_untried,
            tactical_override,
            snap.best_mv_mcts.is_some(),
            snap.total_visits,
        );
        if cfg.debug {
            let dbg_mv = resolve_mv(chosen_mv, &board, cfg);
            print_search_choice(
                candidate_for_mv(&analysis.ranked, dbg_mv),
                start.elapsed().as_secs(),
                threads,
                snap.node_count,
            );
        }

        if cfg.seconds > 0 && start.elapsed().as_secs() >= cfg.seconds {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        if cfg.seconds > 0
            && snap.best_visits >= cfg.early_stop_min_visits
            && snap.total_visits > 0
            && (snap.best_visits as f64 / snap.total_visits as f64) >= cfg.early_stop_ratio
        {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        let workers_done = cfg
            .max_iters
            .map(|_| handles.iter().all(|h| h.is_finished()))
            .unwrap_or(false);
        if workers_done {
            break;
        }
    }

    for h in handles {
        let _ = h.join();
    }

    let (chosen_mv, _, analysis) = analyse_and_pick(&board, &arena, cfg);
    let mv = resolve_mv(chosen_mv, &board, cfg);
    if cfg.debug {
        print_search_choice(
            candidate_for_mv(&analysis.ranked, mv),
            start.elapsed().as_secs(),
            threads,
            analysis.snapshot.node_count,
        );
    }
    print_move_result(mv);
    0
}

// tarpaulin coverage(off)
fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let code = run(&args);
    if code != 0 {
        std::process::exit(code);
    }
}
// tarpaulin coverage(on)

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn test_cfg(max_iters: Option<u64>, local_radius: usize) -> EngineConfig {
        EngineConfig {
            local_radius,
            seconds: 0,
            max_iters,
            ..EngineConfig::default()
        }
    }

    #[test]
    fn board_new_and_legal_moves() {
        let b = Board::new();
        assert_eq!(b.side, BLACK);
        assert_eq!(b.moves_played, 0);
        assert_eq!(b.last, None);
        assert_eq!(b.legal_moves().len(), CELLS);
        assert_eq!(b.local_moves(2).len(), CELLS);
        assert_eq!(b.terminal(), None);
    }

    #[test]
    fn board_play_switches_side_and_tracks_last() {
        let mut b = Board::new();
        b.play(idx(3, 4));
        assert_eq!(b.cells[idx(3, 4)], BLACK);
        assert_eq!(b.side, WHITE);
        assert_eq!(b.moves_played, 1);
        assert_eq!(b.last, Some(idx(3, 4)));
    }

    #[test]
    fn opening_move_empty_and_non_empty() {
        // Move 0: Black always goes to center.
        let b = Board::new();
        assert_eq!(opening_move(&b), Some(idx(7, 7)));

        // Move 1: Black at center → White gets strong diagonal response.
        let mut b_center = Board::new();
        b_center.play(idx(7, 7));
        assert_eq!(opening_move(&b_center), Some(idx(8, 8)));

        // Move 1: Black NOT at center → no opening book response.
        let mut b2 = Board::new();
        b2.play(idx(0, 0));
        assert_eq!(opening_move(&b2), None);

        // Move 2+: no opening book.
        let mut b3 = Board::new();
        b3.play(idx(7, 7));
        b3.play(idx(8, 8));
        assert_eq!(opening_move(&b3), None);
    }

    #[test]
    fn local_moves_filters_by_radius_and_fallbacks() {
        let mut b = Board::new();
        b.play(idx(7, 7));
        let local = b.local_moves(1);
        assert!(local.len() < b.legal_moves().len());
        assert!(local.contains(&(idx(6, 6) as Move)));
        assert!(!local.contains(&(idx(0, 0) as Move)));

        let all = b.local_moves(0);
        assert_eq!(all.len(), b.legal_moves().len());

        let mut fallback_board = Board::new();
        fallback_board.moves_played = 1;
        let fallback = fallback_board.local_moves(1);
        assert_eq!(fallback.len(), fallback_board.legal_moves().len());
    }

    #[test]
    fn local_moves_invariants() {
        let mut b = Board::new();
        b.play(idx(7, 7));
        b.play(idx(8, 7));
        let local = b.local_moves(2);
        let legal = b.legal_moves();

        let local_set: std::collections::HashSet<Move> = local.iter().copied().collect();
        let legal_set: std::collections::HashSet<Move> = legal.iter().copied().collect();
        assert_eq!(local_set.len(), local.len());
        assert!(local_set.is_subset(&legal_set));
        for mv in &local {
            assert_eq!(b.cells[*mv as usize], 0);
        }
    }

    #[test]
    fn on_board_bounds() {
        assert!(Board::on_board(0, 0));
        assert!(Board::on_board(14, 14));
        assert!(!Board::on_board(-1, 0));
        assert!(!Board::on_board(0, -1));
        assert!(!Board::on_board(15, 0));
        assert!(!Board::on_board(0, 15));
    }

    #[test]
    fn winner_horizontal() {
        let mut b = Board::new();
        let y = 2usize;
        for x in 0..5 {
            b.cells[idx(x, y)] = BLACK;
        }
        b.last = Some(idx(2, y));
        assert_eq!(b.winner(), Some(BLACK));
    }

    #[test]
    fn winner_vertical() {
        let mut b = Board::new();
        let x = 6usize;
        for y in 0..5 {
            b.cells[idx(x, y)] = WHITE;
        }
        b.last = Some(idx(x, 2));
        assert_eq!(b.winner(), Some(WHITE));
    }

    #[test]
    fn winner_diagonal() {
        let mut b = Board::new();
        for i in 0..5 {
            b.cells[idx(i + 1, i + 1)] = BLACK;
        }
        b.last = Some(idx(3, 3));
        assert_eq!(b.winner(), Some(BLACK));
    }

    #[test]
    fn winner_anti_diagonal() {
        let mut b = Board::new();
        for i in 0..5 {
            b.cells[idx(10 - i, i)] = WHITE;
        }
        b.last = Some(idx(8, 2));
        assert_eq!(b.winner(), Some(WHITE));
    }

    #[test]
    fn terminal_draw_and_non_terminal() {
        let mut b = Board::new();
        b.moves_played = CELLS;
        b.last = None;
        assert_eq!(b.terminal(), Some(0));

        let b2 = Board::new();
        assert_eq!(b2.terminal(), None);
    }

    #[test]
    fn parse_and_format_moves() {
        assert_eq!(parse_move("0,0"), Some(0));
        assert_eq!(parse_move("14,14"), Some(idx(14, 14)));
        assert_eq!(parse_move("15,0"), None);
        assert_eq!(parse_move("0,15"), None);
        assert_eq!(parse_move("x,1"), None);
        assert_eq!(parse_move("1,x"), None);
        assert_eq!(parse_move("1"), None);
        assert_eq!(move_to_xy(idx(9, 4)), (9, 4));
    }

    #[test]
    fn transform_index_is_a_permutation_for_each_symmetry() {
        for sym in 0..8 {
            let mut seen = vec![false; CELLS];
            for cell in 0..CELLS {
                let t = transform_index(sym, cell);
                assert!(t < CELLS);
                seen[t] = true;
            }
            assert!(seen.into_iter().all(|v| v));
        }
    }

    #[test]
    #[should_panic]
    fn transform_index_panics_on_invalid_symmetry() {
        let _ = transform_index(8, 0);
    }

    #[test]
    fn canonical_hash_matches_across_symmetric_positions() {
        let mut b1 = Board::new();
        b1.play(idx(2, 3));
        b1.play(idx(4, 7));
        b1.play(idx(10, 5));
        b1.play(idx(1, 14));

        let mut b2 = Board::new();
        for cell in 0..CELLS {
            let p = b1.cells[cell];
            if p == 0 {
                continue;
            }
            b2.cells[transform_index(1, cell)] = p;
        }
        b2.side = b1.side;
        b2.moves_played = b1.moves_played;
        b2.last = b1.last.map(|c| transform_index(1, c));

        assert_eq!(b1.tt_key(), b2.tt_key());
    }

    #[test]
    fn alpha_beta_detects_immediate_win() {
        let mut b = Board::new();
        for x in 0..4 {
            b.cells[idx(x, 0)] = BLACK;
        }
        b.side = BLACK;
        b.moves_played = 4;
        b.last = Some(idx(3, 0));
        assert_eq!(alpha_beta(&b, 1, -1, 1, 2), 1);
    }

    #[test]
    fn tactical_pick_prefers_forced_win() {
        let mut b = Board::new();
        for x in 0..4 {
            b.cells[idx(x, 0)] = BLACK;
        }
        b.side = BLACK;
        b.moves_played = 4;
        b.last = Some(idx(3, 0));

        let winning = idx(4, 0);
        let other = idx(7, 7);
        let candidates = vec![
            Candidate {
                mv: other,
                visits: 100,
                winrate: 0.9,
            },
            Candidate {
                mv: winning,
                visits: 10,
                winrate: 0.4,
            },
        ];
        assert_eq!(tactical_pick(&b, &candidates, 1, 2), Some(winning));
    }

    #[test]
    fn tactical_pick_avoids_forced_loss_when_block_exists() {
        let mut b = Board::new();
        for x in 0..4 {
            b.cells[idx(x, 0)] = WHITE;
        }
        b.side = BLACK;
        b.moves_played = 4;
        b.last = Some(idx(3, 0));

        let bad = idx(7, 7);
        let block = idx(4, 0);
        let candidates = vec![
            Candidate {
                mv: bad,
                visits: 100,
                winrate: 0.9,
            },
            Candidate {
                mv: block,
                visits: 10,
                winrate: 0.4,
            },
        ];
        assert_eq!(tactical_pick(&b, &candidates, 2, 2), Some(block));
    }

    #[test]
    fn tactical_pick_none_for_empty_or_zero_depth() {
        let b = Board::new();
        let candidates = vec![Candidate {
            mv: idx(7, 7),
            visits: 1,
            winrate: 0.5,
        }];
        assert_eq!(tactical_pick(&b, &[], 2, 2), None);
        assert_eq!(tactical_pick(&b, &candidates, 0, 2), None);
    }

    #[test]
    fn merged_root_candidates_includes_unexpanded_moves() {
        let mut b = Board::new();
        b.play(idx(7, 7));
        b.play(idx(7, 8));
        let ranked = vec![Candidate {
            mv: idx(6, 7),
            visits: 42,
            winrate: 0.5,
        }];
        let merged = merged_root_candidates(&b, &ranked);
        assert_eq!(merged.len(), b.legal_moves().len());
        assert_eq!(merged[0].mv, idx(6, 7));
        assert_eq!(merged[0].visits, 42);

        let missing_mv = idx(0, 0);
        let found = merged.iter().find(|c| c.mv == missing_mv).unwrap();
        assert_eq!(found.visits, 0);
        assert_eq!(found.winrate, 0.0);
    }

    #[test]
    fn alpha_beta_terminal_and_depth_zero_paths() {
        let mut term = Board::new();
        for x in 0..5 {
            term.cells[idx(x, 0)] = BLACK;
        }
        term.side = WHITE;
        term.moves_played = 5;
        term.last = Some(idx(4, 0));
        assert_eq!(alpha_beta(&term, 3, -1, 1, 2), -1);

        let b = Board::new();
        assert_eq!(alpha_beta(&b, 0, -1, 1, 2), 0);

        let mut draw = Board::new();
        draw.moves_played = CELLS;
        draw.last = None;
        assert_eq!(alpha_beta(&draw, 3, -1, 1, 2), 0);
    }

    #[test]
    fn rng_and_uct_behavior() {
        let mut rng = Rng::new(123);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
        let r = rng.gen_usize(7);
        assert!(r < 7);

        assert!(uct(0, 0, 0, 10).is_infinite());
        let s1 = uct(10, 20, 0, 40);
        let s2 = uct(12, 20, 0, 40);
        assert!(s2 > s1);
    }

    #[test]
    fn arena_helpers_and_evaluate() {
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_node(None, vec![0]))]));
        let child_idx = add_child(&arena, make_node(Some(BLACK), vec![]));
        assert_eq!(child_idx, 1);
        let _n = node_at(&arena, 1);

        // evaluate: empty board → draw
        assert_eq!(evaluate(&[0u8; CELLS]), 0);
        // evaluate: BLACK open four → BLACK wins
        let mut cells = [0u8; CELLS];
        for x in 1..5 {
            cells[idx(x, 0)] = BLACK;
        }
        assert_eq!(evaluate(&cells), BLACK);
    }

    #[test]
    fn threat_score_and_evaluate_patterns() {
        // Empty board: score 0 for both, evaluate returns draw (covers p_count=0 / `_` arm).
        assert_eq!(threat_score(&[0u8; CELLS], BLACK), 0);
        assert_eq!(evaluate(&[0u8; CELLS]), 0);

        // Single stone: several k=1 windows exist (covers p_count=1 arm).
        let mut c1 = [0u8; CELLS];
        c1[idx(7, 7)] = BLACK;
        assert!(threat_score(&c1, BLACK) > 0);

        // Split four XX_XX: BLACK at x=0,1,3,4 y=0.
        // Window x=0..4 has p_count=4 (covers p_count=4 arm).
        // Window x=1..5 has p_count=3 (covers p_count=3 arm).
        // Window x=2..6 has p_count=2 (covers p_count=2 arm).
        let mut cb = [0u8; CELLS];
        for x in [0usize, 1, 3, 4] { cb[idx(x, 0)] = BLACK; }
        let s_split = threat_score(&cb, BLACK);
        assert!(s_split >= 1_000, "xx_xx should contain at least one k=4 window: {s_split}");

        // Open four _XXXX_ at y=1, x=1..4: two k=4 windows → scores higher than split.
        let mut c4 = [0u8; CELLS];
        for x in 1..5 { c4[idx(x, 1)] = BLACK; }
        let s_open = threat_score(&c4, BLACK);
        assert!(s_open >= 2_000, "open four: two k=4 windows expected: {s_open}");
        assert!(s_open > s_split, "open four > split four: {s_open} vs {s_split}");

        // Opponent stone in window → that window skipped (covers has_opp branch).
        let mut co = [0u8; CELLS];
        for x in 1..5 { co[idx(x, 1)] = BLACK; }
        co[idx(0, 1)] = WHITE; // blocks the leftmost k=4 window
        let s_blocked = threat_score(&co, BLACK);
        assert!(s_blocked < s_open, "opponent blocks one window: {s_blocked} < {s_open}");

        // evaluate: returns correct player.
        assert_eq!(evaluate(&c4), BLACK);
        let mut cw = [0u8; CELLS];
        for x in 1..5 { cw[idx(x, 0)] = WHITE; }
        assert_eq!(evaluate(&cw), WHITE);
    }

    #[test]
    fn threat_score_fork_bonus() {
        // arm 1 (f >= 2 → +50_000): fours in two directions sharing a corner stone.
        let mut c_df = [0u8; CELLS];
        for x in 0..4 { c_df[idx(x, 0)] = BLACK; } // horizontal four
        for y in 1..4 { c_df[idx(0, y)] = BLACK; } // vertical four (shares (0,0))
        assert!(threat_score(&c_df, BLACK) >= 50_000, "double four: two directions");

        // arm 2 guard true (n_four=1, n_three>=1 → +10_000):
        // an open four's overlapping k=3 tail window sets three_dirs too.
        let mut c_ft = [0u8; CELLS];
        for x in 1..5 { c_ft[idx(x, 0)] = BLACK; }
        assert!(threat_score(&c_ft, BLACK) >= 10_000, "four+three bonus from open four");

        // arm 2 guard false (n_four=1, n_three=0): right end blocked by opponent,
        // only one window passes and it has k=4 — no k=3 tail windows survive.
        // Falls through arm-2 guard and arm-3 guard to arm 4 → 0 fork bonus.
        let mut c_hf = [0u8; CELLS];
        for x in 1..5 { c_hf[idx(x, 0)] = BLACK; }
        c_hf[idx(5, 0)] = WHITE;
        assert!(threat_score(&c_hf, BLACK) < 10_000, "right-blocked four: no fork bonus");

        // arm 3 (n_four=0, n_three>=2 → +5_000): threes in two directions, no fours.
        let mut c_dt = [0u8; CELLS];
        for x in 1..4 { c_dt[idx(x, 5)] = BLACK; } // horizontal three
        for y in 1..4 { c_dt[idx(7, y)] = BLACK; } // vertical three
        assert!(threat_score(&c_dt, BLACK) >= 5_000, "double three in two directions");

        // arm 4 (_ => 0): single stone, n_four=0, n_three=0 → no fork bonus.
        let mut c_one = [0u8; CELLS];
        c_one[idx(7, 7)] = BLACK;
        assert!(threat_score(&c_one, BLACK) < 5_000, "single stone: no fork bonus");
    }

    #[test]
    fn analyze_root_skips_zero_visit_children_for_best_choice() {
        let root = Node {
            visits: AtomicU64::new(10),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(idx(0, 0) as Move, 1), (idx(1, 1) as Move, 2)]),
        };
        let child_a = make_node(None, vec![]);
        let child_b = Node {
            visits: AtomicU64::new(4),
            win_halves: AtomicU64::new(6),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(RwLock::new(vec![
            Arc::new(root),
            Arc::new(child_a),
            Arc::new(child_b),
        ]));
        let analysis = analyze_root(&arena);
        assert_eq!(analysis.snapshot.best_mv_mcts, Some(idx(1, 1)));
        assert_eq!(analysis.snapshot.best_visits, 4);
        assert_eq!(analysis.snapshot.node_count, 3);
        assert_eq!(analysis.snapshot.total_visits, 10);
        assert_eq!(analysis.ranked.len(), 2);
        let best = analysis.ranked.first().unwrap();
        assert_eq!(best.mv, idx(1, 1));
        assert_eq!(best.visits, 4);
        assert_eq!(best.winrate, 0.75);
    }

    #[test]
    fn worker_stops_immediately_when_flag_set() {
        let board = Board::new();
        let cfg = test_cfg(Some(100), 2);
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_root_node(&board, cfg))]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(true));
        worker(0, board.clone(), board.side, stop, arena.clone(), tt, cfg);
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_respects_zero_iteration_limit() {
        let board = Board::new();
        let cfg = test_cfg(Some(0), 2);
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_root_node(&board, cfg))]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(0, board.clone(), board.side, stop, arena.clone(), tt, cfg);
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_runs_with_iteration_limit() {
        let board = Board::new();
        let cfg = test_cfg(Some(2), 2);
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_root_node(&board, cfg))]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(1, board.clone(), board.side, stop, arena.clone(), tt, cfg);
        let root_node = node_at(&arena, 0);
        assert!(root_node.visits.load(Ordering::Relaxed) >= 2);
        assert!(arena.read().unwrap().len() > 1);
    }

    #[test]
    fn worker_root_terminal_branch() {
        let board = Board::new();
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_node(Some(BLACK), vec![]))]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            2,
            board.clone(),
            BLACK,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 1);
        assert_eq!(root_node.win_halves.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn worker_expand_terminal_child_branch() {
        let mut board = Board::new();
        for x in 0..4 {
            board.cells[idx(x, 0)] = BLACK;
        }
        board.side = BLACK;
        board.moves_played = 4;
        board.last = Some(idx(3, 0));
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_node(None, vec![idx(4, 0) as Move]))]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            3,
            board.clone(),
            BLACK,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );
        assert!(arena.read().unwrap().len() >= 2);
    }

    #[test]
    fn worker_selection_virtual_loss_and_draw_reward() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(5),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(idx(0, 0) as Move, 1)]),
        };
        let child = Node {
            visits: AtomicU64::new(1),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: Some(0),
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(root), Arc::new(child)]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let mut b2 = board.clone();
        b2.play(idx(0, 0));
        tt.insert(b2.tt_key(), 1usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            4,
            board.clone(),
            BLACK,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );

        let root_node = node_at(&arena, 0);
        let child_node = node_at(&arena, 1);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 6);
        assert_eq!(root_node.win_halves.load(Ordering::Relaxed), 1);
        assert_eq!(child_node.virtual_loss.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_dead_end_rollout_branch() {
        let board = Board::new();
        let arena: Arena = Arc::new(RwLock::new(vec![Arc::new(make_node(None, vec![]))]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            5,
            board.clone(),
            board.side,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_expansion_uses_tt_cache_hit() {
        let board = Board::new();
        let mut child_board = board.clone();
        child_board.play(idx(0, 0));
        let arena: Arena = Arc::new(RwLock::new(vec![
            Arc::new(make_node(None, vec![idx(0, 0) as Move])),
            Arc::new(make_node(child_board.terminal(), child_board.legal_moves())),
        ]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        tt.insert(child_board.tt_key(), 1usize);

        let stop = Arc::new(AtomicBool::new(false));
        worker(
            6,
            board.clone(),
            board.side,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );

        let root_node = node_at(&arena, 0);
        let kids = root_node.children.lock().unwrap().clone();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0], (idx(0, 0) as Move, 1usize));
        assert_eq!(arena.read().unwrap().len(), 2);
    }

    #[test]
    fn run_handles_invalid_and_terminal_args() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "0");
        env::set_var(ENV_ITERS, "1");
        assert_eq!(run(&["15,0".to_string()]), 1);

        let mut moves = Vec::new();
        for x in 0..5 {
            moves.push(format!("{x},0"));
            if x < 4 {
                moves.push(format!("{x},1"));
            }
        }
        assert_eq!(run(&moves), 1);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
    }

    #[test]
    fn run_executes_with_iteration_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::remove_var(ENV_SECONDS);
        env::set_var(ENV_ITERS, "1");
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_ITERS);
    }

    #[test]
    fn run_executes_with_zero_iteration_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "0");
        env::set_var(ENV_ITERS, "0");
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
    }

    #[test]
    fn run_executes_with_seconds_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "1");
        env::remove_var(ENV_ITERS);
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
    }

    #[test]
    fn run_executes_with_seconds_before_iteration_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "1");
        env::set_var(ENV_ITERS, "1000000");
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
    }

    #[test]
    fn run_executes_done_false_path_before_seconds_break() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "2");
        env::set_var(ENV_ITERS, "1000000000");
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
    }

    #[test]
    fn run_executes_early_stop_with_custom_thresholds() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "60");
        env::set_var(ENV_EARLY_STOP_RATIO, "0.0");
        env::set_var(ENV_EARLY_STOP_MIN_VISITS, "0");
        env::remove_var(ENV_ITERS);
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_EARLY_STOP_RATIO);
        env::remove_var(ENV_EARLY_STOP_MIN_VISITS);
    }

    #[test]
    fn run_executes_with_tactical_disabled() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "1");
        env::set_var(ENV_TACTICAL_DEPTH, "0");
        env::set_var(ENV_TACTICAL_TOPK, "0");
        env::remove_var(ENV_ITERS);
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_TACTICAL_DEPTH);
        env::remove_var(ENV_TACTICAL_TOPK);
    }

    #[test]
    fn run_fallback_move_when_no_mcts_and_no_tactical() {
        // With 0 iterations and tactical disabled, analyse_and_pick returns None
        // for both the in-loop and post-loop calls, exercising the unwrap_or_else
        // fallback on both lines. debug=1 also covers the in-loop debug branch.
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "0");
        env::set_var(ENV_ITERS, "0");
        env::set_var(ENV_TACTICAL_DEPTH, "0");
        env::set_var(ENV_TACTICAL_TOPK, "0");
        env::set_var(ENV_DEBUG, "1");
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
        env::remove_var(ENV_TACTICAL_DEPTH);
        env::remove_var(ENV_TACTICAL_TOPK);
        env::remove_var(ENV_DEBUG);
    }

    #[test]
    fn run_presearch_instant_win_and_block() {
        let _g = ENV_LOCK.lock().unwrap();
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);

        // White has 4-in-a-row at y=0; it is white's turn (9 moves played).
        // Black stones are scattered so no 5-in-a-row is formed.
        // Covers the else{BLACK} opp branch and the forced Some branch.
        let args = vec![
            "0,14".to_string(), "0,0".to_string(),
            "14,0".to_string(), "1,0".to_string(),
            "0,13".to_string(), "2,0".to_string(),
            "14,1".to_string(), "3,0".to_string(),
            "0,12".to_string(),
        ];
        assert_eq!(run(&args), 0);

        // Black has 4-in-a-row at y=1; it is black's turn (8 moves played).
        // Covers wins_at for board.side finding the win immediately.
        let args2 = vec![
            "0,1".to_string(), "0,2".to_string(),
            "1,1".to_string(), "1,2".to_string(),
            "2,1".to_string(), "2,2".to_string(),
            "3,1".to_string(), "3,2".to_string(),
        ];
        assert_eq!(run(&args2), 0);
    }

    #[test]
    fn run_executes_with_debug_logging_enabled() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "1");
        env::set_var(ENV_DEBUG, "1");
        env::remove_var(ENV_ITERS);
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_DEBUG);
    }

    #[test]
    fn run_executes_with_invalid_local_radius_env() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "1");
        env::set_var(ENV_LOCAL_RADIUS, "nope");
        env::remove_var(ENV_ITERS);
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_LOCAL_RADIUS);
    }

    #[test]
    fn engine_config_parses_seed_and_invalid_seed() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SEED, "123");
        assert_eq!(EngineConfig::from_env().seed_base, Some(123));
        env::set_var(ENV_SEED, "nope");
        assert_eq!(EngineConfig::from_env().seed_base, None);
        env::remove_var(ENV_SEED);
    }

    #[test]
    fn run_executes_opening_center_path() {
        let _g = ENV_LOCK.lock().unwrap();
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
        env::set_var(ENV_DEBUG, "1");
        assert_eq!(run(&[]), 0);
        env::remove_var(ENV_DEBUG);
    }

    #[test]
    fn run_executes_opening_center_path_without_debug() {
        let _g = ENV_LOCK.lock().unwrap();
        env::remove_var(ENV_SECONDS);
        env::remove_var(ENV_ITERS);
        env::remove_var(ENV_DEBUG);
        assert_eq!(run(&[]), 0);
    }

    #[test]
    fn run_seconds_zero_without_iters_uses_default_not_hang() {
        // MCTS_SECONDS=0 without MCTS_ITERS has no termination condition.
        // The guard in run() must fall back to the default time limit.
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var(ENV_SECONDS, "0");
        env::remove_var(ENV_ITERS);
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var(ENV_SECONDS);
    }

    // ── Correctness invariants that code-coverage alone cannot catch ──────────

    /// When root_player wins and the search path is 3 nodes deep:
    ///   depth 0 (root)   → root_player's perspective → BACKPROP_WIN
    ///   depth 1 (child1) → chosen by root_player     → BACKPROP_WIN
    ///   depth 2 (child2) → chosen by opponent        → BACKPROP_LOSS
    /// The old non-alternating code gave BACKPROP_WIN to *every* node, which
    /// made the opponent cooperate with the root player during UCT selection.
    #[test]
    fn backprop_alternates_reward_root_player_wins() {
        let board = Board::new();
        let mv1 = idx(7, 7) as Move;
        let mv2 = idx(7, 8) as Move;
        let root = Node {
            visits: AtomicU64::new(1),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(mv1, 1)]),
        };
        let child1 = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(mv2, 2)]),
        };
        let arena: Arena = Arc::new(RwLock::new(vec![
            Arc::new(root),
            Arc::new(child1),
            Arc::new(make_node(Some(BLACK), vec![])),
        ]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            20,
            board.clone(),
            BLACK,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );
        assert_eq!(
            node_at(&arena, 0).win_halves.load(Ordering::Relaxed),
            BACKPROP_WIN,
            "depth 0: root player's win should be BACKPROP_WIN"
        );
        assert_eq!(
            node_at(&arena, 1).win_halves.load(Ordering::Relaxed),
            BACKPROP_WIN,
            "depth 1: chosen by root_player, should be BACKPROP_WIN"
        );
        assert_eq!(
            node_at(&arena, 2).win_halves.load(Ordering::Relaxed),
            BACKPROP_LOSS,
            "depth 2: chosen by opponent, should be BACKPROP_LOSS (opponent lost)"
        );
    }

    /// Mirror of the above: when the *opponent* wins the rewards must flip.
    ///   depth 0 (root)   → BACKPROP_LOSS
    ///   depth 1 (child1) → BACKPROP_LOSS
    ///   depth 2 (child2) → BACKPROP_WIN  (opponent's perspective: they won)
    #[test]
    fn backprop_alternates_reward_opponent_wins() {
        let board = Board::new();
        let mv1 = idx(7, 7) as Move;
        let mv2 = idx(7, 8) as Move;
        let root = Node {
            visits: AtomicU64::new(1),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(mv1, 1)]),
        };
        let child1 = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(mv2, 2)]),
        };
        let arena: Arena = Arc::new(RwLock::new(vec![
            Arc::new(root),
            Arc::new(child1),
            Arc::new(make_node(Some(WHITE), vec![])),
        ]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            21,
            board.clone(),
            BLACK,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );
        assert_eq!(
            node_at(&arena, 0).win_halves.load(Ordering::Relaxed),
            BACKPROP_LOSS,
            "depth 0: opponent won, root sees BACKPROP_LOSS"
        );
        assert_eq!(
            node_at(&arena, 1).win_halves.load(Ordering::Relaxed),
            BACKPROP_LOSS,
            "depth 1: chosen by root_player, opponent won → BACKPROP_LOSS"
        );
        assert_eq!(
            node_at(&arena, 2).win_halves.load(Ordering::Relaxed),
            BACKPROP_WIN,
            "depth 2: chosen by opponent, opponent won → BACKPROP_WIN"
        );
    }

    /// Draw outcome at an even-depth (opponent's) node must record BACKPROP_DRAW
    /// from the opponent's perspective — the branch `outcome == 0` at depth > 0 && depth % 2 == 0.
    #[test]
    fn backprop_draw_at_opponent_depth() {
        let board = Board::new();
        let mv1 = idx(7, 7) as Move;
        let mv2 = idx(7, 8) as Move;
        let root = Node {
            visits: AtomicU64::new(1),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(mv1, 1)]),
        };
        let child1 = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(mv2, 2)]),
        };
        let arena: Arena = Arc::new(RwLock::new(vec![
            Arc::new(root),
            Arc::new(child1),
            // Draw terminal at depth 2 (chosen by opponent).
            Arc::new(make_node(Some(0), vec![])),
        ]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(
            30,
            board.clone(),
            BLACK,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(1), 2),
        );
        assert_eq!(
            node_at(&arena, 2).win_halves.load(Ordering::Relaxed),
            BACKPROP_DRAW,
            "depth 2: chosen by opponent, draw → BACKPROP_DRAW from opponent's perspective"
        );
    }

    /// Two different untried moves that the TT maps to the same child node
    /// must not produce duplicate entries in the parent's children list.
    /// The old code pushed unconditionally; the guard prevents the second push.
    #[test]
    fn worker_no_duplicate_children_when_tt_returns_same_node() {
        let board = Board::new();
        let mv1 = idx(3, 3) as Move;
        let mv2 = idx(4, 4) as Move;
        // The single shared child that both moves will resolve to via the TT.
        let arena: Arena = Arc::new(RwLock::new(vec![
            Arc::new(make_node(None, vec![mv1, mv2])),
            Arc::new(make_node(None, board.local_moves(2))),
        ]));
        let tt: Tt = tt_with_root(board.tt_key(), 0usize);
        // Force both move-boards to resolve to node 1 in the TT.
        let mut b1 = board.clone();
        b1.play(mv1 as usize);
        let mut b2 = board.clone();
        b2.play(mv2 as usize);
        tt.insert(b1.tt_key(), 1usize);
        tt.insert(b2.tt_key(), 1usize);

        let stop = Arc::new(AtomicBool::new(false));
        worker(
            22,
            board.clone(),
            board.side,
            stop,
            arena.clone(),
            tt,
            test_cfg(Some(2), 2),
        );

        let kids = node_at(&arena, 0).children.lock().unwrap().clone();
        let unique_ids: std::collections::HashSet<_> = kids.iter().map(|&(_, id)| id).collect();
        assert_eq!(
            kids.len(),
            unique_ids.len(),
            "children list must not contain duplicate node IDs"
        );
    }

    #[test]
    fn wins_at_detects_five_in_a_row_and_blocks() {
        let mut cells = [0u8; CELLS];
        // Place BLACK at (0,0),(1,0),(2,0),(3,0) — four in a row.
        for x in 0..4 {
            cells[idx(x, 0)] = BLACK;
        }
        // Playing at (4,0) wins for BLACK.
        assert!(wins_at(&cells, BLACK, idx(4, 0)));
        // Playing at (5,0) does NOT win (only 3 adjacent on the other side).
        assert!(!wins_at(&cells, BLACK, idx(5, 0)));
        // Playing at (4,0) does NOT win for WHITE (those are BLACK stones).
        assert!(!wins_at(&cells, WHITE, idx(4, 0)));
        // Board with empty (4,0) — blocking at (4,0) prevents BLACK's win.
        // wins_at confirms where the threat is.
        cells[idx(4, 0)] = WHITE; // block
        assert!(!wins_at(&cells, BLACK, idx(5, 0)));
    }
}
