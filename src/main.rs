use std::env;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const N: usize = 15;
const CELLS: usize = N * N;
const BLACK: u8 = 1;
const WHITE: u8 = 2;
const DIRS: [(isize, isize); 4] = [(1, 0), (0, 1), (1, 1), (1, -1)];

#[derive(Clone)]
struct Board {
    cells: [u8; CELLS],
    side: u8,
    moves_played: usize,
    last: Option<usize>,
    hash: u64,
}

impl Board {
    fn new() -> Self {
        Self {
            cells: [0; CELLS],
            side: BLACK,
            moves_played: 0,
            last: None,
            hash: 0,
        }
    }

    fn legal_moves(&self) -> Vec<u16> {
        let mut out = Vec::with_capacity(CELLS - self.moves_played);
        for i in 0..CELLS {
            if self.cells[i] == 0 {
                out.push(i as u16);
            }
        }
        out
    }

    fn play(&mut self, mv: usize) {
        self.cells[mv] = self.side;
        self.hash ^= piece_hash(mv, self.side);
        self.hash ^= SIDE_HASH;
        self.moves_played += 1;
        self.last = Some(mv);
        self.side = if self.side == BLACK { WHITE } else { BLACK };
    }

    fn winner(&self) -> Option<u8> {
        let last = self.last?;
        let x = (last % N) as isize;
        let y = (last / N) as isize;
        let p = self.cells[last];
        for (dx, dy) in DIRS {
            let mut count = 1;
            let mut i = 1;
            while Self::on_board(x + dx * i, y + dy * i)
                && self.cells[((y + dy * i) as usize) * N + (x + dx * i) as usize] == p
            {
                count += 1;
                i += 1;
            }
            i = 1;
            while Self::on_board(x - dx * i, y - dy * i)
                && self.cells[((y - dy * i) as usize) * N + (x - dx * i) as usize] == p
            {
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
    untried: Mutex<Vec<u16>>,
    children: Mutex<Vec<(u16, usize)>>,
}

type Arena = Arc<Mutex<Vec<Arc<Node>>>>;
type Tt = Arc<TtSharded>;

const SIDE_HASH: u64 = 0x9E37_79B9_7F4A_7C15;
const TT_SHARDS: usize = 64;

struct TtSharded {
    shards: Vec<Mutex<HashMap<u64, usize>>>,
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

    fn get_or_insert_with<F: FnOnce() -> usize>(&self, hash: u64, create: F) -> usize {
        let mut shard = self.shards[Self::shard_idx(hash)].lock().unwrap();
        if let Some(&idx) = shard.get(&hash) {
            idx
        } else {
            let idx = create();
            shard.insert(hash, idx);
            idx
        }
    }

    fn insert(&self, hash: u64, idx: usize) {
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
    let u = 1.4142135 * ((parent_visits.max(1) as f32).ln() / v as f32).sqrt();
    q + u
}

fn parse_move(s: &str) -> Option<usize> {
    let (a, b) = s.split_once(',')?;
    let x: usize = a.parse().ok()?;
    let y: usize = b.parse().ok()?;
    if x < N && y < N {
        Some(y * N + x)
    } else {
        None
    }
}

fn move_to_xy(mv: usize) -> (usize, usize) {
    (mv % N, mv / N)
}

fn node_at(arena: &Arena, idx: usize) -> Arc<Node> {
    arena.lock().unwrap()[idx].clone()
}

fn add_child(arena: &Arena, node: Node) -> usize {
    let mut a = arena.lock().unwrap();
    a.push(Arc::new(node));
    a.len() - 1
}

fn tt_with_root(hash: u64, idx: usize) -> Tt {
    let tt = Arc::new(TtSharded::new());
    tt.insert(hash, idx);
    tt
}

fn rollout(mut board: Board, rng: &mut Rng) -> u8 {
    let mut outcome = board.terminal();
    while outcome.is_none() {
        let legal = board.legal_moves();
        let mv = legal[rng.gen_usize(legal.len())] as usize;
        board.play(mv);
        outcome = board.terminal();
    }
    outcome.unwrap()
}

fn best_root_child(arena: &Arena) -> (Option<usize>, u64, f64, usize, u64) {
    let root_node = node_at(arena, 0);
    let kids = root_node.children.lock().unwrap().clone();
    let node_count = arena.lock().unwrap().len();
    let total_visits = root_node.visits.load(Ordering::Relaxed);

    let mut best_mv = None::<usize>;
    let mut best_visits = 0u64;
    let mut best_wr = 0.0f64;

    for (mv_u16, c) in kids {
        let cn = node_at(arena, c);
        let v = cn.visits.load(Ordering::Relaxed);
        if v == 0 {
            continue;
        }
        if v > best_visits {
            best_visits = v;
            let w = cn.win_halves.load(Ordering::Relaxed);
            best_wr = w as f64 / (2.0 * v as f64);
            best_mv = Some(mv_u16 as usize);
        }
    }

    (best_mv, best_visits, best_wr, node_count, total_visits)
}

fn worker(
    id: usize,
    root: Board,
    root_player: u8,
    stop: Arc<AtomicBool>,
    arena: Arena,
    tt: Tt,
    max_iters: Option<u64>,
) {
    let seed = Instant::now().elapsed().as_nanos() as u64 ^ ((id as u64) << 32) ^ 0x9E3779B97F4A7C15;
    let mut rng = Rng::new(seed);
    let mut iters = 0u64;

    while !stop.load(Ordering::Relaxed) {
        if let Some(limit) = max_iters {
            if iters >= limit {
                break;
            }
        }
        iters += 1;
        let mut board = root.clone();
        let mut path = Vec::<usize>::with_capacity(256);
        let mut vl_path = Vec::<usize>::with_capacity(256);
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
                let child_hash = board.hash;
                let child_idx = tt.get_or_insert_with(child_hash, || {
                    let child = Node {
                        visits: AtomicU64::new(0),
                        win_halves: AtomicU64::new(0),
                        virtual_loss: AtomicU64::new(0),
                        terminal: child_terminal,
                        untried: Mutex::new(board.legal_moves()),
                        children: Mutex::new(Vec::new()),
                    };
                    add_child(&arena, child)
                });
                node.children.lock().unwrap().push((mv as u16, child_idx));
                path.push(child_idx);

                if let Some(t) = child_terminal {
                    break t;
                }
                break rollout(board, &mut rng);
            }

            let kids = node.children.lock().unwrap().clone();
            if kids.is_empty() {
                break rollout(board, &mut rng);
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

        let reward = if outcome == root_player {
            2
        } else if outcome == 0 {
            1
        } else {
            0
        };

        for idx in path {
            let n = node_at(&arena, idx);
            n.visits.fetch_add(1, Ordering::Relaxed);
            n.win_halves.fetch_add(reward, Ordering::Relaxed);
        }

        for idx in vl_path {
            let n = node_at(&arena, idx);
            n.virtual_loss.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

fn run(args: &[String]) -> i32 {
    let mut board = Board::new();

    for a in args {
        match parse_move(a) {
            Some(mv) if board.cells[mv] == 0 => board.play(mv),
            _ => {
                eprintln!("invalid move: {a} (expected x,y in 0..14 and legal)");
                return 1;
            }
        }
    }

    if board.terminal().is_some() {
        eprintln!("position is already terminal");
        return 1;
    }

    let root = Node {
        visits: AtomicU64::new(0),
        win_halves: AtomicU64::new(0),
        virtual_loss: AtomicU64::new(0),
        terminal: board.terminal(),
        untried: Mutex::new(board.legal_moves()),
        children: Mutex::new(Vec::new()),
    };

    let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
    let tt: Tt = tt_with_root(board.hash, 0usize);
    let root_player = board.side;
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let stop = Arc::new(AtomicBool::new(false));

    let seconds = env::var("MCTS_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);
    let max_iters = env::var("MCTS_ITERS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    let early_stop_ratio = env::var("MCTS_EARLY_STOP_RATIO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.98);
    let early_stop_min_visits = env::var("MCTS_EARLY_STOP_MIN_VISITS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10_000);

    let mut handles = Vec::with_capacity(threads);
    for i in 0..threads {
        let stop_c = stop.clone();
        let arena_c = arena.clone();
        let tt_c = tt.clone();
        let root_c = board.clone();
        let max_iters_c = max_iters;
        handles.push(thread::spawn(move || {
            worker(i, root_c, root_player, stop_c, arena_c, tt_c, max_iters_c);
        }));
    }

    let start = Instant::now();
    loop {
        thread::sleep(Duration::from_secs(1));

        let (best_mv, best_visits, best_wr, node_count, total_visits) = best_root_child(&arena);

        if let Some(mv) = best_mv {
            let (x, y) = move_to_xy(mv);
            println!(
                "best={} {}, visits={}, winrate={:.4}, elapsed={}s, threads={}, nodes={} (shared tree, unbounded growth)",
                x,
                y,
                best_visits,
                best_wr,
                start.elapsed().as_secs(),
                threads,
                node_count
            );
        }

        if seconds > 0 && start.elapsed().as_secs() >= seconds {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        if seconds > 0
            && best_visits >= early_stop_min_visits
            && total_visits > 0
            && (best_visits as f64 / total_visits as f64) >= early_stop_ratio
        {
            stop.store(true, Ordering::Relaxed);
            break;
        }
        let done = max_iters
            .map(|_| handles.iter().all(|h| h.is_finished()))
            .unwrap_or(false);
        if done {
            break;
        }
    }

    for h in handles {
        let _ = h.join();
    }
    0
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let code = run(&args);
    if code != 0 {
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    fn idx(x: usize, y: usize) -> usize {
        y * N + x
    }

    #[test]
    fn board_new_and_legal_moves() {
        let b = Board::new();
        assert_eq!(b.side, BLACK);
        assert_eq!(b.moves_played, 0);
        assert_eq!(b.last, None);
        assert_eq!(b.legal_moves().len(), CELLS);
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
    fn arena_helpers_and_rollout_terminal() {
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(vec![0]),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let child_idx = add_child(
            &arena,
            Node {
                visits: AtomicU64::new(0),
                win_halves: AtomicU64::new(0),
                virtual_loss: AtomicU64::new(0),
                terminal: Some(BLACK),
                untried: Mutex::new(Vec::new()),
                children: Mutex::new(Vec::new()),
            },
        );
        assert_eq!(child_idx, 1);
        let _n = node_at(&arena, 1);

        let mut b = Board::new();
        b.cells[idx(0, 0)] = BLACK;
        b.cells[idx(1, 0)] = BLACK;
        b.cells[idx(2, 0)] = BLACK;
        b.cells[idx(3, 0)] = BLACK;
        b.cells[idx(4, 0)] = BLACK;
        b.last = Some(idx(2, 0));
        let mut rng = Rng::new(1);
        assert_eq!(rollout(b, &mut rng), BLACK);
    }

    #[test]
    fn best_root_child_skips_zero_visit_children() {
        let root = Node {
            visits: AtomicU64::new(10),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(vec![(idx(0, 0) as u16, 1), (idx(1, 1) as u16, 2)]),
        };
        let child_a = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let child_b = Node {
            visits: AtomicU64::new(4),
            win_halves: AtomicU64::new(6),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root), Arc::new(child_a), Arc::new(child_b)]));
        let (best_mv, best_visits, best_wr, node_count, _total_visits) = best_root_child(&arena);
        assert_eq!(best_mv, Some(idx(1, 1)));
        assert_eq!(best_visits, 4);
        assert_eq!(best_wr, 0.75);
        assert_eq!(node_count, 3);
    }

    #[test]
    fn worker_stops_immediately_when_flag_set() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: board.terminal(),
            untried: Mutex::new(board.legal_moves()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let stop = Arc::new(AtomicBool::new(true));
        worker(0, board.clone(), board.side, stop, arena.clone(), tt, Some(100));
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_respects_zero_iteration_limit() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: board.terminal(),
            untried: Mutex::new(board.legal_moves()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(0, board.clone(), board.side, stop, arena.clone(), tt, Some(0));
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_runs_with_iteration_limit() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: board.terminal(),
            untried: Mutex::new(board.legal_moves()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(1, board.clone(), board.side, stop, arena.clone(), tt, Some(2));
        let root_node = node_at(&arena, 0);
        assert!(root_node.visits.load(Ordering::Relaxed) >= 2);
        assert!(arena.lock().unwrap().len() > 1);
    }

    #[test]
    fn worker_root_terminal_branch() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: Some(BLACK),
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(2, board.clone(), BLACK, stop, arena.clone(), tt, Some(1));
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
        board.hash = 0;
        for x in 0..4 {
            board.hash ^= piece_hash(idx(x, 0), BLACK);
        }

        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(vec![idx(4, 0) as u16]),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(3, board.clone(), BLACK, stop, arena.clone(), tt, Some(1));
        assert!(arena.lock().unwrap().len() >= 2);
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
            children: Mutex::new(vec![(idx(0, 0) as u16, 1)]),
        };
        let child = Node {
            visits: AtomicU64::new(1),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: Some(0),
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root), Arc::new(child)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let mut b2 = board.clone();
        b2.play(idx(0, 0));
        tt.insert(b2.hash, 1usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(4, board.clone(), BLACK, stop, arena.clone(), tt, Some(1));

        let root_node = node_at(&arena, 0);
        let child_node = node_at(&arena, 1);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 6);
        assert_eq!(root_node.win_halves.load(Ordering::Relaxed), 1);
        assert_eq!(child_node.virtual_loss.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn worker_dead_end_rollout_branch() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(Vec::new()),
            children: Mutex::new(Vec::new()),
        };
        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        let stop = Arc::new(AtomicBool::new(false));
        worker(5, board.clone(), board.side, stop, arena.clone(), tt, Some(1));
        let root_node = node_at(&arena, 0);
        assert_eq!(root_node.visits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn worker_expansion_uses_tt_cache_hit() {
        let board = Board::new();
        let root = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: None,
            untried: Mutex::new(vec![idx(0, 0) as u16]),
            children: Mutex::new(Vec::new()),
        };

        let mut child_board = board.clone();
        child_board.play(idx(0, 0));
        let child = Node {
            visits: AtomicU64::new(0),
            win_halves: AtomicU64::new(0),
            virtual_loss: AtomicU64::new(0),
            terminal: child_board.terminal(),
            untried: Mutex::new(child_board.legal_moves()),
            children: Mutex::new(Vec::new()),
        };

        let arena: Arena = Arc::new(Mutex::new(vec![Arc::new(root), Arc::new(child)]));
        let tt: Tt = tt_with_root(board.hash, 0usize);
        tt.insert(child_board.hash, 1usize);

        let stop = Arc::new(AtomicBool::new(false));
        worker(6, board.clone(), board.side, stop, arena.clone(), tt, Some(1));

        let root_node = node_at(&arena, 0);
        let kids = root_node.children.lock().unwrap().clone();
        assert_eq!(kids.len(), 1);
        assert_eq!(kids[0], (idx(0, 0) as u16, 1usize));
        assert_eq!(arena.lock().unwrap().len(), 2);
    }

    #[test]
    fn run_handles_invalid_and_terminal_args() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("MCTS_SECONDS", "0");
        env::set_var("MCTS_ITERS", "1");
        assert_eq!(run(&["15,0".to_string()]), 1);

        let mut moves = Vec::new();
        for x in 0..5 {
            moves.push(format!("{x},0"));
            if x < 4 {
                moves.push(format!("{x},1"));
            }
        }
        assert_eq!(run(&moves), 1);
        env::remove_var("MCTS_SECONDS");
        env::remove_var("MCTS_ITERS");
    }

    #[test]
    fn run_executes_with_iteration_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::remove_var("MCTS_SECONDS");
        env::set_var("MCTS_ITERS", "1");
        let args = vec!["7,7".to_string(), "7,8".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var("MCTS_ITERS");
    }

    #[test]
    fn run_executes_with_zero_iteration_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::remove_var("MCTS_SECONDS");
        env::set_var("MCTS_ITERS", "0");
        let args = vec!["7,7".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var("MCTS_ITERS");
    }

    #[test]
    fn run_executes_with_seconds_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("MCTS_SECONDS", "1");
        env::remove_var("MCTS_ITERS");
        let args = vec!["7,7".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var("MCTS_SECONDS");
    }

    #[test]
    fn run_executes_with_seconds_before_iteration_cap() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("MCTS_SECONDS", "1");
        env::set_var("MCTS_ITERS", "1000000");
        let args = vec!["7,7".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var("MCTS_SECONDS");
        env::remove_var("MCTS_ITERS");
    }

    #[test]
    fn run_executes_done_false_path_before_seconds_break() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("MCTS_SECONDS", "2");
        env::set_var("MCTS_ITERS", "1000000000");
        let args = vec!["7,7".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var("MCTS_SECONDS");
        env::remove_var("MCTS_ITERS");
    }

    #[test]
    fn run_executes_early_stop_with_custom_thresholds() {
        let _g = ENV_LOCK.lock().unwrap();
        env::set_var("MCTS_SECONDS", "60");
        env::set_var("MCTS_EARLY_STOP_RATIO", "0.0");
        env::set_var("MCTS_EARLY_STOP_MIN_VISITS", "0");
        env::remove_var("MCTS_ITERS");
        let args = vec!["7,7".to_string()];
        assert_eq!(run(&args), 0);
        env::remove_var("MCTS_SECONDS");
        env::remove_var("MCTS_EARLY_STOP_RATIO");
        env::remove_var("MCTS_EARLY_STOP_MIN_VISITS");
    }
}
