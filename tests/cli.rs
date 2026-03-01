use std::process::Command;

fn bin_path() -> String {
    option_env!("CARGO_BIN_EXE_mcts-gomoku")
        .or(option_env!("CARGO_BIN_EXE_mcts_gomoku"))
        .expect("missing test binary path from cargo")
        .to_string()
}

#[test]
fn cli_exits_nonzero_on_invalid_move() {
    let bin = bin_path();
    let status = Command::new(bin)
        .arg("15,0")
        .status()
        .expect("failed to run binary");
    assert!(!status.success());
}

#[test]
fn cli_exits_zero_with_iteration_cap() {
    let bin = bin_path();
    let status = Command::new(bin)
        .env("MCTS_ITERS", "1")
        .arg("7,7")
        .status()
        .expect("failed to run binary");
    assert!(status.success());
}
