//! Standalone VNTS Native CLI Binary.
//! Zero Python, Zero GPU, Zero PyTorch.

use std::env;
use std::time::Instant;
use vnts_core::{Cholesky, Matrix, NativeMicroLLM, NativeOrthogonalMemory, QR};

fn run_native_benchmarks() {
    println!("=================================================================");
    println!("VNTS Native Rust Engine: Pure Bare-Metal Performance Benchmark");
    println!("Hardware: Commodity CPU (Zero Python GIL, Zero PyTorch, Zero GPU)");
    println!("=================================================================");

    // 1. Cholesky Primal Solve (2000 x 256)
    let n = 2000;
    let d = 256;
    let k = 10;
    let x = Matrix::random(n, d, 1.0);
    let y = Matrix::random(n, k, 1.0);

    let t0 = Instant::now();
    let w = Cholesky::solve_ridge(&x, &y, 1e-3);
    let cholesky_time = t0.elapsed();
    println!(
        "1. Cholesky Ridge Solve ({}x{} -> {}): {:?} (Throughput: {:.1} samples/s)",
        n, d, k, cholesky_time, n as f64 / cholesky_time.as_secs_f64()
    );
    assert_eq!((w.rows, w.cols), (d, k));

    // 2. QR Orthonormal Basis Extraction (256 x 100)
    let h = Matrix::random(100, d, 1.0);
    let t0_qr = Instant::now();
    let q = QR::row_space_basis(&h, 1e-4);
    let qr_time = t0_qr.elapsed();
    println!("2. QR Basis Extraction ({} features -> rank {}): {:?}", d, q.cols, qr_time);

    // 3. Continual Memory Task Learning (2 sequential tasks)
    let mut mem = NativeOrthogonalMemory::new(d, 10, 1e-3);
    let t0_mem = Instant::now();
    let rank1 = mem.learn_task(&x, &y);
    let x2 = Matrix::random(500, d, 1.0);
    let y2 = Matrix::random(500, 10, 1.0);
    let rank2 = mem.learn_task(&x2, &y2);
    let mem_time = t0_mem.elapsed();
    println!("3. Continual Lifelong Memory (2 Tasks, Final Rank {}): {:?}", rank2, mem_time);

    // 4. Native Causal MicroLLM Text Learning
    let corpus = "The Vector Nullspace Transformation System is a native, non-backpropagation deep learning technology. "
        .repeat(20);
    let t0_llm = Instant::now();
    let mut llm = NativeMicroLLM::from_corpus(&corpus, 64, 4, 1e-3);
    let (acc, tokens) = llm.train(&corpus, 16);
    let llm_time = t0_llm.elapsed();
    println!(
        "4. Native MicroLLM Text Learning ({} tokens): {:?} (Accuracy: {:.2}%, {:.1} tokens/s)",
        tokens, llm_time, acc * 100.0, tokens as f64 / llm_time.as_secs_f64()
    );

    let sample = llm.generate("Vector Nullspace", 40, 0.7);
    println!("   Sample Generated: {:?}", sample);

    println!("=================================================================");
    println!("All Native Rust Benchmarks Complete Successfully!");
    println!("=================================================================");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "bench" {
        run_native_benchmarks();
    } else if args[1] == "train" && args.len() >= 3 {
        let text = std::fs::read_to_string(&args[2]).expect("Failed to read text file");
        let mut llm = NativeMicroLLM::from_corpus(&text, 64, 4, 1e-3);
        let (acc, tokens) = llm.train(&text, 16);
        println!("Trained on {} tokens. Next-token accuracy: {:.2}%", tokens, acc * 100.0);
        let gen = llm.generate("The", 50, 0.7);
        println!("Generated: {}", gen);
    } else {
        println!("Usage: vnts-cli [bench | train <file.txt>]");
    }
}
