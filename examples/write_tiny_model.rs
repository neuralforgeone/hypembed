//! Write the tiny test model fixture to a directory.
//!
//! ```bash
//! cargo run --example write_tiny_model -- ./path/to/model
//! ```

use std::path::PathBuf;

mod common {
    include!("../tests/common/mod.rs");
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("Usage: write_tiny_model <output_directory>");
            std::process::exit(1);
        });

    common::write_tiny_model_dir(&dir);
    println!("wrote tiny model to {}", dir.display());
}