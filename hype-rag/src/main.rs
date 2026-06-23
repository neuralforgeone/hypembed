use std::path::PathBuf;

use clap::{Parser, Subcommand};
use hype_rag::{index_directory, search_chunks, text_boundary, ChunkStore, HypeRagError};
use hypembed::Embedder;

#[derive(Parser)]
#[command(name = "hype-rag", about = "Local RAG CLI powered by hypembed")]
struct Cli {
    /// Data directory for index and config (default: .hype-rag)
    #[arg(long, global = true, default_value = ".hype-rag")]
    data_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize or register a local model directory
    Init {
        /// Path to model directory (config.json, vocab.txt, model.safetensors)
        #[arg(long)]
        model_dir: PathBuf,
    },
    /// Index markdown and text files
    Index {
        /// File or directory to index
        path: PathBuf,
        /// Recurse into subdirectories
        #[arg(long)]
        recursive: bool,
        /// Override model directory (otherwise uses value from init)
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// Semantic search over indexed chunks
    Search {
        /// Query string
        query: String,
        /// Number of results to return
        #[arg(long, default_value_t = 5)]
        top_k: usize,
        /// Override model directory
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), HypeRagError> {
    let cli = Cli::parse();
    let store = ChunkStore::open(&cli.data_dir)?;

    match cli.command {
        Commands::Init { model_dir } => {
            validate_model_dir(&model_dir)?;
            store.set_model_dir(&model_dir)?;
            println!(
                "initialized hype-rag data dir '{}' with model '{}'",
                cli.data_dir.display(),
                model_dir.display()
            );
        }
        Commands::Index {
            path,
            recursive,
            model_dir,
        } => {
            let model = resolve_model_dir(&store, model_dir)?;
            let embedder = Embedder::load(&model)?;
            let count = index_directory(&embedder, &store, &path, recursive)?;
            println!("indexed {count} chunks from '{}'", path.display());
        }
        Commands::Search {
            query,
            top_k,
            model_dir,
        } => {
            let model = resolve_model_dir(&store, model_dir)?;
            let embedder = Embedder::load(&model)?;
            let chunks = store.all_chunks()?;
            if chunks.is_empty() {
                return Err(HypeRagError::Other(
                    "index is empty; run `hype-rag index` first".into(),
                ));
            }
            let hits = search_chunks(&embedder, &query, &chunks, top_k)?;
            if hits.is_empty() {
                println!("no results");
                return Ok(());
            }
            for (i, hit) in hits.iter().enumerate() {
                println!(
                    "#{i} score={:.4} path={} chunk={} text={}",
                    hit.score,
                    hit.path,
                    hit.chunk_index,
                    truncate(&hit.text, 120)
                );
            }
        }
    }

    Ok(())
}

fn resolve_model_dir(
    store: &ChunkStore,
    override_dir: Option<PathBuf>,
) -> Result<PathBuf, HypeRagError> {
    if let Some(dir) = override_dir {
        validate_model_dir(&dir)?;
        return Ok(dir);
    }
    let Some(path) = store.model_dir()? else {
        return Err(HypeRagError::Other(
            "model directory not configured; run `hype-rag init --model-dir <path>`".into(),
        ));
    };
    let dir = PathBuf::from(path);
    validate_model_dir(&dir)?;
    Ok(dir)
}

fn validate_model_dir(dir: &PathBuf) -> Result<(), HypeRagError> {
    for file in ["config.json", "vocab.txt", "model.safetensors"] {
        let p = dir.join(file);
        if !p.exists() {
            return Err(HypeRagError::Other(format!(
                "model directory missing '{}'",
                p.display()
            )));
        }
    }
    Ok(())
}

fn truncate(text: &str, max: usize) -> String {
    text_boundary::truncate_chars(text, max)
}