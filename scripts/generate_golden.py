#!/usr/bin/env python3
"""Generate golden reference embeddings for the HypEmbed test suite.

Usage:
    python scripts/generate_golden.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --output tests/fixtures/golden.json

Requires: pip install sentence-transformers
"""

import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Generate golden embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model name or path")
    parser.add_argument("--output", default="tests/fixtures/golden.json",
                        help="Output JSON file path")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    # Test sentences covering various edge cases
    sentences = [
        "Hello world",
        "Rust is a systems programming language.",
        "Machine learning models generate embedding vectors.",
        "The quick brown fox jumps over the lazy dog.",
        "café résumé naïve",  # Accented characters
        "你好世界",  # Chinese characters
        "Short",
        "",  # Empty string
        "A " * 100,  # Long repeated text
        "Hello, World! How are you doing today?",
    ]

    model = SentenceTransformer(args.model)

    golden = {}
    for sentence in sentences:
        if not sentence.strip():
            continue  # Skip empty
        embedding = model.encode([sentence], normalize_embeddings=True)
        golden[sentence] = embedding.tolist()

    with open(args.output, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"Generated golden embeddings for {len(golden)} sentences")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Embedding dim: {len(list(golden.values())[0][0])}")

if __name__ == "__main__":
    main()
