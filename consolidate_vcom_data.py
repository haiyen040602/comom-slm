#!/usr/bin/env python3
"""Consolidate VCOM data from multiple files into single train/dev/test files."""

import os
from pathlib import Path

def consolidate_split(split_name):
    """Consolidate all files in a split into a single file."""
    split_dir = Path(f"datasets/vcom-data/{split_name}")
    output_file = Path(f"datasets/vcom-data/{split_name}.txt")
    
    if not split_dir.exists():
        print(f"⚠️  Directory {split_dir} not found")
        return
    
    # Get all .txt files sorted
    files = sorted([f for f in split_dir.glob("*.txt") if f.is_file()])
    
    if not files:
        print(f"⚠️  No .txt files found in {split_dir}")
        return
    
    print(f"📁 Processing {split_name}: {len(files)} files")
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for filepath in files:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                out.write(content)
                # Ensure newline between files if not present
                if content and not content.endswith('\n'):
                    out.write('\n')
    
    # Count lines (rough estimate of samples)
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"✅ Created {output_file}: {len(lines)} lines")
    return output_file

if __name__ == "__main__":
    print("🔄 Consolidating VCOM data...\n")
    
    for split in ["train", "dev", "test"]:
        consolidate_split(split)
    
    print("\n✨ Done! Created:")
    print("  - datasets/vcom-data/train.txt")
    print("  - datasets/vcom-data/dev.txt")
    print("  - datasets/vcom-data/test.txt")
