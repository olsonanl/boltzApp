#!/bin/bash
#
# preload_boltz_cache.sh - Download all Boltz model weights and cache files
#
# Usage: ./preload_boltz_cache.sh [CACHE_DIR]
#        CACHE_DIR defaults to /cache or $BOLTZ_CACHE or ~/.boltz
#
# This script downloads all required Boltz artifacts so they don't need
# to be downloaded at runtime. Useful for:
#   - Building containers with pre-cached weights
#   - Air-gapped environments
#   - Faster job startup times

set -e

CACHE_DIR="${1:-${BOLTZ_CACHE:-${HOME}/.boltz}}"

echo "Preloading Boltz cache to: $CACHE_DIR"

# Create directory structure
mkdir -p "$CACHE_DIR"

# URLs - using HuggingFace as primary (more reliable for scripted downloads)
HF_BASE_URL="https://huggingface.co/boltz-community"

# Boltz-2 model checkpoints (~8GB total)
echo "Downloading Boltz-2 structure model (boltz2_conf.ckpt)..."
curl -fSL "$HF_BASE_URL/boltz-2/resolve/main/boltz2_conf.ckpt" -o "$CACHE_DIR/boltz2_conf.ckpt"

echo "Downloading Boltz-2 affinity model (boltz2_aff.ckpt)..."
curl -fSL "$HF_BASE_URL/boltz-2/resolve/main/boltz2_aff.ckpt" -o "$CACHE_DIR/boltz2_aff.ckpt"

# CCD dictionary (Chemical Component Dictionary)
echo "Downloading CCD dictionary (ccd.pkl)..."
curl -fSL "$HF_BASE_URL/boltz-1/resolve/main/ccd.pkl" -o "$CACHE_DIR/ccd.pkl"

# Molecular data (contains canonical amino acid/nucleotide definitions)
echo "Downloading molecular data (mols.tar)..."
curl -fSL "$HF_BASE_URL/boltz-2/resolve/main/mols.tar" -o "$CACHE_DIR/mols.tar"

echo "Extracting molecular data..."
tar -xf "$CACHE_DIR/mols.tar" -C "$CACHE_DIR"
rm "$CACHE_DIR/mols.tar"

# Optional: Boltz-1 model (uncomment if needed for backwards compatibility)
# echo "Downloading Boltz-1 structure model (boltz1_conf.ckpt)..."
# curl -fSL "$HF_BASE_URL/boltz-1/resolve/main/boltz1_conf.ckpt" -o "$CACHE_DIR/boltz1_conf.ckpt"

echo ""
echo "Boltz cache preloaded successfully!"
echo "Files downloaded:"
ls -lh "$CACHE_DIR"
echo ""
echo "Total size:"
du -sh "$CACHE_DIR"
