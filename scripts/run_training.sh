#!/bin/bash
# run_training.sh — Execute diagnostic and training scripts

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=========================================================================="
echo "HUNZA VALLEY — DIAGNOSTIC & TRAINING WORKFLOW"
echo "=========================================================================="

# Check venv
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "ERROR: .venv not found at $PROJECT_ROOT/.venv"
    exit 1
fi

# Activate venv
source "$PROJECT_ROOT/.venv/bin/activate"
echo "[*] Virtual environment activated"

cd "$SCRIPT_DIR"

# Step 1: Diagnose
echo ""
echo "=========================================================================="
echo "STEP 1: DIAGNOSE LEAKAGE & SEPARABILITY"
echo "=========================================================================="
echo "This will analyze why you got 100% accuracy..."
echo ""

python diagnose_leakage.py
if [ $? -ne 0 ]; then
    echo "ERROR: diagnose_leakage.py failed"
    exit 1
fi

echo ""
echo "Diagnostic complete. Check ../figures/ for visualizations."
echo "Press Enter to continue with training..."
read

# Step 2: Train
echo ""
echo "=========================================================================="
echo "STEP 2: TRAIN RANDOM FOREST"
echo "=========================================================================="
echo ""

python train_random_forest_v2.py --mode distillation
if [ $? -ne 0 ]; then
    echo "ERROR: train_random_forest_v2.py failed"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "SUCCESS: All training steps complete!"
echo "=========================================================================="
echo "Check these directories:"
echo "  Figures:  ../figures/"
echo "  Models:   ../models/"
echo ""
