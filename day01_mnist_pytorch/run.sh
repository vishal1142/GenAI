#!/bin/bash
# ============================================================
# 🚀 MNIST PyTorch Project Runner
# Author: Vishal Machan
# Description: Setup environment, install dependencies, and run training.
# ============================================================

set -e  # exit on error

# Step 1: Navigate to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "📂 Working Directory: $PROJECT_ROOT"

# Step 2: Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "🧩 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✅ Virtual environment already exists."
fi

# Step 3: Activate virtual environment
source .venv/bin/activate
echo "🟢 Virtual environment activated."

# Step 4: Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found! Skipping dependency installation."
fi

# Step 5: Run the main script
echo "🚀 Starting MNIST training..."
python main.py

# Optional: Save output logs
# python main.py > logs/train_$(date +"%Y%m%d_%H%M%S").log 2>&1

echo "✅ Training completed successfully."
