#!/bin/bash
# -----------------------------------------------------------
# 🧠 Auto Git Commit & Push Script (with Cleanup + Git LFS)
# Repository: https://github.com/vishal1142/GenAI.git
# Author: Vishal Machan
# Project: FullGenAI (Spark / GenAI Framework)
# -----------------------------------------------------------

set -e  # Exit on error

# -----------------------------------------------------------
# 1️⃣  Initialize Git LFS and tracking rules
# -----------------------------------------------------------
echo "🔧 Initializing Git LFS..."
git lfs install

echo "🎯 Tracking large model files (*.pth) with Git LFS..."
git lfs track "*.pth"

# Save LFS configuration
git add .gitattributes
git commit -m "Track large files with Git LFS" || echo "✅ LFS tracking already configured."

# -----------------------------------------------------------
# 2️⃣  Remove unnecessary large/local folders from Git tracking
# -----------------------------------------------------------
echo "🧹 Cleaning up unnecessary folders from Git tracking..."
git rm -r --cached .venv || true
git rm -r --cached __pycache__ || true
git rm -r --cached day05_llmops_mlflow/mlruns || true

# -----------------------------------------------------------
# 3️⃣  Ensure .gitignore exists and is added
# -----------------------------------------------------------
echo "🧾 Adding .gitignore..."
cat > .gitignore << 'EOF'
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.venv/
env/
venv/

# Model / large artifacts
*.pth
*.pt
*.h5
*.pkl
*.dll
*.exe
*.jar
mlruns/
artifacts/

# Logs / misc
*.log
*.csv
*.zip
.DS_Store
Thumbs.db
EOF

git add .gitignore

# -----------------------------------------------------------
# 4️⃣  Standard commit and push workflow
# -----------------------------------------------------------
branch=$(git rev-parse --abbrev-ref HEAD)
echo "🚀 Current branch: $branch"

git status
git add .

echo "📝 Enter commit message (leave blank for auto message):"
read msg
if [ -z "$msg" ]; then
  msg="Auto commit on $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "✅ Committing changes..."
git commit -m "$msg" || echo "⚠️ No changes to commit."

# Ensure remote is correct
git remote set-url origin https://github.com/vishal1142/GenAI.git

# Force push (to clean history if large files were removed)
echo "⬆️ Pushing cleaned repo to GitHub..."
git push -u origin "$branch" --force

echo "🎉 Done! Repo pushed successfully to https://github.com/vishal1142/GenAI.git"
