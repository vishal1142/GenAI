#!/bin/bash
# -----------------------------------------------------------
# 🚀 Auto Git Commit & Push Script for GenAI Repo
# Repository: https://github.com/vishal1142/GenAI.git
# Author: Vishal Machan
# -----------------------------------------------------------

set -e

# Ensure remote is correct
git remote set-url origin https://github.com/vishal1142/GenAI.git

# Initialize Git LFS
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track large files with Git LFS" || true

# Remove large/unnecessary local files
git rm -r --cached .venv || true
git rm -r --cached __pycache__ || true
git rm -r --cached day05_llmops_mlflow/mlruns || true

# Add .gitignore if missing
cat > .gitignore << 'EOF'
.venv/
__pycache__/
mlruns/
*.pth
*.pt
*.pkl
*.h5
*.dll
*.jar
*.exe
*.log
*.zip
*.csv
artifacts/
EOF
git add .gitignore

# Commit all changes
git add .
echo "Enter commit message:"
read msg
if [ -z "$msg" ]; then
  msg="Auto commit on $(date '+%Y-%m-%d %H:%M:%S')"
fi
git commit -m "$msg" || echo "No new changes."

# Push
git push -u origin main --force

echo "✅ Code successfully pushed to https://github.com/vishal1142/GenAI.git"
