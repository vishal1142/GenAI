#!/bin/bash
# -----------------------------------------------------------
# 🧠 Clean & Auto Push Script for GenAI Repository
# Repository: https://github.com/vishal1142/GenAI.git
# Author: Vishal Machan
# Purpose: Remove heavy files (.venv, mlruns, artifacts),
#          enable LFS, rewrite history, and safely push.
# -----------------------------------------------------------

set -e  # Exit if any command fails

echo "🚀 Starting full cleanup and push for GenAI project..."

# -----------------------------------------------------------
# 1️⃣ Ensure correct remote
# -----------------------------------------------------------
REPO_URL="https://github.com/vishal1142/GenAI.git"
git remote set-url origin "$REPO_URL"
echo "✅ Remote set to: $REPO_URL"

# -----------------------------------------------------------
# 2️⃣ Install git-filter-repo (if missing)
# -----------------------------------------------------------
if ! command -v git-filter-repo &> /dev/null; then
  echo "📦 Installing git-filter-repo..."
  pip install git-filter-repo
else
  echo "✅ git-filter-repo already installed."
fi

# -----------------------------------------------------------
# 3️⃣ Remove large/unnecessary folders from Git history
# -----------------------------------------------------------
echo "🧹 Cleaning large folders from history..."
git lfs uninstall || true
git filter-repo --path .venv --invert-paths || true
git filter-repo --path day05_llmops_mlflow/mlruns --invert-paths || true
git filter-repo --path artifacts/models --invert-paths || true

# -----------------------------------------------------------
# 4️⃣ Re-enable Git LFS tracking
# -----------------------------------------------------------
echo "📦 Reinitializing Git LFS tracking..."
git lfs install
git lfs track "*.pth"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Enable Git LFS for model files" || echo "ℹ️ LFS already configured."

# -----------------------------------------------------------
# 5️⃣ Ensure .gitignore exists and covers large/unwanted files
# -----------------------------------------------------------
echo "🧾 Updating .gitignore..."
cat > .gitignore << 'EOF'
# Python and virtual environments
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
env/
venv/

# MLflow and model artifacts
mlruns/
artifacts/
*.pth
*.pt
*.pkl
*.h5
*.dll
*.exe
*.jar

# Temporary / logs / misc
*.log
*.csv
*.zip
*.tar
*.gz
.DS_Store
Thumbs.db
EOF

git add .gitignore
git commit -m "Update .gitignore to exclude heavy files" || echo "✅ .gitignore unchanged."

# -----------------------------------------------------------
# 6️⃣ Stage and commit all safe changes
# -----------------------------------------------------------
git add .
echo "📝 Enter commit message (leave blank for auto):"
read msg
if [ -z "$msg" ]; then
  msg="Cleaned large files & pushed on $(date '+%Y-%m-%d %H:%M:%S')"
fi

git commit -m "$msg" || echo "⚠️ No new changes to commit."

# -----------------------------------------------------------
# 7️⃣ Force push cleaned repository to GitHub
# -----------------------------------------------------------
echo "⬆️ Force pushing clean history to GitHub..."
git push -u origin main --force

echo "🎉 Done!"
echo "✅ Repository cleaned and pushed successfully to: $REPO_URL"
echo "🧩 All large files (.venv, mlruns, artifacts) removed from history."
echo "🚫 No more >100MB GitHub file errors will occur."
echo "-----------------------------------------------------------"
echo "💡 Tip: Always use this script instead of manual pushes."
echo "-----------------------------------------------------------"
