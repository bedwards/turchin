#!/bin/bash
# Spawn a worker for a GitHub issue
# Usage: ./spawn.sh <issue_number>

set -e

ISSUE_NUM=$1
if [ -z "$ISSUE_NUM" ]; then
    echo "Usage: $0 <issue_number>"
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
WORKERS_DIR="$REPO_ROOT/.workers"
BRANCH_NAME="issue-${ISSUE_NUM}"
WORKTREE_PATH="$REPO_ROOT/.worktrees/$BRANCH_NAME"

# Get issue title for logging
ISSUE_TITLE=$(gh issue view "$ISSUE_NUM" --json title -q '.title')

echo "=== Spawning worker for issue #$ISSUE_NUM: $ISSUE_TITLE ==="

# Create feature branch from main
git fetch origin main
git branch "$BRANCH_NAME" origin/main 2>/dev/null || echo "Branch $BRANCH_NAME already exists"

# Create worktree
mkdir -p "$REPO_ROOT/.worktrees"
if [ -d "$WORKTREE_PATH" ]; then
    echo "Worktree already exists at $WORKTREE_PATH"
else
    git worktree add "$WORKTREE_PATH" "$BRANCH_NAME"
fi

# Update status
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
STATUS_FILE="$WORKERS_DIR/status.json"
LOG_FILE="$WORKERS_DIR/logs/issue-${ISSUE_NUM}.log"

# Initialize log
echo "=== Worker Log for Issue #$ISSUE_NUM ===" > "$LOG_FILE"
echo "Title: $ISSUE_TITLE" >> "$LOG_FILE"
echo "Started: $TIMESTAMP" >> "$LOG_FILE"
echo "Branch: $BRANCH_NAME" >> "$LOG_FILE"
echo "Worktree: $WORKTREE_PATH" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"

# Update status.json using Python for proper JSON handling
python3 << EOF
import json
from datetime import datetime

with open("$STATUS_FILE", "r") as f:
    status = json.load(f)

status["workers"]["$ISSUE_NUM"] = {
    "issue": $ISSUE_NUM,
    "title": "$ISSUE_TITLE",
    "branch": "$BRANCH_NAME",
    "worktree": "$WORKTREE_PATH",
    "status": "running",
    "started": "$TIMESTAMP",
    "log": "$LOG_FILE"
}

with open("$STATUS_FILE", "w") as f:
    json.dump(status, f, indent=2)
EOF

echo "Worker spawned. Worktree at: $WORKTREE_PATH"
echo "Log file: $LOG_FILE"
echo ""
echo "To run the worker:"
echo "  cd $WORKTREE_PATH && claude --print \"$(gh issue view $ISSUE_NUM --json body -q '.body')\""
