#!/bin/bash
# Mark a worker as complete and clean up
# Usage: ./complete.sh <issue_number> [merged|failed]

set -e

ISSUE_NUM=$1
RESULT=${2:-merged}

if [ -z "$ISSUE_NUM" ]; then
    echo "Usage: $0 <issue_number> [merged|failed]"
    exit 1
fi

REPO_ROOT=$(git rev-parse --show-toplevel)
WORKERS_DIR="$REPO_ROOT/.workers"
STATUS_FILE="$WORKERS_DIR/status.json"
BRANCH_NAME="issue-${ISSUE_NUM}"
WORKTREE_PATH="$REPO_ROOT/.worktrees/$BRANCH_NAME"

echo "=== Completing worker for issue #$ISSUE_NUM (result: $RESULT) ==="

# Update status.json
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
python3 << EOF
import json

with open("$STATUS_FILE", "r") as f:
    status = json.load(f)

worker = status["workers"].pop("$ISSUE_NUM", None)
if worker:
    worker["completed"] = "$TIMESTAMP"
    worker["result"] = "$RESULT"
    if "$RESULT" == "merged":
        status["completed"].append(worker)
    else:
        status["failed"].append(worker)

with open("$STATUS_FILE", "w") as f:
    json.dump(status, f, indent=2)
EOF

# Remove worktree
if [ -d "$WORKTREE_PATH" ]; then
    git worktree remove "$WORKTREE_PATH" --force
    echo "Removed worktree: $WORKTREE_PATH"
fi

# Delete branch if merged
if [ "$RESULT" = "merged" ]; then
    git branch -d "$BRANCH_NAME" 2>/dev/null || echo "Branch already deleted or not merged"
fi

echo "Worker #$ISSUE_NUM completed with result: $RESULT"
