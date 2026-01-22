#!/bin/bash
# Monitor worker status
# Usage: ./monitor.sh [issue_number]

REPO_ROOT=$(git rev-parse --show-toplevel)
WORKERS_DIR="$REPO_ROOT/.workers"
STATUS_FILE="$WORKERS_DIR/status.json"

if [ -n "$1" ]; then
    # Show specific worker
    ISSUE_NUM=$1
    LOG_FILE="$WORKERS_DIR/logs/issue-${ISSUE_NUM}.log"

    echo "=== Worker #$ISSUE_NUM Status ==="
    python3 << EOF
import json
with open("$STATUS_FILE", "r") as f:
    status = json.load(f)
worker = status["workers"].get("$ISSUE_NUM", {})
if worker:
    print(f"Issue: #{worker['issue']} - {worker['title']}")
    print(f"Status: {worker['status']}")
    print(f"Branch: {worker['branch']}")
    print(f"Started: {worker['started']}")
else:
    print("Worker not found")
EOF

    echo ""
    echo "=== Recent Log ==="
    if [ -f "$LOG_FILE" ]; then
        tail -50 "$LOG_FILE"
    else
        echo "No log file found"
    fi

    echo ""
    echo "=== Git Status ==="
    WORKTREE_PATH="$REPO_ROOT/.worktrees/issue-$ISSUE_NUM"
    if [ -d "$WORKTREE_PATH" ]; then
        cd "$WORKTREE_PATH" && git status --short && git log --oneline -5
    else
        echo "Worktree not found"
    fi
else
    # Show all workers
    echo "=== All Workers ==="
    python3 << EOF
import json
with open("$STATUS_FILE", "r") as f:
    status = json.load(f)

if status["workers"]:
    for issue_num, worker in status["workers"].items():
        print(f"#{worker['issue']:3} | {worker['status']:10} | {worker['title'][:50]}")
else:
    print("No active workers")

if status["completed"]:
    print(f"\nCompleted: {len(status['completed'])} workers")

if status["failed"]:
    print(f"Failed: {len(status['failed'])} workers")
EOF

    echo ""
    echo "=== Worktrees ==="
    git worktree list

    echo ""
    echo "=== Open PRs ==="
    gh pr list --state open
fi
