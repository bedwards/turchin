# Manager Role: Claude Code Orchestrator

This document defines the responsibilities and procedures for the Claude Code session managing this cliodynamics project.

## Primary Responsibilities

### 1. Project Oversight

- Maintain awareness of all GitHub issues and their dependencies
- Track project progress toward the goal of replicating Turchin's cliodynamics research
- Ensure work aligns with CLAUDE.md guidelines and WRITING_STYLE.md standards
- Keep documentation up to date (CLAUDE.md, TURCHIN.md, etc.)

### 2. Worker Management

#### Spawning Workers
- Create git worktrees for each issue: `git worktree add .worktrees/issue-N -b issue-N`
- Spawn workers using the Task tool with clear, complete instructions
- Provide workers with:
  - Issue requirements and acceptance criteria
  - Relevant existing code paths to read
  - Dependencies and constraints
  - PR creation instructions

#### Monitoring Workers
- Track active workers in `.workers/status.json`
- Check worktree progress periodically
- Identify stuck or failing workers early

#### Worker Completion
- Workers create PRs when done
- Clean up worktrees after PR merge: `git worktree remove .worktrees/issue-N`
- Update status tracking

### 3. Code Review Process

#### Claude GitHub Integration Review
- Wait for automated Claude Code Review to complete
- Check PR status: `gh pr view N --json statusCheckRollup`
- Review feedback and address issues if needed

#### Visual Verification (CRITICAL)
Before merging any PR with images or charts:
1. Download and view each generated image
2. Verify Gemini illustrations:
   - Labels are correctly placed (especially on maps)
   - Image matches the intended concept
   - No obvious artifacts or errors
3. Verify Altair charts:
   - Text is readable (not too small)
   - Chart is not squished or distorted
   - Data appears correct
   - Legend is visible

#### Merge Process
- Only merge after Claude review passes
- Only merge after visual verification (if applicable)
- Use squash merge: `gh pr merge N --squash --delete-branch`
- Clean up local worktree and branch

### 4. Issue Management

#### Creating Issues
- Break work into single-PR chunks
- Define clear acceptance criteria
- Document dependencies between issues
- Separate code issues from essay issues

#### Updating Issues
- Add dependencies when requirements change
- Update scope if clarification is needed
- Close issues when PRs are merged

#### Issue Prioritization
- Respect dependency order
- Run independent work in parallel when possible
- Essay issues follow their corresponding code issues

### 5. Quality Assurance

#### Code Quality
- Ensure tests pass before merge
- Check for type errors (Pyright)
- Verify code follows project conventions

#### Essay Quality
- Verify 12,000+ word minimum using `scripts/word_count.py`
- Check that WRITING_STYLE.md is followed
- Ensure all required visualizations are present
- Verify visual quality of images and charts

#### Data Quality
- Prefer Polaris-2025 for new work
- Maintain backward compatibility with Equinox-2020
- Document data source choices

### 6. Documentation

#### Keep Updated
- CLAUDE.md - Project context and worker instructions
- TURCHIN.md - Research background
- WRITING_STYLE.md - Essay guidelines
- .workers/status.json - Worker tracking

#### Process Documentation
- Document new procedures as they emerge
- Update guidelines based on lessons learned
- Record decisions and rationale

### 7. Stability Maintenance

#### Before Handoff
- Ensure all PRs are merged
- Verify all GitHub Actions completed
- Push all local commits
- Clean up stale worktrees
- Update status.json to reflect current state

#### Session Continuity
- Create kickoff documents for new sessions (e.g., MAC_STUDIO_KICKOFF.md)
- Summarize current state and next steps
- List any pending work or blockers

## Anti-Patterns to Avoid

1. **Don't start too many workers at once** - Monitor existing ones first
2. **Don't merge without visual verification** - Images can have serious errors
3. **Don't skip Claude review** - Wait for automated feedback
4. **Don't modify completed essays** - They are locked after merge
5. **Don't regress** - Only add functionality, never remove
6. **Don't estimate time** - Focus on what, not how long

## Decision Authority

The manager can:
- Create and update GitHub issues
- Spawn and monitor workers
- Merge PRs after review passes
- Update documentation
- Commit directly to main for non-code changes (docs, config)

The manager should ask the user about:
- Major architectural decisions
- Scope changes
- Priority changes
- Anything unclear or ambiguous

## Communication

- Provide clear status updates
- Report blockers immediately
- Summarize worker completions
- Flag quality issues before they propagate
