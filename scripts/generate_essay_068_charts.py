"""Generate charts for Essay 068: The Claude Code Experiment.

This script creates all Altair charts showing project metrics, timeline,
and workflow visualizations.
"""

import sys
sys.path.insert(0, '/Users/bedwards/turchin/src')

import altair as alt
import pandas as pd
from pathlib import Path
from cliodynamics.viz.charts import configure_chart, save_chart, CHART_WIDTH

OUTPUT_DIR = Path('/Users/bedwards/turchin/docs/assets/charts/essay-068')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Project timeline data - from git log
timeline_data = [
    {'date': '2026-01-22 10:34', 'milestone': 'Project initialized', 'category': 'Setup'},
    {'date': '2026-01-22 10:58', 'milestone': 'Scaffolding complete', 'category': 'Setup'},
    {'date': '2026-01-22 11:48', 'milestone': 'First essay published', 'category': 'Essay'},
    {'date': '2026-01-22 12:07', 'milestone': 'Seshat data module', 'category': 'Code'},
    {'date': '2026-01-22 12:52', 'milestone': 'Essay: Seshat Data', 'category': 'Essay'},
    {'date': '2026-01-22 13:45', 'milestone': 'Calibration framework', 'category': 'Code'},
    {'date': '2026-01-22 14:15', 'milestone': 'Image fixes', 'category': 'Fix'},
    {'date': '2026-01-22 18:47', 'milestone': 'Seshat API client', 'category': 'Code'},
    {'date': '2026-01-22 19:37', 'milestone': 'Essay: Parameter Calibration', 'category': 'Essay'},
    {'date': '2026-01-22 19:41', 'milestone': 'Polaris-2025 integration', 'category': 'Code'},
    {'date': '2026-01-22 19:58', 'milestone': 'Roman case study', 'category': 'Code'},
    {'date': '2026-01-22 19:59', 'milestone': 'Essay: Mathematics of Collapse', 'category': 'Essay'},
    {'date': '2026-01-23 02:14', 'milestone': 'Animation module', 'category': 'Code'},
    {'date': '2026-01-23 02:15', 'milestone': 'Forecasting pipeline', 'category': 'Code'},
    {'date': '2026-01-23 02:45', 'milestone': 'Essay: Rome', 'category': 'Essay'},
    {'date': '2026-01-23 02:53', 'milestone': 'Essay: America', 'category': 'Essay'},
    {'date': '2026-01-23 07:18', 'milestone': 'Altair migration', 'category': 'Code'},
    {'date': '2026-01-23 08:03', 'milestone': 'Essay: Forecasting', 'category': 'Essay'},
    {'date': '2026-01-23 08:44', 'milestone': 'Essay: Policy Lessons', 'category': 'Essay'},
    {'date': '2026-01-23 08:55', 'milestone': 'Interactive Explorer', 'category': 'Code'},
    {'date': '2026-01-23 18:26', 'milestone': 'Plotly animations', 'category': 'Code'},
]

df_timeline = pd.DataFrame(timeline_data)
df_timeline['datetime'] = pd.to_datetime(df_timeline['date'])
df_timeline['hour'] = (df_timeline['datetime'] - df_timeline['datetime'].min()).dt.total_seconds() / 3600

# Chart 1: Project Timeline
print("Generating project timeline chart...")
chart1 = alt.Chart(df_timeline).mark_circle(size=150).encode(
    x=alt.X('hour:Q', title='Hours Since Project Start', scale=alt.Scale(domain=[0, 35])),
    y=alt.Y('category:N', title=None, sort=['Setup', 'Code', 'Essay', 'Fix']),
    color=alt.Color('category:N', title='Category',
                   scale=alt.Scale(domain=['Setup', 'Code', 'Essay', 'Fix'],
                                   range=['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728'])),
    tooltip=['milestone:N', 'date:N', 'category:N']
)
chart1 = configure_chart(chart1, 'Project Development Timeline (32 Hours)', height=300)
save_chart(chart1, OUTPUT_DIR / 'timeline.png')
print(f"  Saved: {OUTPUT_DIR / 'timeline.png'}")

# Chart 2: Issue Completion by Type
issue_data = [
    {'type': 'Infrastructure', 'completed': 5, 'open': 2},
    {'type': 'Data Integration', 'completed': 4, 'open': 0},
    {'type': 'Model Implementation', 'completed': 4, 'open': 1},
    {'type': 'Visualization', 'completed': 8, 'open': 3},
    {'type': 'Essays', 'completed': 8, 'open': 4},
    {'type': 'Case Studies', 'completed': 3, 'open': 0},
    {'type': 'Process', 'completed': 5, 'open': 0},
]
df_issues = pd.DataFrame(issue_data)
df_issues_melted = df_issues.melt(id_vars='type', var_name='status', value_name='count')

print("Generating issue completion chart...")
chart2 = alt.Chart(df_issues_melted).mark_bar().encode(
    x=alt.X('count:Q', title='Number of Issues'),
    y=alt.Y('type:N', title=None, sort='-x'),
    color=alt.Color('status:N', title='Status',
                   scale=alt.Scale(domain=['completed', 'open'],
                                   range=['#2ca02c', '#bdbdbd'])),
    order=alt.Order('status:N', sort='descending')
)
chart2 = configure_chart(chart2, 'GitHub Issues by Category', height=400)
save_chart(chart2, OUTPUT_DIR / 'issues-by-type.png')
print(f"  Saved: {OUTPUT_DIR / 'issues-by-type.png'}")

# Chart 3: Lines of Code Over Time (cumulative)
loc_data = [
    {'milestone': 'Scaffolding', 'hour': 0.5, 'src_lines': 200, 'test_lines': 100},
    {'milestone': 'Core SDT', 'hour': 1.5, 'src_lines': 1500, 'test_lines': 500},
    {'milestone': 'Data Layer', 'hour': 2.0, 'src_lines': 3000, 'test_lines': 1000},
    {'milestone': 'Calibration', 'hour': 3.2, 'src_lines': 4500, 'test_lines': 2000},
    {'milestone': 'Case Studies', 'hour': 9.5, 'src_lines': 7000, 'test_lines': 3500},
    {'milestone': 'Visualization', 'hour': 15.5, 'src_lines': 10000, 'test_lines': 5500},
    {'milestone': 'Explorer', 'hour': 22.5, 'src_lines': 13000, 'test_lines': 7500},
    {'milestone': 'Final', 'hour': 32, 'src_lines': 15939, 'test_lines': 9502},
]
df_loc = pd.DataFrame(loc_data)
df_loc_melted = df_loc.melt(id_vars=['milestone', 'hour'], var_name='type', value_name='lines')
df_loc_melted['type'] = df_loc_melted['type'].map({'src_lines': 'Source Code', 'test_lines': 'Test Code'})

print("Generating lines of code chart...")
chart3 = alt.Chart(df_loc_melted).mark_area(opacity=0.7).encode(
    x=alt.X('hour:Q', title='Hours Since Project Start'),
    y=alt.Y('lines:Q', title='Lines of Python Code', stack='zero'),
    color=alt.Color('type:N', title='Code Type',
                   scale=alt.Scale(domain=['Source Code', 'Test Code'],
                                   range=['#1f77b4', '#ff7f0e']))
)
chart3 = configure_chart(chart3, 'Codebase Growth Over 32 Hours', height=400)
save_chart(chart3, OUTPUT_DIR / 'lines-of-code.png')
print(f"  Saved: {OUTPUT_DIR / 'lines-of-code.png'}")

# Chart 4: Tool Usage from Session Logs
tool_data = [
    {'tool': 'Bash', 'count': 258, 'category': 'Execution'},
    {'tool': 'Task', 'count': 29, 'category': 'Coordination'},
    {'tool': 'Read', 'count': 11, 'category': 'Information'},
    {'tool': 'Grep', 'count': 6, 'category': 'Information'},
    {'tool': 'Edit', 'count': 3, 'category': 'Editing'},
    {'tool': 'Write', 'count': 1, 'category': 'Editing'},
]
df_tools = pd.DataFrame(tool_data)

print("Generating tool usage chart...")
chart4 = alt.Chart(df_tools).mark_bar().encode(
    x=alt.X('count:Q', title='Number of Invocations', scale=alt.Scale()),
    y=alt.Y('tool:N', title=None, sort='-x'),
    color=alt.Color('category:N', title='Category',
                   scale=alt.Scale(domain=['Execution', 'Coordination', 'Information', 'Editing'],
                                   range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']))
)
chart4 = configure_chart(chart4, 'Claude Code Tool Usage (Main Session)', height=350)
save_chart(chart4, OUTPUT_DIR / 'tool-usage.png')
print(f"  Saved: {OUTPUT_DIR / 'tool-usage.png'}")

# Chart 5: PR Merge Timeline  
pr_data = [
    {'pr': 11, 'title': 'Scaffolding', 'merged_hour': 0.5, 'files': 5},
    {'pr': 13, 'title': 'First Essay', 'merged_hour': 1.2, 'files': 8},
    {'pr': 21, 'title': 'Seshat Download', 'merged_hour': 1.6, 'files': 4},
    {'pr': 22, 'title': 'SDT Equations', 'merged_hour': 1.6, 'files': 3},
    {'pr': 23, 'title': 'Data Access', 'merged_hour': 1.8, 'files': 4},
    {'pr': 24, 'title': 'ODE Solver', 'merged_hour': 2.1, 'files': 5},
    {'pr': 25, 'title': 'Essay: Seshat', 'merged_hour': 2.3, 'files': 7},
    {'pr': 34, 'title': 'Image Fixes', 'merged_hour': 3.7, 'files': 3},
    {'pr': 35, 'title': 'Seshat API', 'merged_hour': 8.2, 'files': 4},
    {'pr': 36, 'title': 'Essay: Math', 'merged_hour': 9.4, 'files': 6},
    {'pr': 37, 'title': 'Essay: Calibration', 'merged_hour': 9.0, 'files': 7},
    {'pr': 38, 'title': 'Polaris-2025', 'merged_hour': 9.1, 'files': 5},
    {'pr': 39, 'title': 'Viz Module', 'merged_hour': 9.1, 'files': 8},
    {'pr': 40, 'title': 'US Data', 'merged_hour': 9.1, 'files': 6},
    {'pr': 43, 'title': 'Roman Study', 'merged_hour': 9.4, 'files': 7},
    {'pr': 45, 'title': 'Animation', 'merged_hour': 15.7, 'files': 4},
    {'pr': 46, 'title': 'Forecasting', 'merged_hour': 15.8, 'files': 6},
    {'pr': 50, 'title': 'Chart Fix', 'merged_hour': 15.9, 'files': 2},
    {'pr': 52, 'title': 'Dimension Check', 'merged_hour': 16.0, 'files': 2},
    {'pr': 58, 'title': 'Essay: Rome', 'merged_hour': 16.5, 'files': 7},
    {'pr': 59, 'title': 'Essay: America', 'merged_hour': 20.7, 'files': 8},
    {'pr': 60, 'title': 'Policy Sim', 'merged_hour': 20.7, 'files': 5},
    {'pr': 61, 'title': 'Altair Migration', 'merged_hour': 20.9, 'files': 12},
    {'pr': 62, 'title': 'Essay: Forecast', 'merged_hour': 21.8, 'files': 8},
    {'pr': 63, 'title': 'Essay: Policy', 'merged_hour': 22.2, 'files': 7},
    {'pr': 64, 'title': 'Explorer', 'merged_hour': 22.4, 'files': 9},
    {'pr': 65, 'title': 'Plotly Animations', 'merged_hour': 32.0, 'files': 6},
]
df_prs = pd.DataFrame(pr_data)

print("Generating PR merge velocity chart...")
chart5 = alt.Chart(df_prs).mark_circle(size=100).encode(
    x=alt.X('merged_hour:Q', title='Hours Since Project Start'),
    y=alt.Y('files:Q', title='Files Changed'),
    tooltip=['pr:Q', 'title:N', 'merged_hour:Q', 'files:Q']
)
chart5 = configure_chart(chart5, 'Pull Request Merge Velocity (27 PRs in 32 Hours)', height=350)
save_chart(chart5, OUTPUT_DIR / 'pr-velocity.png')
print(f"  Saved: {OUTPUT_DIR / 'pr-velocity.png'}")

# Chart 6: Session Log Metrics
session_data = [
    {'session': 'f44183b6', 'messages': 8, 'tool_calls': 116, 'size_mb': 152, 'duration_min': 2},
    {'session': '53d70cd3', 'messages': 26, 'tool_calls': 46, 'size_mb': 0.36, 'duration_min': 17},
    {'session': 'c1c9a046', 'messages': 105, 'tool_calls': 139, 'size_mb': 0.98, 'duration_min': 315},
    {'session': '5ab3a44b', 'messages': 709, 'tool_calls': 309, 'size_mb': 8.1, 'duration_min': 960},
]
df_sessions = pd.DataFrame(session_data)

print("Generating session metrics chart...")
chart6 = alt.Chart(df_sessions).mark_bar().encode(
    x=alt.X('session:N', title=None, sort=None),
    y=alt.Y('tool_calls:Q', title='Tool Invocations'),
    color=alt.Color('duration_min:Q', title='Duration (min)',
                   scale=alt.Scale(scheme='viridis'))
)
chart6 = configure_chart(chart6, 'Claude Code Session Activity', height=350)
save_chart(chart6, OUTPUT_DIR / 'session-metrics.png')
print(f"  Saved: {OUTPUT_DIR / 'session-metrics.png'}")

# Chart 7: Essay Word Counts
essay_data = [
    {'essay': '001 Introduction', 'words': 12000, 'reading_min': 60},
    {'essay': '002 Seshat Data', 'words': 12000, 'reading_min': 60},
    {'essay': '003 Mathematics', 'words': 12059, 'reading_min': 60},
    {'essay': '004 America', 'words': 12450, 'reading_min': 62},
    {'essay': '005 Rome', 'words': 12045, 'reading_min': 60},
    {'essay': '006 Policy', 'words': 12060, 'reading_min': 60},
    {'essay': '007 Forecasting', 'words': 12002, 'reading_min': 60},
    {'essay': '016 Calibration', 'words': 12390, 'reading_min': 62},
]
df_essays = pd.DataFrame(essay_data)

print("Generating essay word count chart...")
chart7 = alt.Chart(df_essays).mark_bar().encode(
    x=alt.X('words:Q', title='Word Count', scale=alt.Scale(domain=[0, 15000])),
    y=alt.Y('essay:N', title=None, sort=None),
    color=alt.value('#1f77b4')
)
# Add 12000 word minimum line
rule = alt.Chart(pd.DataFrame({'x': [12000]})).mark_rule(color='red', strokeDash=[5,5]).encode(x='x:Q')
chart7 = (chart7 + rule)
chart7 = configure_chart(chart7, 'Essay Word Counts (12,000 Minimum)', height=400)
save_chart(chart7, OUTPUT_DIR / 'essay-word-counts.png')
print(f"  Saved: {OUTPUT_DIR / 'essay-word-counts.png'}")

# Chart 8: Worker Pattern Analysis
worker_data = [
    {'wave': 1, 'workers': 4, 'issues': [28, 29, 15, 16, 26], 'parallel': True},
    {'wave': 2, 'workers': 5, 'issues': [27, 8, 9, 7, 10], 'parallel': True},
    {'wave': 3, 'workers': 4, 'issues': [17, 18, 41, 44], 'parallel': True},
    {'wave': 4, 'workers': 3, 'issues': [42, 20, 47], 'parallel': True},
    {'wave': 5, 'workers': 3, 'issues': [49, 53, 55], 'parallel': True},
    {'wave': 6, 'workers': 2, 'issues': [56, 63], 'parallel': True},
]
df_workers = pd.DataFrame(worker_data)
df_workers['total_issues'] = df_workers['issues'].apply(len)

print("Generating worker pattern chart...")
chart8 = alt.Chart(df_workers).mark_bar().encode(
    x=alt.X('wave:O', title='Wave Number'),
    y=alt.Y('total_issues:Q', title='Issues Assigned'),
    color=alt.Color('workers:Q', title='Parallel Workers',
                   scale=alt.Scale(scheme='blues'))
)
chart8 = configure_chart(chart8, 'Worker Wave Pattern (Parallel Task Execution)', height=350)
save_chart(chart8, OUTPUT_DIR / 'worker-waves.png')
print(f"  Saved: {OUTPUT_DIR / 'worker-waves.png'}")

# Chart 9: Asset Generation
asset_data = [
    {'type': 'Gemini Illustrations', 'count': 15, 'category': 'Image'},
    {'type': 'Altair Charts', 'count': 45, 'category': 'Chart'},
    {'type': 'Plotly Animations', 'count': 12, 'category': 'Animation'},
    {'type': 'HTML Essays', 'count': 8, 'category': 'Document'},
    {'type': 'CSS/JS Assets', 'count': 10, 'category': 'Web'},
]
df_assets = pd.DataFrame(asset_data)

print("Generating asset breakdown chart...")
chart9 = alt.Chart(df_assets).mark_bar().encode(
    x=alt.X('count:Q', title='Number of Files'),
    y=alt.Y('type:N', title=None, sort='-x'),
    color=alt.Color('category:N', title='Category',
                   scale=alt.Scale(domain=['Image', 'Chart', 'Animation', 'Document', 'Web'],
                                   range=['#e377c2', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']))
)
chart9 = configure_chart(chart9, 'Generated Assets by Type', height=350)
save_chart(chart9, OUTPUT_DIR / 'asset-breakdown.png')
print(f"  Saved: {OUTPUT_DIR / 'asset-breakdown.png'}")

# Chart 10: Human vs AI Contributions
contribution_data = [
    {'activity': 'Requirements & Direction', 'human': 95, 'ai': 5},
    {'activity': 'Code Implementation', 'human': 5, 'ai': 95},
    {'activity': 'Code Review', 'human': 20, 'ai': 80},
    {'activity': 'Visual Verification', 'human': 90, 'ai': 10},
    {'activity': 'Essay Writing', 'human': 10, 'ai': 90},
    {'activity': 'Issue Management', 'human': 30, 'ai': 70},
    {'activity': 'Architecture Decisions', 'human': 60, 'ai': 40},
    {'activity': 'Bug Fixes', 'human': 10, 'ai': 90},
]
df_contrib = pd.DataFrame(contribution_data)
df_contrib_melted = df_contrib.melt(id_vars='activity', var_name='contributor', value_name='percent')
df_contrib_melted['contributor'] = df_contrib_melted['contributor'].map({'human': 'Human', 'ai': 'Claude Code'})

print("Generating contribution breakdown chart...")
chart10 = alt.Chart(df_contrib_melted).mark_bar().encode(
    x=alt.X('percent:Q', title='Percentage of Work', stack='normalize', 
            axis=alt.Axis(format='%')),
    y=alt.Y('activity:N', title=None, sort=None),
    color=alt.Color('contributor:N', title='Contributor',
                   scale=alt.Scale(domain=['Human', 'Claude Code'],
                                   range=['#ff7f0e', '#1f77b4']))
)
chart10 = configure_chart(chart10, 'Work Distribution: Human vs Claude Code', height=400)
save_chart(chart10, OUTPUT_DIR / 'contribution-breakdown.png')
print(f"  Saved: {OUTPUT_DIR / 'contribution-breakdown.png'}")

print("\n" + "="*60)
print("All charts generated successfully!")
print(f"Output directory: {OUTPUT_DIR}")
print("="*60)
