"""HTML report generator for evaluation results."""

from __future__ import annotations

import base64
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import EvalScore, ResultStatus, ScenarioResult, Stage


def generate_html_report(results: list[ScenarioResult], output_path: Path, json_data: str | None = None) -> Path:
    """Generate a styled HTML report from evaluation results.
    
    Args:
        results: Evaluation results to render.
        output_path: Where to write the HTML file.
        json_data: Optional raw JSON string to embed in the report for easy sharing.
    """
    total = len(results)
    passed = sum(1 for r in results if r.status == ResultStatus.PASSED)
    failed = sum(1 for r in results if r.status == ResultStatus.FAILED)
    skipped = sum(1 for r in results if r.status == ResultStatus.SKIPPED)
    errored = sum(1 for r in results if r.status == ResultStatus.ERROR)
    # Pass rate excludes skipped scenarios
    evaluated = total - skipped
    pass_rate = (passed / evaluated * 100) if evaluated else 0

    by_stage = _group_by_stage(results)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Build embedded JSON section
    json_section = ""
    if json_data:
        json_section = _build_json_section(json_data)

    report_html = _TEMPLATE.format(
        timestamp=timestamp,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=f"{pass_rate:.0f}",
        pass_rate_decimal=f"{pass_rate:.1f}",
        summary_cards=_build_summary_cards(results),
        stage_sections=_build_stage_sections(by_stage),
        evaluator_breakdown=_build_evaluator_breakdown(results),
        tool_call_timeline=_build_tool_timeline(results),
        json_section=json_section,
        pass_color="#22c55e" if pass_rate >= 80 else "#f59e0b" if pass_rate >= 50 else "#ef4444",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")
    return output_path


def _group_by_stage(results: list[ScenarioResult]) -> dict[Stage, list[ScenarioResult]]:
    groups: dict[Stage, list[ScenarioResult]] = {}
    for r in results:
        groups.setdefault(r.stage, []).append(r)
    return groups


def _build_summary_cards(results: list[ScenarioResult]) -> str:
    stages = [Stage.SETUP, Stage.VERIFY, Stage.TEARDOWN]
    cards = []
    for stage in stages:
        stage_results = [r for r in results if r.stage == stage]
        if not stage_results:
            continue
        p = sum(1 for r in stage_results if r.status == ResultStatus.PASSED)
        s = sum(1 for r in stage_results if r.status == ResultStatus.SKIPPED)
        t = len(stage_results)
        evaluated = t - s
        pct = (p / evaluated * 100) if evaluated else 0
        color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
        skip_note = f" ({s} skipped)" if s else ""
        cards.append(f"""
        <div class="card">
            <div class="card-label">{stage.value.upper()}</div>
            <div class="card-value" style="color: {color}">{p}/{evaluated}</div>
            <div class="card-sub">{pct:.0f}% pass rate{skip_note}</div>
        </div>""")

    # Average latency card
    durations = [r.trace.total_duration_ms for r in results
                 if r.trace and r.trace.total_duration_ms]
    avg_ms = sum(durations) / len(durations) if durations else 0
    cards.append(f"""
        <div class="card">
            <div class="card-label">AVG LATENCY</div>
            <div class="card-value" style="color: #6366f1">{avg_ms / 1000:.1f}s</div>
            <div class="card-sub">{len(durations)} scenarios measured</div>
        </div>""")

    # Total tool calls card
    total_tools = sum(len(r.trace.tool_calls) for r in results if r.trace)
    cards.append(f"""
        <div class="card">
            <div class="card-label">TOOL CALLS</div>
            <div class="card-value" style="color: #8b5cf6">{total_tools}</div>
            <div class="card-sub">across all scenarios</div>
        </div>""")

    return "\n".join(cards)


def _build_stage_sections(by_stage: dict[Stage, list[ScenarioResult]]) -> str:
    sections = []
    stage_order = [Stage.SETUP, Stage.VERIFY, Stage.TEARDOWN]

    for stage in stage_order:
        stage_results = by_stage.get(stage, [])
        if not stage_results:
            continue

        rows = []
        for r in stage_results:
            if r.status == ResultStatus.SKIPPED:
                status_class = "skip"
                status_icon = "⊘"
                status_label = "SKIP"
            elif r.status == ResultStatus.PASSED:
                status_class = "pass"
                status_icon = "✓"
                status_label = "PASS"
            elif r.status == ResultStatus.ERROR:
                status_class = "fail"
                status_icon = "⚠"
                status_label = "ERR"
            else:
                status_class = "fail"
                status_icon = "✗"
                status_label = "FAIL"

            duration = ""
            if r.trace and r.trace.total_duration_ms:
                duration = f"{r.trace.total_duration_ms / 1000:.1f}s"

            tool_count = len(r.trace.tool_calls) if r.trace else 0
            tools_used = ", ".join(
                sorted({tc.tool_name for tc in r.trace.tool_calls})
            ) if r.trace else "—"

            score_pills = []
            for s in r.scores:
                pill_class = "pill-pass" if s.passed else "pill-fail"
                score_pills.append(
                    f'<span class="pill {pill_class}" '
                    f'title="{html.escape(s.reasoning)}">'
                    f'{html.escape(s.evaluator)}: {s.score:.0%}</span>'
                )

            error_row = ""
            if r.error:
                error_row = f'<div class="error-msg">⚠ {html.escape(r.error)}</div>'

            rows.append(f"""
            <tr class="scenario-row {status_class}-row">
                <td><span class="status-badge {status_class}">{status_icon} {status_label}</span></td>
                <td>
                    <div class="scenario-name">{html.escape(r.scenario_name)}</div>
                    <div class="scenario-id">{html.escape(r.scenario_id)}</div>
                    {error_row}
                </td>
                <td class="tools-cell">
                    <span class="tool-count">{tool_count}</span>
                    <div class="tools-detail">{html.escape(tools_used)}</div>
                </td>
                <td class="duration-cell">{duration}</td>
                <td class="scores-cell">{"".join(score_pills)}</td>
            </tr>""")

        sections.append(f"""
        <div class="stage-section">
            <h2 class="stage-header">
                <span class="stage-icon">{'🔧' if stage == Stage.SETUP else '🔍' if stage == Stage.VERIFY else '🧹'}</span>
                {stage.value.upper()}
            </h2>
            <table class="results-table">
                <thead>
                    <tr>
                        <th width="100">Status</th>
                        <th>Scenario</th>
                        <th width="140">Tools</th>
                        <th width="90">Duration</th>
                        <th>Evaluators</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>""")

    return "\n".join(sections)


def _build_evaluator_breakdown(results: list[ScenarioResult]) -> str:
    evaluator_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        for s in r.scores:
            if s.evaluator not in evaluator_stats:
                evaluator_stats[s.evaluator] = {
                    "total": 0, "passed": 0, "scores": []
                }
            evaluator_stats[s.evaluator]["total"] += 1
            if s.passed:
                evaluator_stats[s.evaluator]["passed"] += 1
            evaluator_stats[s.evaluator]["scores"].append(s.score)

    rows = []
    for name, stats in sorted(evaluator_stats.items()):
        avg = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
        pct = (stats["passed"] / stats["total"] * 100) if stats["total"] else 0
        bar_color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
        rows.append(f"""
        <tr>
            <td class="eval-name">{html.escape(name)}</td>
            <td>{stats['passed']}/{stats['total']}</td>
            <td>{avg:.0%}</td>
            <td>
                <div class="bar-container">
                    <div class="bar-fill" style="width: {pct}%; background: {bar_color}"></div>
                </div>
            </td>
        </tr>""")

    return f"""
    <table class="eval-table">
        <thead>
            <tr><th>Evaluator</th><th>Pass</th><th>Avg Score</th><th>Rate</th></tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_tool_timeline(results: list[ScenarioResult]) -> str:
    tool_stats: dict[str, dict[str, Any]] = {}
    for r in results:
        if not r.trace:
            continue
        for tc in r.trace.tool_calls:
            if tc.tool_name not in tool_stats:
                tool_stats[tc.tool_name] = {
                    "count": 0, "errors": 0, "durations": []
                }
            tool_stats[tc.tool_name]["count"] += 1
            if tc.error:
                tool_stats[tc.tool_name]["errors"] += 1
            if tc.duration_ms:
                tool_stats[tc.tool_name]["durations"].append(tc.duration_ms)

    rows = []
    for name, stats in sorted(tool_stats.items(), key=lambda x: -x[1]["count"]):
        avg_ms = (sum(stats["durations"]) / len(stats["durations"])
                  if stats["durations"] else 0)
        error_badge = (f'<span class="pill pill-fail">{stats["errors"]} errors</span>'
                       if stats["errors"] else "")
        rows.append(f"""
        <tr>
            <td class="tool-name">{html.escape(name)}</td>
            <td>{stats['count']}</td>
            <td>{avg_ms:.0f}ms</td>
            <td>{error_badge}</td>
        </tr>""")

    return f"""
    <table class="eval-table">
        <thead>
            <tr><th>Tool</th><th>Calls</th><th>Avg Latency</th><th>Errors</th></tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>"""


def _build_json_section(json_data: str) -> str:
    """Build an embedded JSON section with collapsible viewer and download button."""
    b64 = base64.b64encode(json_data.encode("utf-8")).decode("ascii")
    escaped = html.escape(json_data)
    return f"""
    <div class="breakdown-panel" style="margin-top: 1.5rem;">
        <h3>📋 Raw JSON Data</h3>
        <p style="color: var(--text-muted); margin: 0.5rem 0;">
            <a href="data:application/json;base64,{b64}" download="results.json"
               style="color: var(--blue); text-decoration: none; font-weight: 600;">
                ⬇ Download JSON
            </a>
            &nbsp;&middot;&nbsp;
            <span onclick="document.getElementById('json-viewer').style.display = document.getElementById('json-viewer').style.display === 'none' ? 'block' : 'none'"
                  style="color: var(--blue); cursor: pointer; font-weight: 600;">
                ▶ Toggle JSON viewer
            </span>
        </p>
        <pre id="json-viewer" style="display:none; max-height:500px; overflow:auto; background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:1rem; font-size:0.8rem; white-space:pre-wrap; word-break:break-all;">{escaped}</pre>
    </div>"""


_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dataverse MCP Eval Report</title>
<style>
    :root {{
        --bg: #0f172a;
        --surface: #1e293b;
        --surface2: #334155;
        --border: #475569;
        --text: #f1f5f9;
        --text-muted: #94a3b8;
        --green: #22c55e;
        --red: #ef4444;
        --amber: #f59e0b;
        --blue: #3b82f6;
        --purple: #8b5cf6;
        --indigo: #6366f1;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.6;
        padding: 2rem;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}

    /* Header */
    .header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid var(--border);
    }}
    .header h1 {{
        font-size: 1.75rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--blue), var(--purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .header .timestamp {{ color: var(--text-muted); font-size: 0.875rem; }}

    /* Hero score */
    .hero {{
        text-align: center;
        padding: 2rem;
        margin-bottom: 2rem;
        background: var(--surface);
        border-radius: 16px;
        border: 1px solid var(--border);
    }}
    .hero-score {{
        font-size: 4rem;
        font-weight: 800;
        color: {pass_color};
    }}
    .hero-label {{ color: var(--text-muted); font-size: 1rem; margin-top: 0.25rem; }}
    .hero-detail {{
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin-top: 1rem;
        color: var(--text-muted);
        font-size: 0.9rem;
    }}
    .hero-detail span {{ }}
    .hero-detail .pass-count {{ color: var(--green); font-weight: 600; }}
    .hero-detail .fail-count {{ color: var(--red); font-weight: 600; }}

    /* Cards */
    .cards {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }}
    .card-label {{ font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .card-value {{ font-size: 2rem; font-weight: 700; margin: 0.25rem 0; }}
    .card-sub {{ font-size: 0.8rem; color: var(--text-muted); }}

    /* Stage sections */
    .stage-section {{ margin-bottom: 2rem; }}
    .stage-header {{
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .stage-icon {{ font-size: 1.4rem; }}

    /* Tables */
    .results-table {{
        width: 100%;
        border-collapse: collapse;
        background: var(--surface);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border);
    }}
    .results-table th {{
        text-align: left;
        padding: 0.75rem 1rem;
        background: var(--surface2);
        color: var(--text-muted);
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .results-table td {{
        padding: 0.75rem 1rem;
        border-top: 1px solid var(--border);
        vertical-align: top;
    }}
    .scenario-row:hover {{ background: var(--surface2); }}
    .scenario-name {{ font-weight: 600; }}
    .scenario-id {{ font-size: 0.8rem; color: var(--text-muted); font-family: monospace; }}
    .error-msg {{ color: var(--red); font-size: 0.85rem; margin-top: 0.25rem; }}

    /* Status badges */
    .status-badge {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }}
    .status-badge.pass {{ background: rgba(34,197,94,0.15); color: var(--green); }}
    .status-badge.fail {{ background: rgba(239,68,68,0.15); color: var(--red); }}
    .status-badge.skip {{ background: rgba(245,158,11,0.15); color: var(--amber); }}

    /* Pills */
    .pill {{
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin: 2px;
        cursor: help;
    }}
    .pill-pass {{ background: rgba(34,197,94,0.15); color: var(--green); }}
    .pill-fail {{ background: rgba(239,68,68,0.15); color: var(--red); }}

    /* Tool cells */
    .tool-count {{ font-weight: 700; font-size: 1.1rem; }}
    .tools-detail {{ font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem; font-family: monospace; }}
    .duration-cell {{ font-family: monospace; color: var(--text-muted); }}
    .scores-cell {{ max-width: 300px; }}

    /* Breakdown section */
    .breakdown {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }}
    @media (max-width: 800px) {{ .breakdown {{ grid-template-columns: 1fr; }} }}
    .breakdown-panel {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
    }}
    .breakdown-panel h3 {{
        font-size: 1rem;
        margin-bottom: 0.75rem;
        color: var(--text-muted);
    }}
    .eval-table {{ width: 100%; border-collapse: collapse; }}
    .eval-table th {{
        text-align: left;
        padding: 0.5rem 0.75rem;
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        border-bottom: 1px solid var(--border);
    }}
    .eval-table td {{
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid rgba(71,85,105,0.3);
        font-size: 0.9rem;
    }}
    .eval-name, .tool-name {{ font-family: monospace; font-weight: 600; }}
    .bar-container {{
        width: 100%;
        height: 8px;
        background: var(--surface2);
        border-radius: 4px;
        overflow: hidden;
    }}
    .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s; }}

    /* Footer */
    .footer {{
        text-align: center;
        padding-top: 1.5rem;
        border-top: 1px solid var(--border);
        color: var(--text-muted);
        font-size: 0.8rem;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>⚡ Dataverse MCP Eval Report</h1>
        <div class="timestamp">{timestamp}</div>
    </div>

    <div class="hero">
        <div class="hero-score">{pass_rate}%</div>
        <div class="hero-label">Overall Pass Rate</div>
        <div class="hero-detail">
            <span><span class="pass-count">{passed}</span> passed</span>
            <span><span class="fail-count">{failed}</span> failed</span>
            <span>{total} total</span>
        </div>
    </div>

    <div class="cards">
        {summary_cards}
    </div>

    {stage_sections}

    <div class="breakdown">
        <div class="breakdown-panel">
            <h3>📊 Evaluator Breakdown</h3>
            {evaluator_breakdown}
        </div>
        <div class="breakdown-panel">
            <h3>🔧 Tool Call Summary</h3>
            {tool_call_timeline}
        </div>
    </div>

    {json_section}

    <div class="footer">
        Generated by <strong>dataverse-mcp-eval</strong> &middot; Model: gpt-4.1 via Azure AI Foundry
    </div>
</div>
</body>
</html>
"""
