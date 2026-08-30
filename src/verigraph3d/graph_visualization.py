from __future__ import annotations

import html

from .graph import ExecutableSceneGraph


NODE_COLORS = {
    "object": "#dbeafe",
    "container": "#cffafe",
    "camera": "#fef3c7",
    "light": "#fef9c3",
    "material": "#fce7f3",
    "constraint": "#fee2e2",
    "action": "#dcfce7",
}


def graph_to_dot(graph: ExecutableSceneGraph, include_false: bool = False) -> str:
    """Export a self-contained Graphviz DOT representation."""
    lines = [
        "digraph VeriGraph3D {",
        '  graph [rankdir="LR", bgcolor="white"];',
        '  node [shape="box", style="rounded,filled", fontname="Arial"];',
        '  edge [fontname="Arial", fontsize="9"];',
    ]
    for node_id, node in sorted(graph.nodes.items()):
        kind = node.get("type", "node")
        title = node.get("name", node_id)
        subtitle = node.get("action_type") or node.get("group") or kind
        label = _escape(f"{title}\n[{subtitle}]")
        color = NODE_COLORS.get(kind, "#f3f4f6")
        lines.append(f'  "{_escape(node_id)}" [label="{label}", fillcolor="{color}"];')
    for fact in sorted(graph.facts.values(), key=lambda item: (item.subject, item.predicate, item.object or "")):
        if fact.object is None or fact.object not in graph.nodes or fact.subject not in graph.nodes:
            continue
        if not include_false and not bool(fact.value):
            continue
        style = "solid" if fact.confidence >= 0.8 else "dashed"
        color = "#111827" if bool(fact.value) else "#9ca3af"
        label = _escape(f"{fact.predicate} ({fact.confidence:.2f})")
        lines.append(
            f'  "{_escape(fact.subject)}" -> "{_escape(fact.object)}" '
            f'[label="{label}", color="{color}", style="{style}"];'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return html.escape(str(value), quote=True).replace("\n", "\\n")

