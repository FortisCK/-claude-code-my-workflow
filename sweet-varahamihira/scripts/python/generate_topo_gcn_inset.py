#!/usr/bin/env python3
"""
Generate Topo-GCN inset figures for the TRUST framework diagram.

Purpose:
    Draw the actual 10-node anatomical landmark graph used by Topo-GCN.

Outputs:
    figs/topo_gcn_graph_icon.pdf
    figs/topo_gcn_graph_icon.png
    figs/topo_gcn_simple_icon.pdf
    figs/topo_gcn_simple_icon.png
    figs/topo_gcn_simple_tile.pdf
    figs/topo_gcn_simple_tile.png
    figs/topo_gcn_clean_icon.pdf
    figs/topo_gcn_clean_icon.png
    figs/topo_gcn_clean_icon_white.pdf
    figs/topo_gcn_clean_icon_white.png
    figs/topo_gcn_clean_tile.pdf
    figs/topo_gcn_clean_tile.png
    figs/topo_gcn_isometric_icon.pdf
    figs/topo_gcn_isometric_icon.png
    figs/topo_gcn_isometric_tile.pdf
    figs/topo_gcn_isometric_tile.png
    figs/topo_gcn_module_schematic.pdf
    figs/topo_gcn_module_schematic.png
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

Path(".mplconfig").mkdir(exist_ok=True)
Path(".cache/fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath


SEED = 42
DPI = 300
OUT_DIR = Path("figs")

random.seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"


NODE_POS = {
    "P0": (0.00, -1.10),   # NCC hinge
    "P1": (-1.00, 0.53),   # RCC hinge
    "P2": (1.00, 0.53),    # LCC hinge
    "P3": (-0.90, -0.67),  # NCC-RCC commissure
    "P4": (0.00, 1.16),    # RCC-LCC commissure
    "P5": (0.90, -0.67),   # LCC-NCC commissure
    "P6": (0.00, 0.00),    # valve center
    "P7": (-1.58, -0.22),  # membranous septum point
    "P8": (1.56, 0.95),    # left coronary ostium
    "P9": (-1.56, 0.95),   # right coronary ostium
}

RING_EDGES = [
    ("P0", "P3"),
    ("P3", "P1"),
    ("P1", "P4"),
    ("P4", "P2"),
    ("P2", "P5"),
    ("P5", "P0"),
]

CENTER_EDGES = [("P6", node) for node in NODE_POS if node != "P6"]

MS_EDGES = [("P7", "P0"), ("P7", "P1"), ("P7", "P3")]

OSTIUM_EDGES = [
    ("P8", "P2"),
    ("P8", "P4"),
    ("P8", "P5"),
    ("P9", "P1"),
    ("P9", "P3"),
    ("P9", "P4"),
]

NODE_STYLE = {
    "hinge": {
        "nodes": {"P0", "P1", "P2"},
        "face": "#8ecae6",
        "edge": "#2f6f9f",
    },
    "commissure": {
        "nodes": {"P3", "P4", "P5"},
        "face": "#ffd166",
        "edge": "#9c6b00",
    },
    "center": {
        "nodes": {"P6"},
        "face": "#95d5b2",
        "edge": "#2d6a4f",
    },
    "ms": {
        "nodes": {"P7"},
        "face": "#f4a6a0",
        "edge": "#a63c36",
    },
    "ostium": {
        "nodes": {"P8", "P9"},
        "face": "#cdb4db",
        "edge": "#6d3d8f",
    },
}


def node_style(node: str) -> tuple[str, str]:
    """Return face and edge color for a node."""
    for style in NODE_STYLE.values():
        if node in style["nodes"]:
            return style["face"], style["edge"]
    return "#ffffff", "#333333"


def draw_edges(
    ax: Axes,
    edges: Iterable[tuple[str, str]],
    color: str,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    """Draw undirected graph edges."""
    for src, dst in edges:
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=1,
        )


def draw_curved_edges(
    ax: Axes,
    edges: Iterable[tuple[str, str]],
    color: str,
    linewidth: float,
    alpha: float = 1.0,
    curvature: float = 0.18,
) -> None:
    """Draw curved undirected graph edges to reduce crossings in compact icons."""
    for src, dst in edges:
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dx, dy = x1 - x0, y1 - y0
        norm = (dx * dx + dy * dy) ** 0.5
        if norm == 0:
            continue
        ctrl = (mx - curvature * dy / norm, my + curvature * dx / norm)
        path = MplPath(
            [(x0, y0), ctrl, (x1, y1)],
            [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3],
        )
        patch = PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
        ax.add_patch(patch)


def draw_nodes(ax: Axes, node_size: float = 310, font_size: float = 7.6) -> None:
    """Draw colored landmark nodes and labels."""
    for node, (x, y) in NODE_POS.items():
        face, edge = node_style(node)
        ax.scatter(
            [x],
            [y],
            s=node_size,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            zorder=3,
        )
        ax.text(
            x,
            y,
            f"${node}$",
            ha="center",
            va="center",
            fontsize=font_size,
            fontweight="bold",
            zorder=4,
        )


def draw_topology_icon(ax: Axes, show_note: bool = True) -> None:
    """Draw the anatomical Topo-GCN graph."""
    draw_edges(ax, CENTER_EDGES, color="#9a9a9a", linewidth=0.55, alpha=0.58)
    draw_edges(ax, RING_EDGES, color="#343434", linewidth=1.25)
    draw_edges(ax, OSTIUM_EDGES, color="#6d3d8f", linewidth=0.85, alpha=0.82)
    draw_edges(ax, MS_EDGES, color="#a63c36", linewidth=0.95, alpha=0.88)
    draw_nodes(ax)

    if show_note:
        ax.text(
            0.0,
            -1.33,
            r"$\tilde{A}$ with self-loops; messages scaled by $w_i$",
            ha="center",
            va="center",
            fontsize=7.0,
            color="#2d6a4f",
        )

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.43, 1.23)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_simple_topology_icon(ax: Axes) -> None:
    """Draw a compact unlabeled icon using the true Topo-GCN topology."""
    # Self-loops are implicit in the normalized adjacency and omitted from the icon.
    draw_edges(ax, CENTER_EDGES, color="#5e5e5e", linewidth=1.15, alpha=0.62)
    draw_edges(ax, RING_EDGES, color="#4f7f43", linewidth=1.95, alpha=0.96)
    draw_edges(ax, OSTIUM_EDGES, color="#4f7f43", linewidth=1.70, alpha=0.90)
    draw_edges(ax, MS_EDGES, color="#3f3f3f", linewidth=1.55, alpha=0.88)

    for x, y in NODE_POS.values():
        ax.scatter(
            [x],
            [y],
            s=120,
            facecolor="#efe4f3",
            edgecolor="#151515",
            linewidth=1.45,
            zorder=3,
        )

    ax.set_xlim(-1.68, 1.68)
    ax.set_ylim(-1.38, 1.33)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_clean_topology_icon(ax: Axes) -> None:
    """Draw a clearer icon with true topology and visually separated edge groups."""
    center_outer_edges = [(src, dst) for src, dst in CENTER_EDGES if dst not in {"P0", "P4"}]
    center_vertical_edges = [("P6", "P0"), ("P6", "P4")]

    draw_edges(ax, center_outer_edges, color="#847a8a", linewidth=1.05, alpha=0.82, linestyle=(0, (2.0, 2.0)))
    draw_edges(ax, center_vertical_edges, color="#746b79", linewidth=1.15, alpha=0.82, linestyle=(0, (2.0, 2.0)))
    draw_edges(ax, RING_EDGES, color="#4f7f43", linewidth=2.05, alpha=0.98)

    draw_curved_edges(ax, OSTIUM_EDGES, color="#4f7f43", linewidth=1.65, alpha=0.90, curvature=0.14)
    draw_curved_edges(ax, MS_EDGES, color="#333333", linewidth=1.45, alpha=0.86, curvature=-0.16)

    for node, (x, y) in NODE_POS.items():
        node_size = 136 if node == "P6" else 126
        ax.scatter(
            [x],
            [y],
            s=node_size,
            facecolor="#f0e4f4",
            edgecolor="#151515",
            linewidth=1.42,
            clip_on=False,
            zorder=4,
        )

    ax.set_xlim(-1.88, 1.88)
    ax.set_ylim(-1.55, 1.54)
    ax.set_aspect("equal")
    ax.axis("off")


ISO_POS_3D = {
    "P0": (0.00, -1.10, 0.00),
    "P1": (-1.00, 0.53, 0.00),
    "P2": (1.00, 0.53, 0.00),
    "P3": (-0.90, -0.67, 0.00),
    "P4": (0.00, 1.16, 0.00),
    "P5": (0.90, -0.67, 0.00),
    "P6": (0.00, 0.00, 0.46),
    "P7": (-1.58, -0.22, -0.18),
    "P8": (1.56, 0.95, 0.34),
    "P9": (-1.56, 0.95, 0.34),
}


def project_iso(point: tuple[float, float, float]) -> tuple[float, float]:
    """Project a 3D point into a simple isometric view."""
    x, y, z = point
    return x + 0.32 * z, y + 0.46 * z


def draw_projected_edges(
    ax: Axes,
    edges: Iterable[tuple[str, str]],
    color: str,
    linewidth: float,
    alpha: float = 1.0,
    linestyle: str = "-",
) -> None:
    """Draw edges in the isometric projection."""
    for src, dst in edges:
        x0, y0 = project_iso(ISO_POS_3D[src])
        x1, y1 = project_iso(ISO_POS_3D[dst])
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=1,
        )


def draw_isometric_topology_icon(ax: Axes) -> None:
    """Draw a light 3D-style icon of the true Topo-GCN graph."""
    draw_projected_edges(ax, CENTER_EDGES, color="#a99faf", linewidth=0.78, alpha=0.50, linestyle=(0, (1.5, 2.1)))
    draw_projected_edges(ax, RING_EDGES, color="#4f7f43", linewidth=2.0, alpha=0.96)
    draw_projected_edges(ax, OSTIUM_EDGES, color="#4f7f43", linewidth=1.55, alpha=0.88)
    draw_projected_edges(ax, MS_EDGES, color="#333333", linewidth=1.35, alpha=0.82)

    for node, point in ISO_POS_3D.items():
        x, y = project_iso(point)
        node_size = 138 if node == "P6" else 126
        ax.scatter(
            [x],
            [y],
            s=node_size,
            facecolor="#f0e4f4",
            edgecolor="#151515",
            linewidth=1.38,
            zorder=4,
        )

    ax.set_xlim(-1.90, 1.95)
    ax.set_ylim(-1.35, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")


def save_figure(fig: plt.Figure, stem: str, transparent: bool = True) -> None:
    """Save PDF and PNG versions."""
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(
        pdf_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.04,
        format="pdf",
        transparent=transparent,
    )
    fig.savefig(
        png_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0.04,
        format="png",
        transparent=transparent,
    )
    plt.close(fig)


def generate_graph_icon() -> None:
    """Generate a compact graph icon for insertion inside Fig. 2."""
    fig, ax = plt.subplots(figsize=(2.0, 1.75))
    draw_topology_icon(ax, show_note=False)
    save_figure(fig, "topo_gcn_graph_icon")


def generate_simple_icon() -> None:
    """Generate the simplified unlabeled Topo-GCN icon for Fig. 2."""
    fig, ax = plt.subplots(figsize=(1.25, 1.05))
    draw_simple_topology_icon(ax)
    save_figure(fig, "topo_gcn_simple_icon")


def generate_simple_tile() -> None:
    """Generate a title-bearing tile matching the style of the framework box."""
    fig, ax = plt.subplots(figsize=(1.55, 1.65))
    fig.patch.set_facecolor("#d9c7df")
    ax.set_facecolor("#d9c7df")
    draw_simple_topology_icon(ax)
    ax.text(
        0.0,
        1.38,
        "Topo-GCN",
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color="#111111",
    )
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.25, 1.55)
    save_figure(fig, "topo_gcn_simple_tile", transparent=False)


def generate_clean_icon() -> None:
    """Generate a clearer 2D icon with true topology."""
    fig, ax = plt.subplots(figsize=(1.40, 1.05))
    draw_clean_topology_icon(ax)
    save_figure(fig, "topo_gcn_clean_icon")


def generate_clean_icon_white() -> None:
    """Generate a clearer 2D icon with a white background."""
    fig, ax = plt.subplots(figsize=(1.40, 1.05))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    draw_clean_topology_icon(ax)
    save_figure(fig, "topo_gcn_clean_icon_white", transparent=False)


def generate_clean_tile() -> None:
    """Generate a larger no-title 2D tile for preview or direct replacement."""
    fig, ax = plt.subplots(figsize=(1.70, 1.28))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    draw_clean_topology_icon(ax)
    ax.set_xlim(-1.93, 1.93)
    ax.set_ylim(-1.42, 1.42)
    save_figure(fig, "topo_gcn_clean_tile", transparent=False)


def generate_isometric_icon() -> None:
    """Generate a simple 3D-style icon with true topology."""
    fig, ax = plt.subplots(figsize=(1.45, 1.15))
    draw_isometric_topology_icon(ax)
    save_figure(fig, "topo_gcn_isometric_icon")


def generate_isometric_tile() -> None:
    """Generate a title-bearing 3D-style tile."""
    fig, ax = plt.subplots(figsize=(1.75, 1.70))
    fig.patch.set_facecolor("#d9c7df")
    ax.set_facecolor("#d9c7df")
    draw_isometric_topology_icon(ax)
    ax.text(
        0.05,
        1.86,
        "Topo-GCN",
        ha="center",
        va="center",
        fontsize=17.0,
        fontweight="bold",
        color="#111111",
    )
    ax.set_xlim(-1.95, 2.00)
    ax.set_ylim(-1.38, 2.05)
    save_figure(fig, "topo_gcn_isometric_tile", transparent=False)


def draw_arrow(ax: Axes, xy_start: tuple[float, float], xy_end: tuple[float, float], color: str) -> None:
    """Draw a block-diagram arrow."""
    arrow = FancyArrowPatch(
        xy_start,
        xy_end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=1.1,
        color=color,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def draw_box(ax: Axes, center: tuple[float, float], text: str, width: float, height: float, face: str) -> None:
    """Draw a rounded process box."""
    x, y = center
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.0,
        bbox={
            "boxstyle": "round,pad=0.28,rounding_size=0.12",
            "facecolor": face,
            "edgecolor": "#5b4b72",
            "linewidth": 1.0,
        },
        zorder=3,
    )
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)


def generate_module_schematic() -> None:
    """Generate a slightly larger Topo-GCN module schematic."""
    fig, ax = plt.subplots(figsize=(4.9, 1.70))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3.4)

    draw_box(ax, (0.8, 1.7), r"$\hat{p}_k$", 12, 3.4, "#eef3f8")
    draw_box(ax, (2.25, 1.7), "MLP\nin", 12, 3.4, "#eef3f8")

    graph_ax = ax.inset_axes([0.33, 0.18, 0.30, 0.72])
    draw_topology_icon(graph_ax, show_note=False)
    graph_ax.text(
        0.0,
        -1.30,
        r"$4\times$ gated MP",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#4a2c69",
    )

    draw_box(ax, (8.8, 1.7), r"$\gamma_k\odot\Delta p_k$", 12, 3.4, "#f1e4f7")
    draw_box(ax, (11.1, 1.7), r"$p_k^{ref}$", 12, 3.4, "#eef3f8")

    draw_arrow(ax, (1.25, 1.7), (1.83, 1.7), "#333333")
    draw_arrow(ax, (2.72, 1.7), (3.65, 1.7), "#333333")
    draw_arrow(ax, (7.58, 1.7), (8.22, 1.7), "#333333")
    draw_arrow(ax, (9.46, 1.7), (10.55, 1.7), "#333333")

    draw_arrow(ax, (5.75, 3.25), (5.75, 2.72), "#2d6a4f")
    ax.text(
        5.75,
        3.31,
        r"shared reliability $w_i$",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#2d6a4f",
    )
    ax.text(
        5.75,
        0.10,
        r"messages use $\tilde{A}_{ki}w_i$ and relative-position edge features $e_{ki}$",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#333333",
    )

    save_figure(fig, "topo_gcn_module_schematic")


def main() -> None:
    """Entry point."""
    generate_graph_icon()
    generate_simple_icon()
    generate_simple_tile()
    generate_clean_icon()
    generate_clean_icon_white()
    generate_clean_tile()
    generate_isometric_icon()
    generate_isometric_tile()
    generate_module_schematic()


if __name__ == "__main__":
    main()
