#!/usr/bin/env python3
"""
Open Kitchen Design Generator — Apartment 304
===============================================
Reads apartment dimensions from inputs/dimensions.json and generates
multiple open-kitchen floor-plan designs by removing:
  • 2 walls on the left side of the kitchen entrance
  • 2 parallel interior walls inside the kitchen

Run:
    python3 kitchen_design.py

Output:
    outputs/original_floor_plan.png
    outputs/design_1_fully_open_kitchen.png
    outputs/design_2_open_kitchen_with_island.png
    outputs/design_3_open_kitchen_breakfast_bar.png
    outputs/all_designs_comparison.png
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc, Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "bedroom":      "#FCE4EC",
    "kitchen":      "#FFF3E0",
    "dining":       "#E8F5E9",
    "living":       "#E3F2FD",
    "toilet":       "#F3E5F5",
    "balcony":      "#EFEBE9",
    "dry_balcony":  "#FBE9E7",
    "open_kitchen": "#FFFDE7",   # merged kitchen-dining area
    "island":       "#F9A825",
    "breakfast_bar":"#FFB300",
    "wall_line":    "#1A1A1A",
    "removed_wall": "#E53935",
    "new_feature":  "#1565C0",
    "dim_text":     "#424242",
    "hatch_balc":   "#BCAAA4",
}

WT = 0.15   # wall thickness (metres)


# ── Helper drawing functions ──────────────────────────────────────────────────

def draw_room(ax, x, y, w, h, label, dim_str, color, hatch=None, alpha=0.85, fontsize=7):
    """Draw a filled room rectangle with a label and dimension string."""
    rect = Rectangle((x, y), w, h,
                      linewidth=1.8, edgecolor=COLORS["wall_line"],
                      facecolor=color, alpha=alpha,
                      hatch=hatch, zorder=2)
    ax.add_patch(rect)
    cx, cy = x + w / 2, y + h / 2
    ax.text(cx, cy + 0.10, label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="#1A1A1A", zorder=5)
    if dim_str:
        ax.text(cx, cy - 0.22, dim_str,
                ha="center", va="center", fontsize=fontsize - 1,
                color=COLORS["dim_text"], zorder=5)


def draw_wall(ax, x1, y1, x2, y2, lw=3.5, color=None):
    """Draw a solid wall line."""
    color = color or COLORS["wall_line"]
    ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw,
            solid_capstyle="round", zorder=4)


def draw_removed_wall(ax, x1, y1, x2, y2):
    """Draw a wall marked for removal (red dashed with ✕ markers)."""
    ax.plot([x1, x2], [y1, y2],
            color=COLORS["removed_wall"], linewidth=2.5,
            linestyle="--", alpha=0.9, zorder=6)
    # Place ✕ symbols along the wall
    num_marks = max(2, int(np.hypot(x2 - x1, y2 - y1) / 0.6))
    for t in np.linspace(0.2, 0.8, num_marks):
        mx = x1 + t * (x2 - x1)
        my = y1 + t * (y2 - y1)
        ax.text(mx, my, "✕", ha="center", va="center",
                fontsize=7, color=COLORS["removed_wall"],
                fontweight="bold", zorder=7)


def draw_door(ax, x, y, width, swing_dir="up"):
    """Draw a simple door symbol (line + 90° arc)."""
    lw = 1.5
    if swing_dir == "up":
        ax.plot([x, x], [y, y + width * 0.08], color=COLORS["wall_line"], lw=lw, zorder=5)
        arc = Arc((x, y), width * 0.9, width * 0.9,
                  angle=0, theta1=0, theta2=90,
                  color=COLORS["wall_line"], lw=1, zorder=5)
        ax.add_patch(arc)
    elif swing_dir == "right":
        ax.plot([x, x + width * 0.08], [y, y], color=COLORS["wall_line"], lw=lw, zorder=5)
        arc = Arc((x, y), width * 0.9, width * 0.9,
                  angle=0, theta1=0, theta2=90,
                  color=COLORS["wall_line"], lw=1, zorder=5)
        ax.add_patch(arc)


def add_north_arrow(ax, x, y, size=0.4):
    """Add a north arrow indicator."""
    ax.annotate("", xy=(x, y + size), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))
    ax.text(x, y + size + 0.15, "N", ha="center", va="center",
            fontsize=7, fontweight="bold")


def add_legend(ax, extra_patches=None):
    """Add a legend to the axis."""
    legend_items = [
        mpatches.Patch(facecolor=COLORS["bedroom"],  label="Bedroom"),
        mpatches.Patch(facecolor=COLORS["kitchen"],  label="Kitchen"),
        mpatches.Patch(facecolor=COLORS["dining"],   label="Dining"),
        mpatches.Patch(facecolor=COLORS["living"],   label="Living"),
        mpatches.Patch(facecolor=COLORS["toilet"],   label="Toilet"),
        mpatches.Patch(facecolor=COLORS["balcony"],  label="Balcony"),
        mpatches.Patch(facecolor=COLORS["open_kitchen"], label="Open Kitchen-Dining"),
        mpatches.Patch(facecolor=COLORS["removed_wall"],
                       label="Removed wall", linestyle="--", linewidth=1),
    ]
    if extra_patches:
        legend_items.extend(extra_patches)
    ax.legend(handles=legend_items, loc="lower left", fontsize=5.5,
              framealpha=0.85, ncol=2)


# ── Floor-plan geometry (derived from dimensions.json) ───────────────────────

def build_layout(dims):
    """
    Compute (x, y, w, h) for every room, returning a dict.
    Origin (0,0) = south-west outer corner of the slab.
    x grows east, y grows north.
    """
    r = dims["rooms"]
    W = WT  # wall thickness alias

    # Column x-start positions (inner face of each room)
    x_bed2  = W
    x_bed1  = x_bed2 + r["bedroom_2"]["width_m"] + W
    x_kit   = x_bed1 + r["bedroom_1"]["width_m"] + W
    x_liv   = x_kit  + r["kitchen"]["width_m"]   + W

    # Row y-start positions (inner face, 0 = outer south wall face)
    y_balc  = W
    h_balc  = r["balcony_1"]["depth_m"]
    y_main  = y_balc + h_balc + W          # bottom of main rooms

    h_kit   = r["kitchen"]["depth_m"]
    h_bed2  = r["bedroom_2"]["depth_m"]
    h_bed1  = r["bedroom_1"]["depth_m"]
    h_liv   = r["living"]["depth_m"]

    y_din   = y_main + h_kit + W           # bottom of dining (above kitchen)
    h_din   = r["dining"]["depth_m"]

    # Toilets sit above bedrooms, aligned to northern wall of dining
    y_t1    = y_main + h_bed2 + W
    y_t2    = y_main + h_bed1 + W

    # Living room spans from y_main upward, height = living depth_m
    y_liv   = y_main + (h_kit - h_liv) / 2  # roughly centred east column

    # Dry balcony below kitchen on the south side
    w_dry   = r["dry_balcony"]["width_m"]
    h_dry   = r["dry_balcony"]["depth_m"]

    # East balcony (beside living room)
    x_ebalc = x_liv + r["living"]["width_m"] + W
    h_ebalc = r["balcony_east"]["depth_m"]
    w_ebalc = r["balcony_east"]["width_m"]
    y_ebalc = y_main

    layout = {
        # (x, y, width, height)
        "balcony_1":    (x_bed2, y_balc, r["balcony_1"]["width_m"],    h_balc),
        "balcony_2":    (x_bed1, y_balc, r["balcony_2"]["width_m"],    h_balc),
        "dry_balcony":  (x_kit,  y_balc, w_dry,                        h_dry),
        "bedroom_2":    (x_bed2, y_main, r["bedroom_2"]["width_m"],    h_bed2),
        "bedroom_1":    (x_bed1, y_main, r["bedroom_1"]["width_m"],    h_bed1),
        "kitchen":      (x_kit,  y_main, r["kitchen"]["width_m"],      h_kit),
        "dining":       (x_kit,  y_din,  r["dining"]["width_m"],       h_din),
        "living":       (x_liv,  y_liv,  r["living"]["width_m"],       h_liv),
        "toilet_1":     (x_bed2, y_t1,   r["toilet_1"]["width_m"],     r["toilet_1"]["depth_m"]),
        "toilet_2":     (x_bed1, y_t2,   r["toilet_2"]["width_m"],     r["toilet_2"]["depth_m"]),
        "balcony_east": (x_ebalc,y_ebalc,w_ebalc,                      h_ebalc),
    }
    return layout


def total_bounds(layout):
    """Return (xmin, ymin, xmax, ymax) bounding box of the entire plan."""
    xs = [v[0] for v in layout.values()] + [v[0] + v[2] for v in layout.values()]
    ys = [v[1] for v in layout.values()] + [v[1] + v[3] for v in layout.values()]
    return min(xs) - 0.3, min(ys) - 0.3, max(xs) + 0.3, max(ys) + 0.3


# ── Wall segments (the 4 walls to remove) ────────────────────────────────────

def kitchen_removal_walls(layout):
    """
    Return the 4 wall segments that must be removed to open up the kitchen.

    The kitchen is separated from the dining area by its north wall.
    When entering the kitchen from the dining area (walking south into kitchen):
      - Left side of entrance  = west portion of the kitchen's north wall
      - The return wall        = the short stub wall at the west entry corner

    Inside the kitchen, the two parallel walls are the west wall and a
    parallel interior partition running north-south inside the kitchen.

    Returns list of (x1,y1, x2,y2, label) tuples.
    """
    kx, ky, kw, kh = layout["kitchen"]
    dx, dy, dw, dh = layout["dining"]

    # North wall of kitchen / south wall of dining
    wall_top_y = ky + kh          # = dy (shared wall y-coordinate)

    # ── Entry walls (on the LEFT = west side when walking south into kitchen) ──
    # Wall 1: left (west) half of the shared north wall of kitchen
    left_seg = kw * 0.35          # ~35% of width is the left wall segment
    w1_x1, w1_y1 = kx,             wall_top_y
    w1_x2, w1_y2 = kx + left_seg, wall_top_y

    # Wall 2: short return/stub wall on the west side (vertical portion)
    stub_h = kh * 0.18            # short stub going down into the kitchen
    w2_x1, w2_y1 = kx,             wall_top_y
    w2_x2, w2_y2 = kx,             wall_top_y - stub_h

    # ── Interior parallel walls (inside kitchen, parallel to above) ───────────
    # Wall 3: internal horizontal partition at ~40% of kitchen depth from top
    inner_h_y = ky + kh * 0.62
    w3_x1, w3_y1 = kx,             inner_h_y
    w3_x2, w3_y2 = kx + left_seg, inner_h_y

    # Wall 4: internal vertical partition parallel to Wall 2 (west return wall)
    w4_x1, w4_y1 = kx,             inner_h_y
    w4_x2, w4_y2 = kx,             inner_h_y - stub_h

    return [
        (w1_x1, w1_y1, w1_x2, w1_y2, "Entry wall 1\n(left, north)"),
        (w2_x1, w2_y1, w2_x2, w2_y2, "Entry wall 2\n(left return)"),
        (w3_x1, w3_y1, w3_x2, w3_y2, "Parallel wall 3\n(interior horizontal)"),
        (w4_x1, w4_y1, w4_x2, w4_y2, "Parallel wall 4\n(interior vertical)"),
    ]


# ── Common room-drawing helper ────────────────────────────────────────────────

ROOM_META = {
    "balcony_1":   ("BALCONY",     "balcony",    True),
    "balcony_2":   ("BALCONY",     "balcony",    True),
    "dry_balcony": ("DRY\nBALCONY","dry_balcony",True),
    "balcony_east":("BALCONY",     "balcony",    True),
    "bedroom_2":   ("BEDROOM 2",   "bedroom",    False),
    "bedroom_1":   ("BEDROOM 1",   "bedroom",    False),
    "kitchen":     ("KITCHEN",     "kitchen",    False),
    "dining":      ("DINING",      "dining",     False),
    "living":      ("LIVING",      "living",     False),
    "toilet_1":    ("TOILET",      "toilet",     False),
    "toilet_2":    ("TOILET",      "toilet",     False),
}

def draw_all_rooms(ax, layout, dims, skip_rooms=None, override_colors=None):
    """Draw every room in the layout. Rooms in skip_rooms are omitted."""
    skip_rooms = skip_rooms or []
    override_colors = override_colors or {}
    r = dims["rooms"]

    room_dim_keys = {
        "bedroom_2":   ("bedroom_2",   "width_m", "depth_m"),
        "bedroom_1":   ("bedroom_1",   "width_m", "depth_m"),
        "kitchen":     ("kitchen",     "width_m", "depth_m"),
        "dining":      ("dining",      "width_m", "depth_m"),
        "living":      ("living",      "width_m", "depth_m"),
        "toilet_1":    ("toilet_1",    "width_m", "depth_m"),
        "toilet_2":    ("toilet_2",    "width_m", "depth_m"),
        "balcony_1":   ("balcony_1",   "width_m", "depth_m"),
        "balcony_2":   ("balcony_2",   "width_m", "depth_m"),
        "dry_balcony": ("dry_balcony", "width_m", "depth_m"),
        "balcony_east":("balcony_east","width_m", "depth_m"),
    }

    for room_key, (x, y, w, h) in layout.items():
        if room_key in skip_rooms:
            continue
        label, rtype, is_balcony = ROOM_META[room_key]
        color = override_colors.get(room_key, COLORS[rtype])
        hatch = "///" if is_balcony else None
        dk   = room_dim_keys.get(room_key)
        if dk and dk[0] in r:
            w_val = r[dk[0]][dk[1]]
            h_val = r[dk[0]][dk[2]]
            dim_str = f"{w_val:.2f} × {h_val:.2f} m"
        else:
            dim_str = f"{w:.2f} × {h:.2f} m"
        draw_room(ax, x, y, w, h, label, dim_str, color, hatch=hatch)


def style_ax(ax, title, bounds):
    """Apply common axis styling."""
    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=8)
    ax.set_xlabel("Width (m)", fontsize=7)
    ax.set_ylabel("Depth (m)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_facecolor("#FAFAFA")


# ══════════════════════════════════════════════════════════════════════════════
# Design 0 — Original Floor Plan
# ══════════════════════════════════════════════════════════════════════════════

def draw_original(ax, layout, dims):
    bounds = total_bounds(layout)
    draw_all_rooms(ax, layout, dims)

    kx, ky, kw, kh = layout["kitchen"]
    dy = ky + kh   # top of kitchen / bottom of dining – show shared wall
    draw_wall(ax, kx, dy, kx + kw, dy)

    # Stub walls at kitchen entrance
    stub_h = kh * 0.18
    left_seg = kw * 0.35
    draw_wall(ax, kx, dy, kx, dy - stub_h)                        # west return
    draw_wall(ax, kx + left_seg, dy, kx + left_seg, dy - stub_h)  # door jamb

    # Door symbol
    draw_door(ax, kx + left_seg, ky + kh - stub_h, kw * 0.3, swing_dir="up")

    add_north_arrow(ax, bounds[2] - 0.6, bounds[3] - 0.8)
    style_ax(ax,
             "ORIGINAL FLOOR PLAN\n(Apartment 304 – 2BHK, "
             f"C.A. {dims['apartment']['carpet_area_sqm']} m²)",
             bounds)

    # Annotate the 4 walls to be removed
    walls = kitchen_removal_walls(layout)
    for i, (x1, y1, x2, y2, lbl) in enumerate(walls):
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        ax.annotate(f"Remove\n{lbl}",
                    xy=(mx, my), xytext=(mx - 1.2, my + 0.4 * ((-1) ** i)),
                    fontsize=5, color=COLORS["removed_wall"],
                    arrowprops=dict(arrowstyle="->",
                                   color=COLORS["removed_wall"], lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=COLORS["removed_wall"], alpha=0.8),
                    zorder=8)

    add_legend(ax)


# ══════════════════════════════════════════════════════════════════════════════
# Design 1 — Fully Open Kitchen
# (All 4 walls removed; kitchen + dining merge into one open-plan area)
# ══════════════════════════════════════════════════════════════════════════════

def draw_design_1(ax, layout, dims):
    bounds = total_bounds(layout)

    kx, ky, kw, kh = layout["kitchen"]
    dx, dy, dw, dh = layout["dining"]

    # Merge kitchen + dining into one open room
    open_x = min(kx, dx)
    open_y = ky
    open_w = max(kx + kw, dx + dw) - open_x
    open_h = kh + dh + WT           # height = kitchen + dining + wall

    skip = {"kitchen", "dining"}
    draw_all_rooms(ax, layout, dims, skip_rooms=skip)
    draw_room(ax, open_x, open_y, open_w, open_h,
              "OPEN KITCHEN\n+ DINING",
              f"{open_w:.2f} × {open_h:.2f} m  (merged)",
              COLORS["open_kitchen"], fontsize=7)

    # Show the removed walls in red dashed
    walls = kitchen_removal_walls(layout)
    for x1, y1, x2, y2, _ in walls:
        draw_removed_wall(ax, x1, y1, x2, y2)

    # Suggest appliance positions as simple grey boxes
    # Counter along south wall
    counter_h = 0.60
    counter = Rectangle((kx + 0.05, ky + 0.05), kw - 0.10, counter_h,
                         linewidth=1, edgecolor="#607D8B",
                         facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(counter)
    ax.text(kx + kw / 2, ky + 0.05 + counter_h / 2, "Counter / Hob",
            ha="center", va="center", fontsize=5.5, color="#263238", zorder=5)

    # Sink on east wall
    sink = Rectangle((kx + kw - 0.65, ky + 0.75), 0.55, 0.45,
                      linewidth=1, edgecolor="#607D8B",
                      facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(sink)
    ax.text(kx + kw - 0.65 + 0.55 / 2, ky + 0.75 + 0.45 / 2, "Sink",
            ha="center", va="center", fontsize=5, color="#263238", zorder=5)

    add_north_arrow(ax, bounds[2] - 0.6, bounds[3] - 0.8)
    style_ax(ax,
             "DESIGN 1 — FULLY OPEN KITCHEN\n"
             "All 4 walls removed · Kitchen merges with Dining",
             bounds)

    extra = [
        mpatches.Patch(facecolor=COLORS["open_kitchen"], label="Open Kitchen-Dining"),
        mpatches.Patch(facecolor="#B0BEC5", label="Counter / Appliances"),
        mpatches.Patch(facecolor=COLORS["removed_wall"],
                       label="Removed wall", linestyle="--", linewidth=1),
    ]
    add_legend(ax, extra)


# ══════════════════════════════════════════════════════════════════════════════
# Design 2 — Semi-Open Kitchen with Central Island
# (Entry walls removed; interior parallel walls converted to an island counter)
# ══════════════════════════════════════════════════════════════════════════════

def draw_design_2(ax, layout, dims):
    bounds = total_bounds(layout)

    kx, ky, kw, kh = layout["kitchen"]
    dx, dy, dw, dh = layout["dining"]

    # Merge kitchen + dining (same as Design 1)
    open_x = min(kx, dx)
    open_y = ky
    open_w = max(kx + kw, dx + dw) - open_x
    open_h = kh + dh + WT

    skip = {"kitchen", "dining"}
    draw_all_rooms(ax, layout, dims, skip_rooms=skip)
    draw_room(ax, open_x, open_y, open_w, open_h,
              "OPEN KITCHEN + DINING",
              "", COLORS["open_kitchen"], fontsize=7)

    # Show removed entry walls
    walls = kitchen_removal_walls(layout)
    for x1, y1, x2, y2, _ in walls[:2]:    # only entry walls removed
        draw_removed_wall(ax, x1, y1, x2, y2)

    # Interior parallel walls become an island counter
    island_w = kw * 0.55
    island_h = 0.90
    island_x = kx + (kw - island_w) / 2
    island_y = ky + kh * 0.45

    island = FancyBboxPatch((island_x, island_y), island_w, island_h,
                             boxstyle="round,pad=0.05",
                             linewidth=2, edgecolor=COLORS["new_feature"],
                             facecolor=COLORS["island"], alpha=0.9, zorder=4)
    ax.add_patch(island)
    ax.text(island_x + island_w / 2, island_y + island_h / 2,
            "KITCHEN\nISLAND",
            ha="center", va="center", fontsize=6,
            fontweight="bold", color="#1A237E", zorder=6)
    ax.text(island_x + island_w / 2, island_y - 0.20,
            f"({island_w:.2f} × {island_h:.2f} m)",
            ha="center", va="center", fontsize=5,
            color=COLORS["dim_text"], zorder=6)

    # L-shaped counter along south + east walls
    counter_depth = 0.60
    s_counter = Rectangle((kx + 0.05, ky + 0.05), kw - 0.10, counter_depth,
                           linewidth=1, edgecolor="#607D8B",
                           facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(s_counter)
    ax.text(kx + kw / 2, ky + 0.05 + counter_depth / 2, "Hob / Counter",
            ha="center", va="center", fontsize=5, color="#263238", zorder=5)

    e_counter = Rectangle((kx + kw - counter_depth, ky + 0.05 + counter_depth),
                           counter_depth, kh - 0.15 - counter_depth,
                           linewidth=1, edgecolor="#607D8B",
                           facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(e_counter)
    ax.text(kx + kw - counter_depth / 2,
            ky + 0.05 + counter_depth + (kh - 0.15 - counter_depth) / 2,
            "Sink &\nStorage",
            ha="center", va="center", fontsize=5, color="#263238",
            rotation=90, zorder=5)

    # Annotation arrows
    ax.annotate("Kitchen Island\n(replaces interior walls)",
                xy=(island_x + island_w / 2, island_y + island_h),
                xytext=(island_x - 1.1, island_y + island_h + 0.5),
                fontsize=5.5, color=COLORS["new_feature"],
                arrowprops=dict(arrowstyle="->",
                                color=COLORS["new_feature"], lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=COLORS["new_feature"], alpha=0.85), zorder=8)

    add_north_arrow(ax, bounds[2] - 0.6, bounds[3] - 0.8)
    style_ax(ax,
             "DESIGN 2 — SEMI-OPEN KITCHEN WITH CENTRAL ISLAND\n"
             "Entry walls removed · Interior walls replaced by Island counter",
             bounds)

    extra = [
        mpatches.Patch(facecolor=COLORS["island"],
                       label="Kitchen Island (new feature)"),
        mpatches.Patch(facecolor="#B0BEC5", label="Counter / Appliances"),
        mpatches.Patch(facecolor=COLORS["removed_wall"],
                       label="Removed wall", linestyle="--", linewidth=1),
    ]
    add_legend(ax, extra)


# ══════════════════════════════════════════════════════════════════════════════
# Design 3 — Open Kitchen with Breakfast Bar / Pass-Through Counter
# (All 4 walls removed; a low breakfast bar defines the boundary)
# ══════════════════════════════════════════════════════════════════════════════

def draw_design_3(ax, layout, dims):
    bounds = total_bounds(layout)

    kx, ky, kw, kh = layout["kitchen"]
    dx, dy, dw, dh = layout["dining"]

    open_x = min(kx, dx)
    open_y = ky
    open_w = max(kx + kw, dx + dw) - open_x
    open_h = kh + dh + WT

    skip = {"kitchen", "dining"}
    draw_all_rooms(ax, layout, dims, skip_rooms=skip)
    draw_room(ax, open_x, open_y, open_w, open_h,
              "OPEN KITCHEN + DINING",
              "", COLORS["open_kitchen"], fontsize=7)

    # All 4 walls removed
    walls = kitchen_removal_walls(layout)
    for x1, y1, x2, y2, _ in walls:
        draw_removed_wall(ax, x1, y1, x2, y2)

    # Breakfast bar — low partition at the kitchen-dining boundary
    bar_y    = ky + kh - WT / 2
    bar_x    = kx
    bar_w    = kw
    bar_thick = 0.35   # shallow bar depth
    bar = Rectangle((bar_x, bar_y - bar_thick / 2), bar_w, bar_thick,
                     linewidth=2, edgecolor=COLORS["new_feature"],
                     facecolor=COLORS["breakfast_bar"],
                     alpha=0.9, zorder=4)
    ax.add_patch(bar)
    ax.text(bar_x + bar_w / 2, bar_y,
            "BREAKFAST BAR / PASS-THROUGH",
            ha="center", va="center", fontsize=5.5,
            fontweight="bold", color="#1A237E", zorder=6)

    # Stools on dining side (circles labelled "S")
    stool_y = bar_y + bar_thick / 2 + 0.22
    for i in range(3):
        sx = bar_x + 0.4 + i * 0.75
        stool = plt.Circle((sx, stool_y), 0.18,
                            color="#FFA000", alpha=0.9, zorder=4)
        ax.add_patch(stool)
        ax.text(sx, stool_y, "S", ha="center", va="center",
                fontsize=6, fontweight="bold", color="#5D4037", zorder=5)

    # Counter + appliances along south and west walls
    counter_d = 0.60
    s_ctr = Rectangle((kx + 0.05, ky + 0.05), kw - 0.10, counter_d,
                       linewidth=1, edgecolor="#607D8B",
                       facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(s_ctr)
    ax.text(kx + kw / 2, ky + 0.05 + counter_d / 2, "Hob / Oven",
            ha="center", va="center", fontsize=5, color="#263238", zorder=5)

    w_ctr = Rectangle((kx + 0.05, ky + counter_d + 0.10),
                       counter_d, kh - counter_d - 0.45,
                       linewidth=1, edgecolor="#607D8B",
                       facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(w_ctr)
    ax.text(kx + 0.05 + counter_d / 2,
            ky + counter_d + 0.10 + (kh - counter_d - 0.45) / 2,
            "Sink &\nStorage",
            ha="center", va="center", fontsize=5,
            color="#263238", rotation=90, zorder=5)

    # Annotation
    ax.annotate("Low breakfast bar\ndefines kitchen zone\nwithout enclosing it",
                xy=(bar_x + bar_w / 2, bar_y + bar_thick / 2),
                xytext=(bar_x - 1.5, bar_y + 0.9),
                fontsize=5.5, color=COLORS["new_feature"],
                arrowprops=dict(arrowstyle="->",
                                color=COLORS["new_feature"], lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=COLORS["new_feature"], alpha=0.85), zorder=8)

    add_north_arrow(ax, bounds[2] - 0.6, bounds[3] - 0.8)
    style_ax(ax,
             "DESIGN 3 — OPEN KITCHEN WITH BREAKFAST BAR\n"
             "All 4 walls removed · Low bar defines kitchen zone",
             bounds)

    extra = [
        mpatches.Patch(facecolor=COLORS["breakfast_bar"],
                       label="Breakfast Bar (low partition)"),
        mpatches.Patch(facecolor="#B0BEC5", label="Counter / Appliances"),
        mpatches.Patch(facecolor=COLORS["removed_wall"],
                       label="Removed wall", linestyle="--", linewidth=1),
    ]
    add_legend(ax, extra)


# ══════════════════════════════════════════════════════════════════════════════
# Design 4 — Open Kitchen with Peninsular Counter
# (Entry walls removed; a peninsular counter extends from east kitchen wall)
# ══════════════════════════════════════════════════════════════════════════════

def draw_design_4(ax, layout, dims):
    bounds = total_bounds(layout)

    kx, ky, kw, kh = layout["kitchen"]
    dx, dy, dw, dh = layout["dining"]

    open_x = min(kx, dx)
    open_y = ky
    open_w = max(kx + kw, dx + dw) - open_x
    open_h = kh + dh + WT

    skip = {"kitchen", "dining"}
    draw_all_rooms(ax, layout, dims, skip_rooms=skip)
    draw_room(ax, open_x, open_y, open_w, open_h,
              "OPEN KITCHEN + DINING",
              "", COLORS["open_kitchen"], fontsize=7)

    # All 4 walls removed
    walls = kitchen_removal_walls(layout)
    for x1, y1, x2, y2, _ in walls:
        draw_removed_wall(ax, x1, y1, x2, y2)

    # Peninsular counter extending from east wall inward
    pen_d   = 0.65     # depth (east-west)
    pen_h   = kh * 0.55
    pen_x   = kx + kw - pen_d
    pen_y   = ky + (kh - pen_h) / 2
    peninsula = FancyBboxPatch((pen_x, pen_y), pen_d, pen_h,
                                boxstyle="round,pad=0.05",
                                linewidth=2, edgecolor=COLORS["new_feature"],
                                facecolor=COLORS["island"],
                                alpha=0.9, zorder=4)
    ax.add_patch(peninsula)
    ax.text(pen_x + pen_d / 2, pen_y + pen_h / 2,
            "PENIN-\nSULA",
            ha="center", va="center", fontsize=5.5,
            fontweight="bold", color="#1A237E", zorder=6)

    # U-shaped counter
    counter_d = 0.58
    # south wall counter
    south_counter = Rectangle((kx + 0.05, ky + 0.05), kw - pen_d - 0.15, counter_d,
                              linewidth=1, edgecolor="#607D8B",
                              facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(south_counter)
    ax.text(kx + (kw - pen_d - 0.10) / 2, ky + 0.05 + counter_d / 2,
            "Hob / Counter", ha="center", va="center",
            fontsize=5, color="#263238", zorder=5)

    # west wall counter
    west_counter = Rectangle((kx + 0.05, ky + counter_d + 0.10),
                              counter_d, kh - counter_d - 0.20,
                              linewidth=1, edgecolor="#607D8B",
                              facecolor="#B0BEC5", alpha=0.8, zorder=3)
    ax.add_patch(west_counter)
    ax.text(kx + 0.05 + counter_d / 2,
            ky + counter_d + 0.10 + (kh - counter_d - 0.20) / 2,
            "Sink &\nStorage",
            ha="center", va="center", fontsize=5,
            color="#263238", rotation=90, zorder=5)

    # Stools at peninsula (dining side)
    for i in range(2):
        sy = pen_y + 0.3 + i * 0.70
        stool = plt.Circle((pen_x - 0.22, sy), 0.16,
                            color="#FFA000", alpha=0.85, zorder=4)
        ax.add_patch(stool)
        ax.text(pen_x - 0.22, sy, "S", fontweight="bold", color="#5D4037",
                ha="center", va="center", fontsize=6, zorder=5)

    ax.annotate("Peninsular counter\n— seating on dining side",
                xy=(pen_x, pen_y + pen_h / 2),
                xytext=(pen_x - 2.1, pen_y + pen_h / 2 + 0.7),
                fontsize=5.5, color=COLORS["new_feature"],
                arrowprops=dict(arrowstyle="->",
                                color=COLORS["new_feature"], lw=0.9),
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=COLORS["new_feature"], alpha=0.85), zorder=8)

    add_north_arrow(ax, bounds[2] - 0.6, bounds[3] - 0.8)
    style_ax(ax,
             "DESIGN 4 — OPEN KITCHEN WITH PENINSULAR COUNTER\n"
             "All 4 walls removed · Peninsula extends into open space",
             bounds)

    extra = [
        mpatches.Patch(facecolor=COLORS["island"],
                       label="Peninsular Counter (new feature)"),
        mpatches.Patch(facecolor="#B0BEC5", label="Counter / Appliances"),
        mpatches.Patch(facecolor=COLORS["removed_wall"],
                       label="Removed wall", linestyle="--", linewidth=1),
    ]
    add_legend(ax, extra)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load dimensions ───────────────────────────────────────────────────────
    dims_path = os.path.join(os.path.dirname(__file__), "inputs", "dimensions.json")
    if not os.path.exists(dims_path):
        print(f"ERROR: Dimensions file not found at {dims_path}", file=sys.stderr)
        sys.exit(1)

    with open(dims_path) as f:
        dims = json.load(f)

    print(f"Loaded dimensions for Apartment {dims['apartment']['number']} "
          f"({dims['apartment']['type']}, "
          f"C.A. {dims['apartment']['carpet_area_sqm']} m²)")

    # ── Build layout ──────────────────────────────────────────────────────────
    layout = build_layout(dims)

    # ── Create output directory ───────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Individual: Original floor plan ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F5F5F5")
    draw_original(ax, layout, dims)
    path = os.path.join(out_dir, "original_floor_plan.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 2. Individual: Design 1 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F5F5F5")
    draw_design_1(ax, layout, dims)
    path = os.path.join(out_dir, "design_1_fully_open_kitchen.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 3. Individual: Design 2 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F5F5F5")
    draw_design_2(ax, layout, dims)
    path = os.path.join(out_dir, "design_2_open_kitchen_with_island.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 4. Individual: Design 3 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F5F5F5")
    draw_design_3(ax, layout, dims)
    path = os.path.join(out_dir, "design_3_open_kitchen_breakfast_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 5. Individual: Design 4 ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#F5F5F5")
    draw_design_4(ax, layout, dims)
    path = os.path.join(out_dir, "design_4_open_kitchen_peninsula.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # ── 6. Comparison sheet (2×3 grid) ───────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(26, 16))
    fig.patch.set_facecolor("#ECEFF1")
    fig.suptitle(
        "OPEN KITCHEN DESIGN OPTIONS — Apartment 304  |  "
        f"Kitchen {dims['rooms']['kitchen']['width_m']} × "
        f"{dims['rooms']['kitchen']['depth_m']} m  ·  "
        f"Dining {dims['rooms']['dining']['width_m']} × "
        f"{dims['rooms']['dining']['depth_m']} m\n"
        "4 walls removed: 2 on left side of kitchen entrance + "
        "2 parallel interior walls",
        fontsize=13, fontweight="bold", y=0.99
    )

    draw_original(axes[0, 0], layout, dims)
    draw_design_1(axes[0, 1], layout, dims)
    draw_design_2(axes[0, 2], layout, dims)
    draw_design_3(axes[1, 0], layout, dims)
    draw_design_4(axes[1, 1], layout, dims)

    # Summary panel
    axes[1, 2].axis("off")
    summary = (
        "OPEN KITCHEN RENOVATION SUMMARY\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Apartment 304 — 2BHK\n"
        f"Carpet Area : {dims['apartment']['carpet_area_sqm']} m²\n"
        f"Super BUA   : {dims['apartment']['super_built_up_area_sqm']} m²\n\n"
        "Kitchen Dimensions:\n"
        f"  {dims['rooms']['kitchen']['width_m']} m × "
        f"{dims['rooms']['kitchen']['depth_m']} m\n\n"
        "Dining Dimensions:\n"
        f"  {dims['rooms']['dining']['width_m']} m × "
        f"{dims['rooms']['dining']['depth_m']} m\n\n"
        "Walls to Remove:\n"
        "  • Entry wall 1 — left north wall segment\n"
        "  • Entry wall 2 — left return/stub wall\n"
        "  • Parallel wall 3 — interior horizontal\n"
        "  • Parallel wall 4 — interior vertical\n\n"
        "Combined Open Area After Removal:\n"
        f"  ≈ {dims['rooms']['kitchen']['width_m']} m × "
        f"({dims['rooms']['kitchen']['depth_m']} + "
        f"{dims['rooms']['dining']['depth_m']:.2f}) m\n"
        f"  = {dims['rooms']['kitchen']['width_m'] * (dims['rooms']['kitchen']['depth_m'] + dims['rooms']['dining']['depth_m']):.1f} m²\n\n"
        "Design Options:\n"
        "  1. Fully Open Kitchen (complete merge)\n"
        "  2. Kitchen Island (social cooking hub)\n"
        "  3. Breakfast Bar (low visual separation)\n"
        "  4. Peninsular Counter (U-shape work zone)\n\n"
        "⚠  Consult a structural engineer before\n"
        "    removing any load-bearing walls."
    )
    axes[1, 2].text(0.05, 0.97, summary,
                     transform=axes[1, 2].transAxes,
                     fontsize=9, va="top", ha="left",
                     family="monospace",
                     bbox=dict(boxstyle="round,pad=0.6",
                               facecolor="#FFFFFF", edgecolor="#455A64",
                               linewidth=1.5))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "all_designs_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    print("\nDone! All designs saved to the 'outputs/' directory.")
    print("Files generated:")
    for f in sorted(os.listdir(out_dir)):
        print(f"  outputs/{f}")


if __name__ == "__main__":
    main()
