"""
Exercise 3 — Digit Classification MLP overview.

Five sections:
  1. Architecture  — 784 → 256 → 128 → 10, forward-pass animation
  2. Data          — combined dataset, augmentation, class imbalance
  3. Experiments   — 5 strategies, animated accuracy bar chart
  4. Results       — per-class accuracy + key metrics for the best model
  5. Takeaway      — summary card

Run via render_nn_overview.sh (Docker) or:
  manim -qh nn_overview_scene.py Exercise3NNOverview
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (
    LEFT, RIGHT, UP, DOWN, ORIGIN,
    WHITE, GRAY,
    AnimationGroup, Arrow, Circle, Create, Dot, FadeIn, FadeOut,
    Flash, GrowArrow, Indicate, LaggedStart, Line, MoveAlongPath,
    Rectangle, RoundedRectangle, Scene, Square, Text, VGroup,
    always_redraw,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "nn_overview.json"


def _load_data() -> dict:
    override = os.environ.get("EX3_VIS_DATA")
    path = Path(override).expanduser() if override else DATA_PATH
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Palette & typography  (matches Exercise 1)
# ---------------------------------------------------------------------------
BG_COLOR    = "#08111f"
CARD_FILL   = "#0f172a"
CARD_STROKE = "#334155"
MUTED       = "#94a3b8"
ACCENT      = "#38bdf8"
POSITIVE    = "#fb923c"
NEGATIVE    = "#60a5fa"
SUCCESS     = "#22c55e"
WARNING     = "#facc15"

# Layer colours: input → hidden1 → hidden2 → output
L_COLORS = ["#94a3b8", "#818cf8", "#a78bfa", "#22c55e"]

TITLE_FONT = "Noto Serif Display"
BODY_FONT  = "Nimbus Sans"
MONO_FONT  = "Nimbus Mono PS"

FRAME_W = 13.25
SECTION_LABELS = ["Architecture", "Data", "Experiments", "Results", "Takeaway"]

# Digit names for axis labels
DIGIT_LABELS = [str(d) for d in range(10)]

# Pixel pattern for a "7" in a 7×7 grid (row-major, 1 = lit)
DIGIT_7_PATTERN = [
    [1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _t(s: str, fs: int, color: str = WHITE) -> Text:
    return Text(s, font_size=fs, color=color, font=TITLE_FONT)

def _b(s: str, fs: int, color: str = WHITE) -> Text:
    return Text(s, font_size=fs, color=color, font=BODY_FONT)

def _m(s: str, fs: int, color: str = WHITE) -> Text:
    return Text(s, font_size=fs, color=color, font=MONO_FONT)

def _card(w: float, h: float, fill: str = CARD_FILL) -> RoundedRectangle:
    return RoundedRectangle(
        corner_radius=0.18, width=w, height=h,
        fill_color=fill, fill_opacity=0.93,
        stroke_color=CARD_STROKE, stroke_width=1.6,
    )

def _card_group(title: str, body: VGroup, w: float, h: float) -> VGroup:
    t = _t(title, 24)
    avail_w = w - 0.84
    avail_h = h - 0.84
    if t.width > avail_w:
        t.scale_to_fit_width(avail_w)
    if body.width > avail_w:
        body.scale_to_fit_width(avail_w)
    content = VGroup(t, body).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
    if content.height > avail_h:
        content.scale_to_fit_height(avail_h)
    box = _card(w, h)
    box.move_to(content.get_center())
    return VGroup(box, content)

def _pill(label: str, value: str, color: str) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.18, width=2.4, height=1.1,
        fill_color=CARD_FILL, fill_opacity=0.96,
        stroke_color=color, stroke_width=2.0,
    )
    title = _b(label, 18, MUTED)
    val   = _m(value, 26, color)
    content = VGroup(title, val).arrange(DOWN, buff=0.04)
    content.move_to(box.get_center())
    return VGroup(box, content)

def _pipe_block(title: str, sub: str, w: float, h: float, accent: str) -> VGroup:
    box = RoundedRectangle(
        corner_radius=0.16, width=w, height=h,
        fill_color=CARD_FILL, fill_opacity=0.96,
        stroke_color=accent, stroke_width=2.0,
    )
    t = _t(title, 20)
    s = _b(sub, 16, MUTED)
    content = VGroup(t, s).arrange(DOWN, buff=0.08)
    content.move_to(box.get_center())
    return VGroup(box, content)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
class Exercise3NNOverview(Scene):

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR
        data = _load_data()
        header = self._build_header(data)

        self._section_1_architecture(header, data)
        self._section_2_data(header, data)
        self._section_3_experiments(header, data)
        self._section_4_results(header, data)
        self._section_5_takeaway(header, data)

    # -----------------------------------------------------------------------
    # Header & stepper
    # -----------------------------------------------------------------------
    def _build_header(self, data: dict) -> VGroup:
        title    = _t(data["title"], 34)
        subtitle = _b("784 → 256 → 128 → 10   |   Adam · Leaky ReLU · min-max", 18, MUTED)
        header_text = VGroup(title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        header_text.to_corner(UP + LEFT, buff=0.36)

        self._stepper_dots: list[Circle] = []
        dots = []
        for _ in SECTION_LABELS:
            d = Circle(radius=0.08, color=CARD_STROKE, stroke_width=1.6)
            d.set_fill(CARD_STROKE, opacity=0.35)
            dots.append(d)
            self._stepper_dots.append(d)
        stepper = VGroup(*dots).arrange(RIGHT, buff=0.18)
        caption = _b("story flow", 13, MUTED)
        caption.next_to(stepper, DOWN, buff=0.12)
        stepper_group = VGroup(stepper, caption)
        stepper_group.to_corner(UP + RIGHT, buff=0.46)

        divider = Line(
            LEFT * (FRAME_W / 2), RIGHT * (FRAME_W / 2),
            color=CARD_STROKE, stroke_width=1.2,
        )
        divider.move_to(np.array([0.0, 2.82, 0.0]))

        header = VGroup(header_text, stepper_group, divider)
        self.play(FadeIn(header, shift=DOWN * 0.2))
        return header

    def _step(self, idx: int) -> AnimationGroup:
        anims = []
        for i, dot in enumerate(self._stepper_dots):
            if i == idx:
                anims.append(dot.animate.set_stroke(ACCENT, 2.0).set_fill(ACCENT, 1.0))
            else:
                anims.append(dot.animate.set_stroke(CARD_STROKE, 1.6).set_fill(CARD_STROKE, 0.35))
        return AnimationGroup(*anims, run_time=0.45)

    def _sec_title(self, label: str, header: VGroup) -> Text:
        t = _t(label, 28, ACCENT)
        t.next_to(header, DOWN, aligned_edge=LEFT, buff=0.34)
        return t

    # -----------------------------------------------------------------------
    # Section 1: Architecture
    # -----------------------------------------------------------------------
    def _section_1_architecture(self, header: VGroup, data: dict) -> None:
        sec = self._sec_title("1. Network architecture", header)
        self.play(self._step(0), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.8)

        arch = data["architecture"]  # [784, 256, 128, 10]
        mlp  = self._build_mlp(arch)

        # Build layers one by one
        self.play(LaggedStart(
            *[FadeIn(g, shift=RIGHT * 0.12) for g in mlp["layer_groups"]],
            lag_ratio=0.35, run_time=2.0,
        ))
        self.wait(1.0)
        # Draw edges between each pair of layers
        for edge_group in mlp["edge_groups"]:
            self.play(LaggedStart(
                *[Create(e) for e in edge_group],
                lag_ratio=0.012, run_time=1.1,
            ))
        self.wait(1.0)

        # Forward-pass pulse animation
        self._animate_forward_pass(mlp)
        self.wait(1.5)

        # Show activation-function note
        act_note = VGroup(
            _b("hidden layers", 17, MUTED),
            _m("Leaky ReLU  f(x) = x  if x > 0", 16, L_COLORS[1]),
            _m("             αx  if x ≤ 0  (α=0.01)", 16, L_COLORS[2]),
            _b("output layer", 17, MUTED),
            _m("Logistic  f(x) = σ(x)  ∈ (0, 1)", 16, L_COLORS[3]),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        act_card = _card_group("Activation functions", act_note, 5.6, 3.0)
        act_card.move_to(np.array([3.8, -1.0, 0.0]))
        self.play(FadeIn(act_card, shift=UP * 0.12))
        self.wait(3.5)

        self.play(FadeOut(mlp["all"], shift=LEFT * 0.2), FadeOut(act_card), FadeOut(sec))

    def _build_mlp(self, arch: list[int]) -> dict:
        """Build visible MLP nodes and sampled edges."""
        # How many circles to show per layer
        visible = [6, 5, 4, 10]
        x_pos   = [-5.4, -2.3, 0.8, 4.4]
        radii   = [0.15, 0.155, 0.16, 0.17]

        layer_nodes: list[list[Circle]] = []
        layer_groups: list[VGroup] = []

        for li, (x, n_vis, color, real_n, r) in enumerate(
            zip(x_pos, visible, L_COLORS, arch, radii)
        ):
            y_spread = (n_vis - 1) * 0.66
            ys = np.linspace(y_spread / 2, -y_spread / 2, n_vis)
            nodes = []
            mobjs = []
            for ni, y in enumerate(ys):
                circle = Circle(radius=r, color=color, stroke_width=2.0)
                circle.set_fill(color, opacity=0.18)
                circle.move_to(np.array([x, float(y), 0.0]))
                nodes.append(circle)
                mobjs.append(circle)

                # Digit label in output layer
                if li == 3:
                    lbl = _b(str(ni), 17, WHITE)
                    lbl.move_to(circle.get_center())
                    mobjs.append(lbl)

            # Layer annotations
            size_lbl = _m(f"×{real_n}", 16, color)
            size_lbl.next_to(
                VGroup(*nodes), DOWN, buff=0.22,
            )
            act_lbl = _b(
                "input" if li == 0 else ("leaky relu" if li < 3 else "logistic"),
                15, MUTED,
            )
            act_lbl.next_to(size_lbl, DOWN, buff=0.08)
            mobjs += [size_lbl, act_lbl]

            layer_nodes.append(nodes)
            layer_groups.append(VGroup(*mobjs))

        # Pixel-grid icon left of the input layer
        pixel_grid = self._make_pixel_grid()
        pixel_grid.next_to(layer_groups[0], LEFT, buff=0.55)
        grid_arrow = Arrow(
            pixel_grid.get_right(), layer_groups[0].get_left(),
            buff=0.1, color=MUTED, stroke_width=2,
        )
        input_label = _b("28 × 28\npixels", 16, MUTED)
        input_label.next_to(pixel_grid, DOWN, buff=0.14)
        pixel_group = VGroup(pixel_grid, grid_arrow, input_label)

        # Sampled edges between consecutive layers
        rng = np.random.default_rng(7)
        edge_groups: list[list[Line]] = []
        for li in range(len(layer_nodes) - 1):
            edges = []
            for fn in layer_nodes[li]:
                for tn in layer_nodes[li + 1]:
                    weight_sign = rng.choice([-1, 1])
                    col = POSITIVE if weight_sign > 0 else NEGATIVE
                    e = Line(
                        fn.get_center(), tn.get_center(),
                        color=col, stroke_width=0.55, stroke_opacity=0.18,
                    )
                    edges.append(e)
            edge_groups.append(edges)

        all_edges = VGroup(*[e for eg in edge_groups for e in eg])
        all_mobs  = VGroup(pixel_group, *layer_groups, all_edges)

        return {
            "layer_nodes": layer_nodes,
            "layer_groups": [pixel_group, *layer_groups],
            "edge_groups": edge_groups,
            "all": all_mobs,
        }

    def _make_pixel_grid(self) -> VGroup:
        """7×7 grid representing the digit '7'."""
        cell  = 0.145
        gap   = 0.03
        step  = cell + gap
        cells = []
        for row_i, row in enumerate(DIGIT_7_PATTERN):
            for col_i, val in enumerate(row):
                sq = Square(
                    side_length=cell,
                    fill_color=WHITE if val else CARD_FILL,
                    fill_opacity=0.88 if val else 0.55,
                    stroke_color=CARD_STROKE,
                    stroke_width=0.5,
                )
                sq.move_to(np.array([col_i * step, -row_i * step, 0.0]))
                cells.append(sq)
        grid = VGroup(*cells)
        grid.move_to(ORIGIN)
        return grid

    def _animate_forward_pass(self, mlp: dict) -> None:
        """Animate signal pulses flowing left → right through the network."""
        layer_nodes = mlp["layer_nodes"]
        rng = np.random.default_rng(3)

        # Pick representative paths: one active neuron per layer
        active_idxs = [rng.integers(0, len(layer_nodes[li])) for li in range(len(layer_nodes))]
        # Force output to digit 7 — consistent with the pixel grid that shows a "7"
        active_idxs[-1] = 7

        pulses   = []
        paths    = []
        for li in range(len(layer_nodes) - 1):
            src = layer_nodes[li][active_idxs[li]]
            dst = layer_nodes[li + 1][active_idxs[li + 1]]
            path = Line(src.get_center(), dst.get_center())
            dot  = Dot(radius=0.08, color=WARNING)
            dot.move_to(src.get_center())
            pulses.append(dot)
            paths.append(path)
            self.add(dot)

        # Light up input neurons
        self.play(LaggedStart(
            *[Flash(n, color=L_COLORS[0], flash_radius=0.28, line_length=0.12)
              for n in layer_nodes[0]],
            lag_ratio=0.12, run_time=1.0,
        ))

        # Propagate through layers
        for li in range(len(pulses)):
            self.play(
                MoveAlongPath(pulses[li], paths[li]),
                run_time=0.8,
            )
            dst_node = layer_nodes[li + 1][active_idxs[li + 1]]
            self.play(
                Indicate(dst_node, color=L_COLORS[li + 1], scale_factor=1.4),
                run_time=0.5,
            )

        # Output decision light-up
        out_digit = active_idxs[-1]
        self.play(Flash(
            layer_nodes[-1][out_digit],
            color=SUCCESS, flash_radius=0.38, line_length=0.16,
            run_time=0.8,
        ))
        for p in pulses:
            self.remove(p)

    # -----------------------------------------------------------------------
    # Section 2: Data pipeline
    # -----------------------------------------------------------------------
    def _section_2_data(self, header: VGroup, data: dict) -> None:
        sec = self._sec_title("2. Data pipeline", header)
        cfg = data["config"]
        ds  = data["dataset"]

        self.play(self._step(1), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.8)

        # Dataset sources → combined
        digits_box  = _pipe_block("digits.csv",      "original training set", 3.4, 1.2, ACCENT)
        more_box    = _pipe_block("more_digits.csv", "extra collected data",  3.4, 1.2, NEGATIVE)
        combined    = _pipe_block(
            "Combined dataset",
            f"{ds['train_total'] + ds['val_total']:,} samples  (augmented train)",
            3.8, 1.2, SUCCESS,
        )
        digits_box.move_to(np.array([-4.5,  0.8, 0.0]))
        more_box.move_to(  np.array([-4.5, -0.8, 0.0]))
        combined.move_to(  np.array([-0.2,  0.0, 0.0]))

        arr1 = Arrow(digits_box.get_right(), combined.get_left(), buff=0.12, color=MUTED, stroke_width=2.5)
        arr2 = Arrow(more_box.get_right(),   combined.get_left(), buff=0.12, color=MUTED, stroke_width=2.5)

        self.play(FadeIn(digits_box, shift=RIGHT * 0.12), FadeIn(more_box, shift=RIGHT * 0.12))
        self.wait(1.0)
        self.play(GrowArrow(arr1), GrowArrow(arr2))
        self.play(FadeIn(combined, shift=RIGHT * 0.12))
        self.wait(1.5)

        # Augmentation note
        aug_lines = VGroup(
            _b("pixel shift ± 2 px    →   translation invariance", 17, WHITE),
            _m(f"Gaussian noise  σ = {cfg['augment_noise_std']}  →  robustness", 17, MUTED),
            _b("1× repeat  →  dataset size doubles", 17, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        aug_card = _card_group("Augmentation", aug_lines, 6.2, 2.0)
        aug_card.move_to(np.array([3.8, 0.6, 0.0]))
        self.play(FadeIn(aug_card, shift=LEFT * 0.12))
        self.wait(2.5)

        # Class-imbalance bar chart (digits 0-9, training support from baseline)
        imb_chart = self._build_class_bars(
            data["experiments"][0]["per_class_support"] if data["experiments"] else [0] * 10,
            title="Test-set class support",
        )
        imb_chart.move_to(np.array([3.8, -1.8, 0.0]))
        self.play(FadeIn(imb_chart, shift=UP * 0.12))
        self.wait(2.5)

        # Strategies card
        strat_lines = VGroup(
            _b("1 — Baseline              (no rebalancing)", 16, MUTED),
            _b("2 — Weighted Loss          (scale gradient by class weight)", 16, MUTED),
            _b("3 — Weighted Sampling     (oversample rare classes)", 16, SUCCESS),
            _b("4 — Synthetic Balancing   (shift+noise augment minority)", 16, MUTED),
            _b("5 — SMOTE                  (k-NN interpolation)", 16, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        strat_card = _card_group("Five training strategies", strat_lines, 8.8, 2.2)
        strat_card.move_to(np.array([0.0, -2.5, 0.0]))
        self.play(FadeIn(strat_card, shift=UP * 0.12))
        self.wait(4.0)

        all_g = VGroup(digits_box, more_box, combined, arr1, arr2, aug_card, imb_chart, strat_card)
        self.play(FadeOut(all_g, shift=LEFT * 0.2), FadeOut(sec))

    def _build_class_bars(self, supports: list[int], title: str) -> VGroup:
        if not supports:
            return VGroup()
        max_s = max(supports) if supports else 1
        bar_h = 0.22
        bar_max_w = 2.2
        rows = []
        for digit, s in enumerate(supports):
            lbl  = _b(str(digit), 16, MUTED)
            fill_w = max(bar_max_w * s / max_s, 0.05)
            bar = RoundedRectangle(
                corner_radius=0.04, width=fill_w, height=bar_h,
                fill_color=ACCENT, fill_opacity=0.75, stroke_width=0,
            )
            val  = _m(str(s), 14, MUTED)
            row  = VGroup(lbl, bar, val).arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
            rows.append(row)
        body = VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        return _card_group(title, body, 3.8, 3.2)

    # -----------------------------------------------------------------------
    # Section 3: Experiments
    # -----------------------------------------------------------------------
    def _section_3_experiments(self, header: VGroup, data: dict) -> None:
        sec = self._sec_title("3. Five experiments — accuracy comparison", header)
        self.play(self._step(2), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.8)

        experiments = data["experiments"]
        best_idx    = data["best_experiment_index"]
        target      = 0.98

        chart = self._build_experiment_chart(experiments, best_idx, target)
        self.play(FadeIn(chart["frame"]))
        self.wait(0.4)
        for i, bar_anim in enumerate(chart["bar_anims"]):
            self.play(bar_anim, run_time=0.7)
            self.play(FadeIn(chart["val_labels"][i], shift=RIGHT * 0.08), run_time=0.4)
            self.wait(0.3)

        self.play(Create(chart["ref_line"]), FadeIn(chart["ref_label"]), run_time=0.7)
        self.wait(0.5)
        self.play(Indicate(chart["bars"][best_idx], color=SUCCESS, scale_factor=1.03))
        self.wait(2.5)

        # Delta-from-baseline note
        if experiments:
            base_acc = experiments[0]["test_accuracy"]
            delta_lines = []
            for exp in experiments:
                delta = exp["test_accuracy"] - base_acc
                sign  = "+" if delta >= 0 else ""
                color = SUCCESS if delta > 0.001 else (WARNING if abs(delta) < 0.001 else MUTED)
                delta_lines.append(_m(
                    f"{exp['name']:<28} acc {exp['test_accuracy']:.2%}  ({sign}{delta:.2%})",
                    15, color,
                ))
            delta_card = _card_group(
                "vs baseline",
                VGroup(*delta_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.10),
                9.0, 2.2,
            )
            delta_card.move_to(np.array([0.0, -2.7, 0.0]))
            self.play(FadeIn(delta_card, shift=UP * 0.1))
            self.wait(4.0)

        self.play(FadeOut(chart["all"]),
                  FadeOut(delta_card if experiments else VGroup()),
                  FadeOut(sec))

    def _build_experiment_chart(
        self, experiments: list[dict], best_idx: int, target: float
    ) -> dict:
        bar_h   = 0.55
        gap     = 0.18
        bar_max = 7.2
        acc_min = 0.975
        acc_max = 0.985
        x0      = -3.8
        y_top   = 2.0

        frame_h = len(experiments) * (bar_h + gap) + 0.6
        frame   = _card(bar_max + 2.4, frame_h)
        frame.move_to(np.array([x0 + bar_max / 2 + 0.6, y_top - frame_h / 2 - 0.05, 0.0]))

        bars        = []
        bar_anims   = []
        val_labels  = []
        name_labels = []

        for i, exp in enumerate(experiments):
            y = y_top - i * (bar_h + gap) - bar_h / 2
            color = SUCCESS if i == best_idx else ACCENT
            acc_clamp = min(max(exp["test_accuracy"], acc_min), acc_max)
            fill_w = bar_max * (acc_clamp - acc_min) / (acc_max - acc_min)
            fill_w = max(fill_w, 0.05)

            name_lbl = _b(exp["name"], 16, WHITE if i == best_idx else MUTED)
            name_lbl.move_to(np.array([x0 - 0.18, y, 0.0]))
            name_lbl.align_to(np.array([x0 - 0.18, y, 0.0]), RIGHT)
            name_labels.append(name_lbl)

            # Start bar collapsed to a sliver at the left edge; animate it growing right
            bar = Rectangle(
                width=0.001, height=bar_h,
                fill_color=color, fill_opacity=0.82, stroke_width=0,
            )
            bar.move_to(np.array([x0 + 0.0005, y, 0.0]))

            # Value label positioned at the final bar right edge
            val_lbl = _m(f"{exp['test_accuracy']:.2%}", 16, color)
            val_lbl.move_to(np.array([x0 + fill_w + 0.52, y, 0.0]))

            grow_anim = AnimationGroup(
                FadeIn(name_lbl, run_time=0.3),
                bar.animate.set_width(fill_w).shift(RIGHT * (fill_w - 0.001) / 2),
                run_time=0.55,
            )

            bars.append(bar)
            bar_anims.append(grow_anim)
            val_labels.append(val_lbl)

        # 98% reference line
        ref_x = x0 + bar_max * (target - acc_min) / (acc_max - acc_min)
        ref_line = Line(
            np.array([ref_x, y_top + 0.15, 0.0]),
            np.array([ref_x, y_top - frame_h + 0.55, 0.0]),
            color=WARNING, stroke_width=2.0, stroke_opacity=0.8,
        )
        ref_lbl = _b("98% target", 15, WARNING)
        ref_lbl.next_to(ref_line, UP, buff=0.10)

        all_g = VGroup(
            frame, *bars, *name_labels, *val_labels,
            ref_line, ref_lbl,
        )
        return {
            "all": all_g,
            "frame": frame,
            "bars": bars,
            "bar_anims": bar_anims,
            "val_labels": val_labels,
            "ref_line": ref_line,
            "ref_label": ref_lbl,
        }

    # -----------------------------------------------------------------------
    # Section 4: Results — best model
    # -----------------------------------------------------------------------
    def _section_4_results(self, header: VGroup, data: dict) -> None:
        sec = self._sec_title("4. Best model — per-class results", header)
        self.play(self._step(3), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.8)

        best = data["best"]
        recalls = best["per_class_recall"]  # one per digit 0-9

        # Per-class accuracy bar chart (vertical)
        digit_chart = self._build_vertical_bars(recalls)
        digit_chart.move_to(np.array([-2.2, -0.6, 0.0]))
        # Fade in digit labels/dashes first, then grow bars upward
        self.play(LaggedStart(
            *[FadeIn(VGroup(grp[2], grp[3]), shift=UP * 0.05) for grp in digit_chart[:-1]],
            lag_ratio=0.06, run_time=0.9,
        ))
        self.play(FadeIn(digit_chart[-1]))  # axis label
        self.wait(0.4)
        self.play(LaggedStart(
            *[AnimationGroup(
                grp[0].animate.set_height(grp[0]._target_height).move_to(
                    np.array([grp[0].get_center()[0], grp[0]._target_y, 0.0])
                ),
                FadeIn(grp[1]),
            ) for grp in digit_chart[:-1]],
            lag_ratio=0.10, run_time=2.0,
        ))
        self.wait(2.0)

        # Metric pills
        pills = VGroup(
            _pill("Test Accuracy", f"{best['test_accuracy']:.2%}", SUCCESS),
            _pill("F1 macro",      f"{best['test_f1_macro']:.4f}", ACCENT),
            _pill("Best epoch",    str(best["best_epoch"]),         WARNING),
        ).arrange(DOWN, buff=0.22)
        pills.move_to(np.array([4.6, 0.8, 0.0]))
        self.play(LaggedStart(
            *[FadeIn(p, shift=LEFT * 0.1) for p in pills],
            lag_ratio=0.25,
        ))
        self.wait(2.0)

        # Protocol note
        protocol = VGroup(
            _b("Training: digits.csv + more_digits.csv + augmentation", 16, MUTED),
            _b("Test set: digits_test.csv — never seen during training", 16, WHITE),
            _b(f"Strategy: {best['name']}", 17, SUCCESS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        proto_card = _card_group("Evaluation protocol", protocol, 6.0, 1.85)
        proto_card.move_to(np.array([4.2, -2.0, 0.0]))
        self.play(FadeIn(proto_card, shift=UP * 0.1))
        self.wait(3.5)

        self.play(FadeOut(digit_chart), FadeOut(pills), FadeOut(proto_card), FadeOut(sec))

    def _build_vertical_bars(self, recalls: list[float]) -> VGroup:
        bar_w   = 0.55
        gap     = 0.14
        bar_max = 3.4
        y_base  = -2.1
        acc_min = 0.94

        bars_group = []
        for i, rec in enumerate(recalls):
            x = i * (bar_w + gap) - (len(recalls) * (bar_w + gap)) / 2 + bar_w / 2
            fill_h = bar_max * (rec - acc_min) / (1.0 - acc_min)
            fill_h = max(fill_h, 0.05)
            color  = SUCCESS if rec >= 0.98 else (WARNING if rec >= 0.96 else POSITIVE)
            # Start bar collapsed to a sliver at the baseline; animate growing up
            bar = Rectangle(
                width=bar_w, height=0.001,
                fill_color=color, fill_opacity=0.80, stroke_width=0,
            )
            bar.move_to(np.array([x, y_base + 0.0005, 0.0]))

            val_lbl = _m(f"{rec:.1%}", 13, color)
            val_lbl.move_to(np.array([x, y_base + fill_h + 0.18, 0.0]))

            dig_lbl = _b(str(i), 18, WHITE)
            dig_lbl.move_to(np.array([x, y_base - 0.28, 0.0]))

            # 98% reference dash
            ref_y  = y_base + bar_max * (0.98 - acc_min) / (1.0 - acc_min)
            ref_dash = Line(
                np.array([x - bar_w / 2, ref_y, 0.0]),
                np.array([x + bar_w / 2, ref_y, 0.0]),
                color=WARNING, stroke_width=1.8, stroke_opacity=0.7,
            )
            # Store fill_h on bar for the grow animation caller
            bar._target_height = fill_h
            bar._target_y      = y_base + fill_h / 2
            bars_group.append(VGroup(bar, val_lbl, dig_lbl, ref_dash))

        # Axis label
        axis_lbl = _b("per-digit accuracy  (recall)   — dashed = 98%", 15, MUTED)
        axis_lbl.move_to(np.array([0.0, y_base - 0.62, 0.0]))
        return VGroup(*bars_group, axis_lbl)

    # -----------------------------------------------------------------------
    # Section 5: Takeaway
    # -----------------------------------------------------------------------
    def _section_5_takeaway(self, header: VGroup, data: dict) -> None:
        sec = self._sec_title("5. Takeaway", header)
        self.play(self._step(4), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.8)

        cfg  = data["config"]
        best = data["best"]
        arch = data["architecture"]
        arch_str  = " → ".join(str(n) for n in arch)
        n_params  = sum(arch[i] * arch[i+1] + arch[i+1] for i in range(len(arch) - 1))

        lines = VGroup(
            _b(f"Architecture:  {arch_str}  ({n_params:,} trainable parameters)", 20, WHITE),
            _b(f"Activations:   {cfg['hidden_activation']} (hidden)  ·  {cfg['output_activation']} (output)", 19, MUTED),
            _b(f"Optimizer:     {cfg['optimizer'].upper()}  lr={cfg['learning_rate']}  "
               f"batch={cfg['batch_size']}  λ={cfg['l2_lambda']}", 19, MUTED),
            _b("Data:          digits + more_digits  +  augmentation (shift · noise)", 19, MUTED),
            _b(f"Best strategy: {best['name']}  →  test accuracy {best['test_accuracy']:.2%}",
               20, SUCCESS),
            _b("Key insight:   more data + simple reweighting beats complex synthesis.",
               19, ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        card = _card_group("Summary", lines, 11.2, 3.4)
        card.move_to(np.array([0.0, -0.2, 0.0]))
        self.play(FadeIn(card, shift=UP * 0.15))
        self.wait(6.0)
