"""
Validation — Single-layer perceptron: linear and non-linear regression.

Sections:
  1. Architecture  — single neuron diagram, two activation variants
  2. Linear live   — y = x: line sweeps from y = 0 to y = x via online GD
  3. All four      — y=2x, y=x+1, y=−1.5x+0.5 each fit in sequence
  4. Tanh live     — tanh(w·x+b) starts flat, inflates to tanh(x) over 200 epochs
  5. Takeaway      — what one neuron can and cannot express

Run:
  manim -qh validation_scene.py ValidationRegressionScene
"""
from __future__ import annotations

import numpy as np
from manim import (
    LEFT, RIGHT, UP, DOWN,
    WHITE,
    AnimationGroup, Arrow, Axes, Circle, Create, DashedLine,
    DecimalNumber, Dot, FadeIn, FadeOut, Flash, GrowArrow,
    LaggedStart, Line, RoundedRectangle,
    Scene, Text, Transform, ValueTracker, VGroup,
)

# ── Palette (matches existing scenes) ────────────────────────────────────────
BG_COLOR    = "#08111f"
CARD_FILL   = "#0f172a"
CARD_STROKE = "#334155"
MUTED       = "#94a3b8"
ACCENT      = "#38bdf8"
POSITIVE    = "#fb923c"
NEGATIVE    = "#60a5fa"
SUCCESS     = "#22c55e"
WARNING     = "#facc15"
ERROR_COLOR = "#f87171"

TITLE_FONT = "Noto Serif Display"
BODY_FONT  = "Nimbus Sans"
MONO_FONT  = "Nimbus Mono PS"

FRAME_W = 13.25
SECTION_LABELS = ["Arch", "Linear", "Targets", "Tanh", "Takeaway"]

# ── Training constants (mirror the validation scripts exactly) ────────────────
LINEAR_EPOCHS  = 50
LINEAR_LR      = 0.1
NONLIN_EPOCHS  = 200
NONLIN_LR      = 0.01
N_SAMPLES      = 50
TRAIN_RATIO    = 0.8

LINEAR_TARGETS = [
    {"slope": 1.0,  "bias": 0.0,  "label": "y = x",           "color": ACCENT},
    {"slope": 2.0,  "bias": 0.0,  "label": "y = 2x",          "color": SUCCESS},
    {"slope": 1.0,  "bias": 1.0,  "label": "y = x + 1",       "color": POSITIVE},
    {"slope": -1.5, "bias": 0.5,  "label": "y = −1.5x + 0.5", "color": NEGATIVE},
]


# ── Typography helpers ─────────────────────────────────────────────────────────
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
    avail_w, avail_h = w - 0.84, h - 0.84
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


# ── Training helpers ───────────────────────────────────────────────────────────
def _linear_train(slope: float, bias: float) -> list[tuple[float, float]]:
    """Online GD for a linear neuron; returns (w, b) after each epoch."""
    x = np.linspace(-1, 1, N_SAMPLES)
    y = slope * x + bias
    split = int(N_SAMPLES * TRAIN_RATIO)
    tx, ty = x[:split], y[:split]
    w, b = 0.0, 0.0
    traj = [(w, b)]
    for _ in range(LINEAR_EPOCHS):
        for xi, yi in zip(tx, ty):
            err = yi - (w * xi + b)
            w  += LINEAR_LR * err * xi
            b  += LINEAR_LR * err
        traj.append((w, b))
    return traj

def _tanh_train() -> list[tuple[float, float]]:
    """Online GD for a tanh neuron fitting tanh(x); returns (w, b) per epoch."""
    x = np.linspace(-2, 2, N_SAMPLES)
    y = np.tanh(x)
    split = int(N_SAMPLES * TRAIN_RATIO)
    tx, ty = x[:split], y[:split]
    w, b = 0.0, 0.0
    traj = [(w, b)]
    for _ in range(NONLIN_EPOCHS):
        for xi, yi in zip(tx, ty):
            z    = w * xi + b
            yp   = np.tanh(z)
            err  = yi - yp
            delta = err * (1.0 - np.tanh(z) ** 2)
            w += NONLIN_LR * delta * xi
            b += NONLIN_LR * delta
        traj.append((w, b))
    return traj

def _linear_mse(w: float, b: float, slope: float, bias: float) -> float:
    x = np.linspace(-1, 1, N_SAMPLES)
    y = slope * x + bias
    split = int(N_SAMPLES * TRAIN_RATIO)
    tx, ty = x[:split], y[:split]
    return float(np.mean((w * tx + b - ty) ** 2))

def _tanh_mse(w: float, b: float) -> float:
    x  = np.linspace(-2, 2, N_SAMPLES)
    y  = np.tanh(x)
    split = int(N_SAMPLES * TRAIN_RATIO)
    tx, ty = x[:split], y[:split]
    preds = np.tanh(w * tx + b)
    return float(np.mean((preds - ty) ** 2))


# ── Axes builders ──────────────────────────────────────────────────────────────
def _linear_axes() -> Axes:
    return Axes(
        x_range=[-1.1, 1.1, 0.5],
        y_range=[-2.5, 2.5, 1.0],
        x_length=8.0,
        y_length=5.2,
        axis_config={
            "color": CARD_STROKE, "stroke_width": 1.8,
            "include_tip": False, "include_numbers": False,
        },
    ).move_to(np.array([-1.2, -0.3, 0.0]))

def _tanh_axes() -> Axes:
    return Axes(
        x_range=[-2.2, 2.2, 1.0],
        y_range=[-1.3, 1.3, 0.5],
        x_length=8.0,
        y_length=5.2,
        axis_config={
            "color": CARD_STROKE, "stroke_width": 1.8,
            "include_tip": False, "include_numbers": False,
        },
    ).move_to(np.array([-1.2, -0.3, 0.0]))


# ── Scene ──────────────────────────────────────────────────────────────────────
class ValidationRegressionScene(Scene):

    def construct(self) -> None:
        self.camera.background_color = BG_COLOR
        header = self._build_header()

        self._section_1_architecture(header)
        self._section_2_linear_live(header)
        self._section_3_all_targets(header)
        self._section_4_tanh_live(header)
        self._section_5_takeaway(header)

    # ── Header & stepper ──────────────────────────────────────────────────────
    def _build_header(self) -> VGroup:
        title    = _t("Single-layer perceptron — regression validation", 32)
        subtitle = _b("linear activation  vs  tanh activation  ·  online gradient descent", 18, MUTED)
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
        VGroup(stepper, caption).to_corner(UP + RIGHT, buff=0.46)

        divider = Line(
            LEFT * (FRAME_W / 2), RIGHT * (FRAME_W / 2),
            color=CARD_STROKE, stroke_width=1.2,
        ).move_to(np.array([0.0, 2.82, 0.0]))

        header = VGroup(header_text, VGroup(stepper, caption), divider)
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

    # ── Section 1: Architecture ────────────────────────────────────────────────
    def _section_1_architecture(self, header: VGroup) -> None:
        sec = self._sec_title("1. Architecture: one neuron, two activations", header)
        self.play(self._step(0), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.6)

        # Shared neuron diagram parts
        def _neuron_diagram(label_act: str, color_act: str, y_offset: float) -> VGroup:
            x_node = Circle(radius=0.28, color=MUTED, stroke_width=2)
            x_node.set_fill(CARD_FILL, opacity=0.9)
            x_node.move_to(np.array([-4.2, y_offset, 0.0]))
            x_lbl = _m("x", 22, WHITE).move_to(x_node.get_center())

            neuron = Circle(radius=0.42, color=color_act, stroke_width=2.5)
            neuron.set_fill(CARD_FILL, opacity=0.9)
            neuron.move_to(np.array([-1.2, y_offset, 0.0]))
            n_lbl = _b(label_act, 16, color_act).move_to(neuron.get_center())

            y_node = Circle(radius=0.28, color=color_act, stroke_width=2)
            y_node.set_fill(CARD_FILL, opacity=0.9)
            y_node.move_to(np.array([2.0, y_offset, 0.0]))
            y_lbl = _m("ŷ", 22, color_act).move_to(y_node.get_center())

            arr1 = Arrow(x_node.get_right(), neuron.get_left(), buff=0.08,
                         color=MUTED, stroke_width=2.5)
            arr2 = Arrow(neuron.get_right(), y_node.get_left(), buff=0.08,
                         color=color_act, stroke_width=2.5)

            w_lbl = _m("w·x + b", 15, MUTED)
            w_lbl.next_to(arr1, UP, buff=0.08)

            return VGroup(x_node, x_lbl, neuron, n_lbl, y_node, y_lbl,
                          arr1, arr2, w_lbl)

        linear_diag = _neuron_diagram("f(z) = z", ACCENT, 1.0)
        tanh_diag   = _neuron_diagram("f(z) = tanh(z)", POSITIVE, -1.0)

        # Side labels
        lin_badge  = _b("Linear", 18, ACCENT).next_to(linear_diag, RIGHT, buff=0.5)
        tanh_badge = _b("Non-linear", 18, POSITIVE).next_to(tanh_diag, RIGHT, buff=0.5)

        divider_h = DashedLine(LEFT * 5.5, RIGHT * 3.5, color=CARD_STROKE,
                               stroke_width=1.0, stroke_opacity=0.5)

        self.play(FadeIn(linear_diag, shift=RIGHT * 0.12))
        self.play(FadeIn(lin_badge, shift=LEFT * 0.1))
        self.wait(0.6)
        self.play(FadeIn(divider_h))
        self.play(FadeIn(tanh_diag, shift=RIGHT * 0.12))
        self.play(FadeIn(tanh_badge, shift=LEFT * 0.1))
        self.wait(3.5)

        all_g = VGroup(linear_diag, tanh_diag, lin_badge, tanh_badge, divider_h)
        self.play(FadeOut(all_g, shift=LEFT * 0.12), FadeOut(sec))

    # ── Section 2: Linear live (y = x) ────────────────────────────────────────
    def _section_2_linear_live(self, header: VGroup) -> None:
        sec = self._sec_title("2. Linear activation — y = x  (live gradient descent)", header)
        self.play(self._step(1), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.6)

        traj = _linear_train(slope=1.0, bias=0.0)
        N    = len(traj) - 1  # 50

        ax    = _linear_axes()
        x_lbl = _b("x", 19, MUTED).next_to(ax, RIGHT, buff=0.2)
        y_lbl = _b("y", 19, MUTED).rotate(np.pi / 2).next_to(ax, LEFT, buff=0.22)
        self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=1.0)

        # Target line (dashed)
        target_line = DashedLine(
            ax.c2p(-1.0, -1.0), ax.c2p(1.0, 1.0),
            color=SUCCESS, stroke_width=2.2, stroke_opacity=0.55,
        )
        target_lbl = _b("y = x  (target)", 16, SUCCESS)
        target_lbl.next_to(target_line.get_end(), RIGHT, buff=0.12)
        self.play(Create(target_line), FadeIn(target_lbl), run_time=0.9)

        # Data points
        x_data = np.linspace(-1, 1, N_SAMPLES)
        dots = VGroup(*[
            Dot(ax.c2p(xi, xi), radius=0.07, color=ACCENT, fill_opacity=0.85)
            for xi in x_data
        ])
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.04, run_time=1.0))

        # Perceptron line — updater
        step_t = ValueTracker(0.0)

        def _get_params(t: float):
            i = int(np.clip(t, 0, N))
            return traj[i]

        def _line_pts(t: float):
            w, b = _get_params(t)
            return (
                ax.c2p(-1.0, float(np.clip(w * -1.0 + b, -2.4, 2.4))),
                ax.c2p( 1.0, float(np.clip(w *  1.0 + b, -2.4, 2.4))),
            )

        perc_line = Line(*_line_pts(0.0), color=ACCENT, stroke_width=3.0)
        perc_line.add_updater(lambda m: m.put_start_and_end_on(*_line_pts(step_t.get_value())))

        # Info card
        cx, cy = 4.85, 0.5
        card_bg    = _card(4.0, 3.0)
        card_bg.move_to(np.array([cx, cy, 0.0]))
        card_title = _b("Linear perceptron", 19, WHITE).move_to(np.array([cx, cy + 1.10, 0.0]))
        eq_lbl     = _b("ŷ = w · x + b", 17, MUTED).move_to(np.array([cx, cy + 0.68, 0.0]))

        w_label  = _b("w =", 16, MUTED).move_to(np.array([cx - 0.55, cy + 0.22, 0.0]))
        w_num    = DecimalNumber(0.0, num_decimal_places=4, font_size=26,
                                 color=ACCENT)
        w_num.move_to(np.array([cx + 0.45, cy + 0.22, 0.0]))
        w_num.add_updater(lambda m: m.set_value(_get_params(step_t.get_value())[0]))

        b_label  = _b("b =", 16, MUTED).move_to(np.array([cx - 0.55, cy - 0.18, 0.0]))
        b_num    = DecimalNumber(0.0, num_decimal_places=4, font_size=26,
                                 color=ACCENT)
        b_num.move_to(np.array([cx + 0.45, cy - 0.18, 0.0]))
        b_num.add_updater(lambda m: m.set_value(_get_params(step_t.get_value())[1]))

        mse_label = _b("MSE", 16, MUTED).move_to(np.array([cx - 0.55, cy - 0.62, 0.0]))
        mse_num   = DecimalNumber(
            _linear_mse(0.0, 0.0, 1.0, 0.0),
            num_decimal_places=6, font_size=22, color=WARNING,
        )
        mse_num.move_to(np.array([cx + 0.55, cy - 0.62, 0.0]))
        mse_num.add_updater(lambda m: m.set_value(
            _linear_mse(*_get_params(step_t.get_value()), 1.0, 0.0)
        ))

        ep_label = _b("epoch", 14, MUTED).move_to(np.array([cx, cy - 1.05, 0.0]))
        ep_num   = DecimalNumber(0, num_decimal_places=0, font_size=22,
                                 color=MUTED)
        ep_num.move_to(np.array([cx, cy - 1.28, 0.0]))
        ep_num.add_updater(lambda m: m.set_value(int(step_t.get_value())))

        info_grp = VGroup(card_bg, card_title, eq_lbl,
                          w_label, w_num, b_label, b_num,
                          mse_label, mse_num, ep_label, ep_num)

        self.add(perc_line)
        self.play(FadeIn(info_grp))
        self.wait(0.6)

        # Animate: converges fast — slow it down visually
        # Phase 1: first 15 epochs (where all learning happens) — slow
        self.play(step_t.animate.set_value(15), run_time=5.0, rate_func=lambda t: t ** 0.5)
        self.wait(0.8)
        # Phase 2: remaining epochs — fast (nothing changes visually)
        self.play(step_t.animate.set_value(N), run_time=1.5)
        self.wait(1.0)

        # Flash convergence
        self.play(Flash(perc_line.get_center(), color=SUCCESS,
                        flash_radius=0.4, line_length=0.15, run_time=0.7))
        conv_body = VGroup(
            _b("Converged at epoch ~10", 16, MUTED),
            _m("w = 1.0000   b = 0.0000", 18, SUCCESS),
            _m("MSE → 0", 18, SUCCESS),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        conv_card = _card_group("Result", conv_body, 4.0, 1.8)
        conv_card.move_to(np.array([4.85, -1.65, 0.0]))
        self.play(FadeIn(conv_card, shift=UP * 0.12))
        self.wait(3.0)

        perc_line.clear_updaters()
        w_num.clear_updaters(); b_num.clear_updaters()
        mse_num.clear_updaters(); ep_num.clear_updaters()

        all_g = VGroup(ax, x_lbl, y_lbl, target_line, target_lbl,
                       dots, perc_line, info_grp, conv_card)
        self.play(FadeOut(all_g, shift=LEFT * 0.12), FadeOut(sec))

    # ── Section 3: All four linear targets ────────────────────────────────────
    def _section_3_all_targets(self, header: VGroup) -> None:
        sec = self._sec_title("3. Linear activation — all four target functions", header)
        self.play(self._step(2), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.6)

        ax    = _linear_axes()
        x_lbl = _b("x", 19, MUTED).next_to(ax, RIGHT, buff=0.2)
        y_lbl = _b("y", 19, MUTED).rotate(np.pi / 2).next_to(ax, LEFT, buff=0.22)
        self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=0.9)

        x_data = np.linspace(-1, 1, N_SAMPLES)

        legend_rows: list[VGroup] = []
        fitted_lines = VGroup()

        for tgt in LINEAR_TARGETS:
            s, bv, lbl, col = tgt["slope"], tgt["bias"], tgt["label"], tgt["color"]
            traj = _linear_train(slope=s, bias=bv)
            w_f, b_f = traj[-1]

            # Target dashed line
            y0 = float(np.clip(s * -1.0 + bv, -2.4, 2.4))
            y1 = float(np.clip(s *  1.0 + bv, -2.4, 2.4))
            target_dl = DashedLine(
                ax.c2p(-1.0, y0), ax.c2p(1.0, y1),
                color=col, stroke_width=1.5, stroke_opacity=0.45,
            )

            # Learned line (from w=0 to final params — animate sweep)
            start_line = Line(ax.c2p(-1.0, 0.0), ax.c2p(1.0, 0.0),
                              color=col, stroke_width=2.6)
            final_y0 = float(np.clip(w_f * -1.0 + b_f, -2.4, 2.4))
            final_y1 = float(np.clip(w_f *  1.0 + b_f, -2.4, 2.4))
            final_line = Line(ax.c2p(-1.0, final_y0), ax.c2p(1.0, final_y1),
                              color=col, stroke_width=2.6)

            mse_f = _linear_mse(w_f, b_f, s, bv)

            self.play(Create(target_dl), run_time=0.35)
            self.play(Transform(start_line, final_line), run_time=1.0)
            fitted_lines.add(target_dl, start_line)
            self.wait(0.3)

            # Legend row
            swatch = Line(LEFT * 0.25, RIGHT * 0.25, color=col, stroke_width=3)
            row = VGroup(
                swatch,
                _b(lbl, 15, col),
                _m(f"w={w_f:.2f}  b={b_f:.2f}  mse={mse_f:.1e}", 14, MUTED),
            ).arrange(RIGHT, buff=0.14)
            legend_rows.append(row)

        legend_body = VGroup(*legend_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        legend_card = _card_group("Fitted functions", legend_body, 4.3, 2.8)
        legend_card.move_to(np.array([4.85, 0.2, 0.0]))
        self.play(FadeIn(legend_card, shift=LEFT * 0.12))
        self.wait(4.0)

        all_g = VGroup(ax, x_lbl, y_lbl, fitted_lines, legend_card)
        self.play(FadeOut(all_g, shift=LEFT * 0.12), FadeOut(sec))

    # ── Section 4: Tanh live ───────────────────────────────────────────────────
    def _section_4_tanh_live(self, header: VGroup) -> None:
        sec = self._sec_title("4. tanh activation — non-linear fit (live)", header)
        self.play(self._step(3), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.6)

        traj = _tanh_train()  # 201 entries: epoch 0 … 200
        x_data = np.linspace(-2, 2, N_SAMPLES)

        ax    = _tanh_axes()
        x_lbl = _b("x", 19, MUTED).next_to(ax, RIGHT, buff=0.2)
        y_lbl = _b("y", 19, MUTED).rotate(np.pi / 2).next_to(ax, LEFT, buff=0.22)
        self.play(Create(ax), FadeIn(x_lbl), FadeIn(y_lbl), run_time=1.0)

        # Target tanh curve (dashed, faint)
        target_curve = ax.plot(
            lambda x: np.tanh(x),
            x_range=[-2.1, 2.1],
            color=SUCCESS, stroke_width=2.0, stroke_opacity=0.45,
        )
        target_lbl = _b("y = tanh(x)  (target)", 16, SUCCESS)
        target_lbl.next_to(ax.c2p(1.5, 1.0), UP, buff=0.12)
        self.play(Create(target_curve), FadeIn(target_lbl), run_time=1.0)

        # Data points
        dots = VGroup(*[
            Dot(ax.c2p(xi, float(np.tanh(xi))), radius=0.07, color=ACCENT, fill_opacity=0.85)
            for xi in x_data
        ])
        self.play(LaggedStart(*[FadeIn(d, scale=1.4) for d in dots], lag_ratio=0.04, run_time=1.0))

        # Info card
        cx, cy = 4.85, 0.5
        card_bg    = _card(4.0, 3.0)
        card_bg.move_to(np.array([cx, cy, 0.0]))
        card_title = _b("tanh perceptron", 19, WHITE).move_to(np.array([cx, cy + 1.10, 0.0]))
        eq_lbl     = _b("ŷ = tanh(w · x + b)", 17, MUTED).move_to(np.array([cx, cy + 0.68, 0.0]))

        # We animate using keyframes via Transform (every 10 epochs = 21 keyframes)
        # Pre-compute current curve object
        w0, b0 = traj[0]
        curve = ax.plot(
            lambda x: np.tanh(w0 * x + b0),
            x_range=[-2.1, 2.1],
            color=POSITIVE, stroke_width=3.0,
        )
        self.add(curve)

        # Live w/b/mse labels — we update them manually between Transforms
        w_label  = _b("w =", 16, MUTED).move_to(np.array([cx - 0.55, cy + 0.22, 0.0]))
        w_num    = _m(f"{w0:.4f}", 26, POSITIVE).move_to(np.array([cx + 0.45, cy + 0.22, 0.0]))
        b_label  = _b("b =", 16, MUTED).move_to(np.array([cx - 0.55, cy - 0.18, 0.0]))
        b_num    = _m(f"{b0:.4f}", 26, POSITIVE).move_to(np.array([cx + 0.45, cy - 0.18, 0.0]))
        mse_label = _b("MSE", 16, MUTED).move_to(np.array([cx - 0.55, cy - 0.62, 0.0]))
        mse_num   = _m(f"{_tanh_mse(w0, b0):.6f}", 22, WARNING).move_to(np.array([cx + 0.55, cy - 0.62, 0.0]))
        ep_label  = _b("epoch", 14, MUTED).move_to(np.array([cx, cy - 1.05, 0.0]))
        ep_num    = _m("0", 22, MUTED).move_to(np.array([cx, cy - 1.28, 0.0]))

        info_grp = VGroup(card_bg, card_title, eq_lbl,
                          w_label, w_num, b_label, b_num,
                          mse_label, mse_num, ep_label, ep_num)
        self.play(FadeIn(info_grp))
        self.wait(0.5)

        # Choose keyframes: denser at start (fast learning), sparser later
        keyframes = list(range(0, 21, 2)) + list(range(20, 61, 5)) + list(range(60, 201, 20))
        keyframes = sorted(set(keyframes))

        for epoch in keyframes[1:]:
            w_e, b_e = traj[epoch]
            mse_e    = _tanh_mse(w_e, b_e)

            # Color shifts from POSITIVE → ACCENT → SUCCESS as it converges
            t = epoch / 200.0
            col = POSITIVE if t < 0.25 else (ACCENT if t < 0.75 else SUCCESS)

            new_curve = ax.plot(
                lambda x, w=w_e, b=b_e: np.tanh(w * x + b),
                x_range=[-2.1, 2.1],
                color=col, stroke_width=3.0,
            )
            new_w_num   = _m(f"{w_e:.4f}", 26, col).move_to(np.array([cx + 0.45, cy + 0.22, 0.0]))
            new_b_num   = _m(f"{b_e:.4f}", 26, col).move_to(np.array([cx + 0.45, cy - 0.18, 0.0]))
            new_mse_num = _m(f"{mse_e:.6f}", 22, WARNING).move_to(np.array([cx + 0.55, cy - 0.62, 0.0]))
            new_ep_num  = _m(str(epoch), 22, MUTED).move_to(np.array([cx, cy - 1.28, 0.0]))

            speed = 0.6 if epoch < 40 else (0.35 if epoch < 100 else 0.2)
            self.play(
                Transform(curve,   new_curve),
                Transform(w_num,   new_w_num),
                Transform(b_num,   new_b_num),
                Transform(mse_num, new_mse_num),
                Transform(ep_num,  new_ep_num),
                run_time=speed,
            )

        self.wait(1.0)
        self.play(Flash(ax.c2p(0, 0), color=SUCCESS,
                        flash_radius=0.4, line_length=0.15, run_time=0.7))

        final_body = VGroup(
            _b("Converged at epoch ~120", 16, MUTED),
            _m("w ≈ 1.0000   b ≈ 0.0000", 18, SUCCESS),
            _m("MSE → 0", 18, SUCCESS),
            _b("tanh(1·x + 0) = tanh(x)  ✓", 16, WHITE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        final_card = _card_group("Result", final_body, 4.0, 2.0)
        final_card.move_to(np.array([4.85, -1.65, 0.0]))
        self.play(FadeIn(final_card, shift=UP * 0.12))
        self.wait(4.0)

        all_g = VGroup(ax, x_lbl, y_lbl, target_curve, target_lbl,
                       dots, curve, info_grp, final_card)
        self.play(FadeOut(all_g, shift=LEFT * 0.12), FadeOut(sec))

    # ── Section 5: Takeaway ────────────────────────────────────────────────────
    def _section_5_takeaway(self, header: VGroup) -> None:
        sec = self._sec_title("5. Takeaway", header)
        self.play(self._step(4), FadeIn(sec, shift=RIGHT * 0.15))
        self.wait(0.8)

        # Left card — linear
        lin_body = VGroup(
            _m("ŷ = w·x + b", 22, ACCENT),
            _b("activation: identity", 17, MUTED),
            _b("fits only linear functions", 17, WHITE),
            _b("converges in ~10 epochs", 17, MUTED),
            _b("lr = 0.1", 16, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        lin_card = _card_group("Linear neuron", lin_body, 5.2, 3.0)
        lin_card.move_to(np.array([-3.2, -0.5, 0.0]))

        # Right card — tanh
        tanh_body = VGroup(
            _m("ŷ = tanh(w·x + b)", 22, POSITIVE),
            _b("activation: tanh", 17, MUTED),
            _b("fits smooth non-linear curves", 17, WHITE),
            _b("converges in ~120 epochs", 17, MUTED),
            _b("lr = 0.01", 16, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        tanh_card = _card_group("tanh neuron", tanh_body, 5.2, 3.0)
        tanh_card.move_to(np.array([3.2, -0.5, 0.0]))

        # Bottom insight
        insight = _b(
            "Stack neurons in layers  →  universal approximation  →  MLP",
            22, ACCENT,
        )
        insight.move_to(np.array([0.0, -2.55, 0.0]))

        self.play(FadeIn(lin_card, shift=RIGHT * 0.15))
        self.wait(0.4)
        self.play(FadeIn(tanh_card, shift=LEFT * 0.15))
        self.wait(1.5)
        self.play(FadeIn(insight, shift=UP * 0.12))
        self.wait(5.0)
