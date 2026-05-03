from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import numpy as np
from manim import (
    AnimationGroup,
    Arrow,
    Circle,
    Create,
    DecimalNumber,
    DOWN,
    Dot,
    FadeIn,
    FadeOut,
    GrowArrow,
    Indicate,
    LEFT,
    LaggedStart,
    Line,
    MoveAlongPath,
    ORIGIN,
    RIGHT,
    RoundedRectangle,
    Scene,
    Square,
    Text,
    UP,
    ValueTracker,
    VGroup,
    WHITE,
    always_redraw,
)


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "final_model_overview.json"

BG_COLOR = "#08111f"
CARD_FILL = "#0f172a"
CARD_STROKE = "#334155"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
POSITIVE = "#fb923c"
NEGATIVE = "#60a5fa"
SUCCESS = "#22c55e"
WARNING = "#facc15"
SECTION_LABELS = ["Architecture", "Weights", "Setup", "Metrics", "Takeaway"]
CARD_INSET = 0.42
FRAME_SAFE_WIDTH = 13.25
TITLE_FONT = "Noto Serif Display"
BODY_FONT = "Nimbus Sans"
MONO_FONT = "Nimbus Mono PS"

PRETTY_FEATURE_NAMES = {
    "timestamp": "timestamp",
    "amount_usd": "amount USD",
    "quantity_purchased": "quantity purchased",
    "session_duration_seconds": "session duration",
    "days_since_last_purchase": "days since last purchase",
    "account_age_days": "account age",
    "device_screen_resolution": "screen resolution",
    "time_since_last_login_s": "time since last login",
    "items_viewed_before_purchase": "items viewed before purchase",
}

SHORT_FEATURE_NAMES = {
    "timestamp": "timestamp",
    "amount_usd": "amount",
    "quantity_purchased": "quantity",
    "session_duration_seconds": "session",
    "days_since_last_purchase": "days since buy",
    "account_age_days": "account age",
    "device_screen_resolution": "screen res",
    "time_since_last_login_s": "login delta",
    "items_viewed_before_purchase": "items viewed",
}


def load_visualization_data() -> dict:
    override = os.environ.get("EX1_VIS_DATA")
    path = Path(override).expanduser() if override else DATA_PATH
    return json.loads(path.read_text())


def pretty_name(name: str) -> str:
    return PRETTY_FEATURE_NAMES.get(name, name.replace("_", " "))


def short_name(name: str) -> str:
    return SHORT_FEATURE_NAMES.get(name, name.replace("_", " "))


def metric_text(label: str, value: float, pct: bool = False) -> str:
    if pct:
        return f"{label}: {value:.2%}"
    return f"{label}: {value:.4f}"


def sigmoid_scalar(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))


def display_text(content: str, font_size: int, color: str = WHITE) -> Text:
    return Text(content, font_size=font_size, color=color, font=TITLE_FONT)


def body_text(content: str, font_size: int, color: str = WHITE) -> Text:
    return Text(content, font_size=font_size, color=color, font=BODY_FONT)


def numeric_text(content: str, font_size: int, color: str = WHITE) -> Text:
    return Text(content, font_size=font_size, color=color, font=MONO_FONT)


def wrapped_text(content: str, font_size: int, color: str = WHITE, max_chars: int | None = None) -> Text:
    if max_chars is not None:
        content = textwrap.fill(content, width=max_chars)
    return Text(content, font_size=font_size, color=color, font=BODY_FONT)


def make_card(width: float, height: float, fill_color: str = CARD_FILL) -> RoundedRectangle:
    return RoundedRectangle(
        corner_radius=0.18,
        width=width,
        height=height,
        fill_color=fill_color,
        fill_opacity=0.93,
        stroke_color=CARD_STROKE,
        stroke_width=1.6,
    )


def make_card_group(title: str, body: VGroup, width: float, height: float) -> VGroup:
    title_text = display_text(title, font_size=24, color=WHITE)
    available_width = width - 2 * CARD_INSET
    available_height = height - 2 * CARD_INSET
    if title_text.width > available_width:
        title_text.scale_to_fit_width(available_width)
    if body.width > available_width:
        body.scale_to_fit_width(available_width)
    content = VGroup(title_text, body).arrange(DOWN, aligned_edge=LEFT, buff=0.24)
    if content.width > available_width:
        content.scale_to_fit_width(available_width)
    if content.height > available_height:
        content.scale_to_fit_height(available_height)
    box = make_card(width, height)
    box.move_to(content.get_center())
    return VGroup(box, content)


class Exercise1FinalModelOverview(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG_COLOR
        data = load_visualization_data()

        header = self.build_header(data)
        section_title = display_text("1. Real architecture", font_size=28, color=ACCENT)
        section_title.next_to(header, DOWN, aligned_edge=LEFT, buff=0.34)

        intro = self.build_intro_card(data)
        intro.move_to(DOWN * 0.1)

        self.play(FadeIn(header, shift=DOWN * 0.2))
        self.play(self.set_stepper_state(0))
        self.play(FadeIn(section_title, shift=RIGHT * 0.15))
        self.play(FadeIn(intro, shift=UP * 0.2))
        self.wait(1.4)
        self.play(FadeOut(intro, shift=UP * 0.2))

        architecture = self.build_architecture_stage(data)
        self.play(LaggedStart(*[FadeIn(mob, shift=RIGHT * 0.1) for mob in architecture["inputs"]], lag_ratio=0.08))
        self.play(Create(architecture["connections"]))
        self.play(FadeIn(architecture["sum_card"], shift=UP * 0.15), GrowArrow(architecture["sum_to_sigmoid"]))
        self.play(FadeIn(architecture["sigmoid_group"], shift=UP * 0.15), GrowArrow(architecture["sigmoid_to_threshold"]))
        self.play(FadeIn(architecture["threshold_card"], shift=UP * 0.15), GrowArrow(architecture["threshold_to_output"]))
        self.play(FadeIn(architecture["output_card"], shift=UP * 0.15), FadeIn(architecture["notes"], shift=UP * 0.1))
        score_demo = self.build_score_demo(data["selection"]["threshold"])
        self.play(FadeIn(score_demo["group"], shift=UP * 0.12))
        self.play(
            MoveAlongPath(score_demo["pulse"], architecture["sum_to_sigmoid"]),
            score_demo["z_tracker"].animate.set_value(0.55),
            run_time=0.95,
        )
        self.play(
            MoveAlongPath(score_demo["pulse"], architecture["sigmoid_to_threshold"]),
            score_demo["z_tracker"].animate.set_value(1.52),
            run_time=1.05,
        )
        self.play(MoveAlongPath(score_demo["pulse"], architecture["threshold_to_output"]), run_time=0.85)
        self.play(
            Indicate(score_demo["decision_card"], color=SUCCESS, scale_factor=1.02),
            Indicate(architecture["output_card"], color=SUCCESS, scale_factor=1.02),
            run_time=0.8,
        )
        self.wait(0.9)

        self.play(
            FadeOut(architecture["group"], shift=LEFT * 0.2),
            FadeOut(score_demo["group"], shift=LEFT * 0.12),
            FadeOut(section_title, shift=UP * 0.1),
        )

        section_title = display_text("2. Learned weights", font_size=28, color=ACCENT)
        section_title.next_to(header, DOWN, aligned_edge=LEFT, buff=0.34)
        self.play(self.set_stepper_state(1))
        self.play(FadeIn(section_title, shift=RIGHT * 0.15))

        weight_stage = self.build_weight_stage(data)
        self.play(LaggedStart(*[FadeIn(mob, shift=RIGHT * 0.1) for mob in weight_stage["network_items"]], lag_ratio=0.06))
        self.play(FadeIn(weight_stage["legend"], shift=UP * 0.1), FadeIn(weight_stage["summary_cards"], shift=LEFT * 0.1))

        for idx in weight_stage["highlight_order"]:
            self.play(
                Indicate(weight_stage["label_map"][idx], color=weight_stage["line_map"][idx].get_color(), scale_factor=1.05),
                Indicate(weight_stage["line_map"][idx], color=weight_stage["line_map"][idx].get_color(), scale_factor=1.02),
                Indicate(weight_stage["row_map"][idx], color=weight_stage["line_map"][idx].get_color(), scale_factor=1.02),
                run_time=0.65,
            )
        self.wait(1.4)

        self.play(
            FadeOut(weight_stage["group"], shift=LEFT * 0.2),
            FadeOut(section_title, shift=UP * 0.1),
        )

        section_title = display_text("3. Training setup", font_size=28, color=ACCENT)
        section_title.next_to(header, DOWN, aligned_edge=LEFT, buff=0.34)
        self.play(self.set_stepper_state(2))
        self.play(FadeIn(section_title, shift=RIGHT * 0.15))

        setup_stage = self.build_setup_stage(data)
        self.play(FadeIn(setup_stage["training_card"], shift=RIGHT * 0.15))
        self.play(FadeIn(setup_stage["protocol_card"], shift=LEFT * 0.15))
        self.play(FadeIn(setup_stage["split_bar"], shift=UP * 0.1))
        self.wait(1.5)

        self.play(
            FadeOut(setup_stage["group"], shift=LEFT * 0.2),
            FadeOut(section_title, shift=UP * 0.1),
        )

        section_title = display_text("4. Final evaluation", font_size=28, color=ACCENT)
        section_title.next_to(header, DOWN, aligned_edge=LEFT, buff=0.34)
        self.play(self.set_stepper_state(3))
        self.play(FadeIn(section_title, shift=RIGHT * 0.15))

        metrics_stage = self.build_metrics_stage(data)
        self.play(FadeIn(metrics_stage["metrics_card"], shift=RIGHT * 0.15))
        self.play(FadeIn(metrics_stage["confusion_card"], shift=LEFT * 0.15), FadeIn(metrics_stage["threshold_note"], shift=UP * 0.1))
        self.wait(1.8)

        self.play(
            FadeOut(metrics_stage["group"], shift=LEFT * 0.2),
            FadeOut(section_title, shift=UP * 0.1),
        )

        section_title = display_text("5. Final takeaway", font_size=28, color=ACCENT)
        section_title.next_to(header, DOWN, aligned_edge=LEFT, buff=0.34)
        outro = self.build_outro_stage(data)
        self.play(self.set_stepper_state(4))
        self.play(FadeIn(section_title, shift=RIGHT * 0.15), FadeIn(outro, shift=UP * 0.15))
        self.wait(2.2)

    def build_header(self, data: dict) -> VGroup:
        title = display_text(data["title"], font_size=34, color=WHITE)
        subtitle = wrapped_text(
            "TinyModel distilled from BigModel for fraud detection",
            font_size=18,
            color=MUTED,
        )
        header_text = VGroup(title, subtitle).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        header_text.to_corner(UP + LEFT, buff=0.36)
        self.stepper = self.build_stepper()
        self.stepper.to_corner(UP + RIGHT, buff=0.46)
        divider = Line(LEFT * (FRAME_SAFE_WIDTH / 2), RIGHT * (FRAME_SAFE_WIDTH / 2), color=CARD_STROKE, stroke_width=1.2)
        divider.move_to(np.array([0.0, 2.82, 0.0]))
        header = VGroup(header_text, self.stepper, divider)
        return header

    def build_stepper(self) -> VGroup:
        items = []
        self.stepper_dots = []
        for label in SECTION_LABELS:
            dot = Circle(radius=0.08, color=CARD_STROKE, stroke_width=1.6)
            dot.set_fill(CARD_STROKE, opacity=0.35)
            items.append(dot)
            self.stepper_dots.append(dot)
        dots = VGroup(*items).arrange(RIGHT, buff=0.18)
        caption = body_text("story flow", font_size=13, color=MUTED)
        caption.next_to(dots, DOWN, buff=0.12)
        return VGroup(dots, caption)

    def set_stepper_state(self, active_idx: int) -> AnimationGroup:
        animations = []
        for idx, dot in enumerate(self.stepper_dots):
            if idx == active_idx:
                animations.append(dot.animate.set_stroke(ACCENT, width=2.0).set_fill(ACCENT, opacity=1.0))
            else:
                animations.append(dot.animate.set_stroke(CARD_STROKE, width=1.6).set_fill(CARD_STROKE, opacity=0.35))
        return AnimationGroup(*animations, lag_ratio=0.0, run_time=0.45)

    def build_intro_card(self, data: dict) -> VGroup:
        model = data["model"]
        metrics = data["metrics"]["test"]
        selection = data["selection"]
        body = VGroup(
            wrapped_text("Goal: mimic BigModel's fraud score with a tiny, cheap model.", font_size=22, color=WHITE, max_chars=58),
            wrapped_text(f"Model class: {model['type']}   |   activation: {model['activation']}", font_size=20, color=MUTED, max_chars=58),
            wrapped_text(f"Architecture: {len(model['feature_names'])} inputs -> 1 neuron", font_size=22, color=ACCENT, max_chars=58),
            wrapped_text(
                f"Final test snapshot: accuracy {metrics['accuracy']:.2%}   |   F1 {metrics['f1']:.4f}   |   ROC-AUC {selection['test_roc_auc']:.4f}",
                font_size=20,
                color=SUCCESS,
                max_chars=60,
            ),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        card = make_card_group("Exercise 1 overview", body, width=10.8, height=3.1)
        return card

    def build_architecture_stage(self, data: dict) -> dict:
        feature_names = data["model"]["feature_names"]
        threshold = data["selection"]["threshold"]

        inputs = []
        input_nodes = []
        y_positions = np.linspace(1.75, -2.35, len(feature_names))
        for feature_name, y in zip(feature_names, y_positions):
            node = Circle(radius=0.12, color=WHITE, stroke_width=2)
            node.move_to(np.array([-5.0, float(y), 0.0]))
            label = body_text(short_name(feature_name), font_size=16, color=WHITE)
            label.next_to(node, LEFT, buff=0.18)
            inputs.append(VGroup(node, label))
            input_nodes.append(node)

        inputs_group = VGroup(*inputs)
        input_header = body_text("standardized inputs (9)", font_size=18, color=MUTED)
        input_header.move_to(np.array([-3.15, 1.92, 0.0]))

        sum_card = self.make_pipeline_block(
            "weighted sum",
            "z = w · x + b",
            width=2.5,
            height=1.35,
            accent=ACCENT,
        )
        sum_card.move_to(np.array([-1.35, 0.0, 0.0]))

        sigmoid_node = Circle(radius=0.58, color=SUCCESS, stroke_width=3)
        sigmoid_node.move_to(np.array([0.95, 0.0, 0.0]))
        sigmoid_label = body_text("sigmoid", font_size=20, color=WHITE).move_to(sigmoid_node.get_center())
        sigmoid_note = body_text("score p in [0, 1]", font_size=16, color=MUTED)
        sigmoid_note.next_to(sigmoid_node, DOWN, buff=0.16)
        sigmoid_group = VGroup(sigmoid_node, sigmoid_label, sigmoid_note)

        threshold_card = self.make_pipeline_block(
            "threshold",
            f"tau = {threshold:.3f}",
            width=2.15,
            height=1.2,
            accent=WARNING,
        )
        threshold_card.move_to(np.array([3.35, 0.0, 0.0]))

        output_card = self.make_pipeline_block(
            "decision",
            "fraud / not fraud",
            width=2.45,
            height=1.2,
            accent=SUCCESS,
        )
        output_card.move_to(np.array([5.75, 0.0, 0.0]))

        connections = VGroup()
        for node in input_nodes:
            line = Line(node.get_right(), sum_card[0].get_left(), color=MUTED, stroke_width=1.2)
            connections.add(line)

        sum_to_sigmoid = Arrow(sum_card[0].get_right(), sigmoid_node.get_left(), buff=0.10, color=ACCENT, stroke_width=3)
        sigmoid_to_threshold = Arrow(sigmoid_node.get_right(), threshold_card[0].get_left(), buff=0.10, color=SUCCESS, stroke_width=3)
        threshold_to_output = Arrow(threshold_card[0].get_right(), output_card[0].get_left(), buff=0.10, color=WARNING, stroke_width=3)

        notes = VGroup(
            Text("architecture: 9 -> 1", font_size=22, color=ACCENT),
            wrapped_text("single non-linear neuron, no hidden layers", font_size=18, color=MUTED, max_chars=48),
            wrapped_text("this is intentional here: Exercise 1 asks for a simple perceptron", font_size=17, color=MUTED, max_chars=54),
            wrapped_text("the binary decision only happens after thresholding the sigmoid output", font_size=16, color=MUTED, max_chars=56),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        notes.move_to(np.array([0.5, -2.95, 0.0]))

        group = VGroup(
            input_header,
            inputs_group,
            connections,
            sum_card,
            sigmoid_group,
            threshold_card,
            output_card,
            sum_to_sigmoid,
            sigmoid_to_threshold,
            threshold_to_output,
            notes,
        )
        return {
            "group": group,
            "inputs": VGroup(input_header, inputs_group),
            "connections": connections,
            "sum_card": sum_card,
            "sigmoid_group": sigmoid_group,
            "threshold_card": threshold_card,
            "output_card": output_card,
            "sum_to_sigmoid": sum_to_sigmoid,
            "sigmoid_to_threshold": sigmoid_to_threshold,
            "threshold_to_output": threshold_to_output,
            "notes": notes,
        }

    def build_weight_stage(self, data: dict) -> dict:
        feature_names = data["model"]["feature_names"]
        weights = np.array(data["model"]["weights"], dtype=float)
        bias = float(data["model"]["bias"])
        max_abs_weight = max(float(np.max(np.abs(weights))), abs(bias), 1e-6)

        input_items = []
        label_map: dict[int, Text] = {}
        line_map: dict[int, Line] = {}
        y_positions = np.linspace(1.95, -2.15, len(feature_names))
        sum_anchor = np.array([-0.9, 0.0, 0.0])
        for idx, (feature_name, weight, y) in enumerate(zip(feature_names, weights, y_positions)):
            node = Circle(radius=0.11, color=WHITE, stroke_width=2)
            node.move_to(np.array([-5.2, float(y), 0.0]))
            label = body_text(short_name(feature_name), font_size=16, color=WHITE)
            label.next_to(node, LEFT, buff=0.18)
            edge_color = POSITIVE if weight >= 0 else NEGATIVE
            line = Line(
                node.get_right(),
                sum_anchor + UP * (float(y) * 0.10),
                color=edge_color,
                stroke_width=1.0 + 6.2 * abs(weight) / max_abs_weight,
            )
            line.set_opacity(0.9)
            input_items.append(VGroup(line, node, label))
            label_map[idx] = label
            line_map[idx] = line

        sum_card = self.make_pipeline_block(
            "weighted sum",
            "z = w · x + b",
            width=2.55,
            height=1.35,
            accent=ACCENT,
        )
        sum_card.move_to(np.array([-0.6, 0.0, 0.0]))
        sigmoid_node = Circle(radius=0.55, color=SUCCESS, stroke_width=3)
        sigmoid_node.move_to(np.array([1.75, 0.0, 0.0]))
        sigmoid_label = body_text("sigmoid", font_size=20, color=WHITE).move_to(sigmoid_node.get_center())
        sigmoid_group = VGroup(sigmoid_node, sigmoid_label)
        arrow = Arrow(sum_card[0].get_right(), sigmoid_node.get_left(), buff=0.10, color=ACCENT, stroke_width=3)

        bias_note = numeric_text(f"bias = {bias:+.3f}", font_size=18, color=MUTED)
        bias_note.move_to(np.array([-0.55, -1.38, 0.0]))
        architecture_note = wrapped_text(
            "thickness = |weight|   |   orange = positive   |   blue = negative",
            font_size=15,
            color=MUTED,
            max_chars=52,
        )
        architecture_note.move_to(np.array([-0.55, -3.08, 0.0]))
        network_group = VGroup(*input_items, sum_card, sigmoid_group, arrow, bias_note)

        positive_rows, positive_map = self.build_feature_rows(feature_names, weights, positive=True)
        negative_rows, negative_map = self.build_feature_rows(feature_names, weights, positive=False)

        positive_card = make_card_group("Top positive drivers", positive_rows, width=4.7, height=2.45)
        negative_card = make_card_group("Top negative drivers", negative_rows, width=4.7, height=2.45)
        positive_card.move_to(np.array([4.35, 1.10, 0.0]))
        negative_card.move_to(np.array([4.35, -1.50, 0.0]))
        summary_cards = VGroup(positive_card, negative_card)

        row_map = {**positive_map, **negative_map}
        positives = [idx for idx, weight in enumerate(weights) if weight >= 0]
        negatives = [idx for idx, weight in enumerate(weights) if weight < 0]
        highlight_order = sorted(positives, key=lambda idx: abs(weights[idx]), reverse=True)[:3]
        highlight_order += sorted(negatives, key=lambda idx: abs(weights[idx]), reverse=True)[:3]

        top_indices = set(highlight_order)
        for idx in range(len(feature_names)):
            if idx in top_indices:
                label_map[idx].set_opacity(1.0)
                line_map[idx].set_opacity(0.95)
            else:
                label_map[idx].set_opacity(0.58)
                line_map[idx].set_opacity(0.28)

        impact_note = VGroup(
            wrapped_text("positive edges push the fraud score upward", font_size=16, color=POSITIVE, max_chars=42),
            wrapped_text("negative edges pull the fraud score downward", font_size=16, color=NEGATIVE, max_chars=42),
            architecture_note,
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        legend = VGroup(impact_note).move_to(np.array([-0.15, -3.00, 0.0]))
        group = VGroup(network_group, summary_cards, legend)
        return {
            "group": group,
            "network_items": VGroup(*input_items, sum_card, sigmoid_group, arrow, bias_note),
            "legend": legend,
            "summary_cards": summary_cards,
            "label_map": label_map,
            "line_map": line_map,
            "row_map": row_map,
            "highlight_order": highlight_order,
        }

    def build_score_demo(self, threshold: float) -> dict:
        z_tracker = ValueTracker(-0.45)

        def probability() -> float:
            return sigmoid_scalar(z_tracker.get_value())

        z_value = always_redraw(
            lambda: numeric_text(f"{z_tracker.get_value():+.2f}", font_size=26, color=ACCENT)
        )
        p_value = always_redraw(
            lambda: numeric_text(f"{probability():.3f}", font_size=26, color=SUCCESS)
        )
        decision_value = always_redraw(
            lambda: body_text(
                "fraud" if probability() >= threshold else "not fraud",
                font_size=20,
                color=SUCCESS if probability() >= threshold else MUTED,
            )
        )

        z_col = VGroup(body_text("z", font_size=14, color=MUTED), z_value).arrange(DOWN, buff=0.05)
        p_col = VGroup(body_text("sigmoid(z)", font_size=14, color=MUTED), p_value).arrange(DOWN, buff=0.05)
        d_col = VGroup(body_text("decision", font_size=14, color=MUTED), decision_value).arrange(DOWN, buff=0.05)
        columns = VGroup(z_col, p_col, d_col).arrange(RIGHT, buff=0.42, aligned_edge=DOWN)
        body = VGroup(body_text("example score flow", font_size=14, color=MUTED), columns).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.08,
        )
        card = make_card_group("Live threshold demo", body, width=5.4, height=1.7)
        card.move_to(np.array([3.55, -2.62, 0.0]))

        pulse = Dot(radius=0.065, color=WARNING)
        pulse.move_to(np.array([-0.1, 0.0, 0.0]))
        return {
            "group": VGroup(card, pulse),
            "pulse": pulse,
            "z_tracker": z_tracker,
            "decision_card": card,
        }

    def build_setup_stage(self, data: dict) -> dict:
        hyper = data["hyperparameters"]
        splits = data["splits"]
        selection = data["selection"]

        training_lines = VGroup(
            body_text("single-layer non-linear perceptron", font_size=19, color=WHITE),
            body_text(f"activation: {data['model']['activation']}", font_size=18, color=MUTED),
            numeric_text(f"learning rate: {hyper['learning_rate']}", font_size=18, color=MUTED),
            numeric_text(f"epochs: {hyper['epochs']}", font_size=18, color=MUTED),
            numeric_text(f"batch size: {hyper['batch_size']}", font_size=18, color=MUTED),
            body_text(f"normalization: {hyper['normalization']}", font_size=18, color=MUTED),
            wrapped_text(f"teacher target: {hyper['teacher_target']}", font_size=18, color=MUTED, max_chars=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        training_card = make_card_group("Training configuration", training_lines, width=5.8, height=4.1)
        training_card.move_to(np.array([-3.4, -0.35, 0.0]))

        protocol_lines = VGroup(
            body_text(f"decision target: {hyper['decision_target']}", font_size=18, color=MUTED),
            wrapped_text("threshold is chosen on validation, not on test", font_size=18, color=WHITE, max_chars=31),
            wrapped_text(f"threshold source: {selection['threshold_source']}", font_size=18, color=MUTED, max_chars=29),
            numeric_text(f"threshold value: {selection['threshold']:.4f}", font_size=18, color=WARNING),
            wrapped_text("test split is used once for the final report", font_size=18, color=MUTED, max_chars=31),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        protocol_card = make_card_group("Evaluation protocol", protocol_lines, width=6.0, height=4.1)
        protocol_card.move_to(np.array([3.45, -0.35, 0.0]))

        split_bar = self.build_split_bar(splits)
        split_bar.move_to(np.array([0.0, -2.45, 0.0]))

        group = VGroup(training_card, protocol_card, split_bar)
        return {
            "group": group,
            "training_card": training_card,
            "protocol_card": protocol_card,
            "split_bar": split_bar,
        }

    def build_metrics_stage(self, data: dict) -> dict:
        metrics = data["metrics"]["test"]
        selection = data["selection"]

        big_numbers = VGroup(
            self.make_metric_pill("Accuracy", f"{metrics['accuracy']:.2%}", SUCCESS),
            self.make_metric_pill("F1", f"{metrics['f1']:.4f}", ACCENT),
        ).arrange(RIGHT, buff=0.24)

        metric_lines = VGroup(
            numeric_text(metric_text("Precision", metrics["precision"]), font_size=19, color=MUTED),
            numeric_text(metric_text("Recall", metrics["recall"]), font_size=19, color=MUTED),
            numeric_text(metric_text("ROC-AUC", selection["test_roc_auc"]), font_size=19, color=MUTED),
            numeric_text(metric_text("PR-AUC", selection["test_pr_auc"]), font_size=19, color=MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        metrics_body = VGroup(big_numbers, metric_lines).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        metrics_card = make_card_group("Final test performance", metrics_body, width=5.6, height=4.2)
        metrics_card.move_to(np.array([-3.45, -0.35, 0.0]))

        confusion_card = self.build_confusion_card(metrics["confusion_matrix"])
        confusion_card.move_to(np.array([3.55, -0.35, 0.0]))

        threshold_row = VGroup(
            numeric_text(f"tau = {selection['threshold']:.4f}", font_size=20, color=WARNING),
            self.build_threshold_gauge(selection['threshold']).scale(0.82),
        ).arrange(RIGHT, buff=0.36)
        threshold_note = make_card_group(
            "Decision boundary",
            VGroup(
                threshold_row,
                wrapped_text("selected on validation to keep the test report honest", font_size=16, color=MUTED, max_chars=44),
            ).arrange(DOWN, aligned_edge=LEFT, buff=0.10),
            width=8.2,
            height=1.55,
        )
        threshold_note.move_to(np.array([0.0, -2.80, 0.0]))

        group = VGroup(metrics_card, confusion_card, threshold_note)
        return {
            "group": group,
            "metrics_card": metrics_card,
            "confusion_card": confusion_card,
            "threshold_note": threshold_note,
        }

    def build_outro_stage(self, data: dict) -> VGroup:
        feature_count = len(data["model"]["feature_names"])
        metrics = data["metrics"]["test"]
        selection = data["selection"]
        lines = VGroup(
            wrapped_text(f"Architecture: {feature_count} inputs -> 1 sigmoid neuron -> threshold -> binary decision", font_size=22, color=WHITE, max_chars=70),
            wrapped_text("No hidden layers. That is not weird here: it is exactly what a simple perceptron is.", font_size=20, color=MUTED, max_chars=72),
            wrapped_text(
                f"Final test result: accuracy {metrics['accuracy']:.2%}   |   F1 {metrics['f1']:.4f}   |   ROC-AUC {selection['test_roc_auc']:.4f}",
                font_size=21,
                color=SUCCESS,
                max_chars=74,
            ),
            wrapped_text("Threshold selection is separated from test evaluation.", font_size=20, color=ACCENT, max_chars=72),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        outro = make_card_group("TinyModel summary", lines, width=11.0, height=3.15)
        outro.move_to(DOWN * 0.1)
        return outro

    def build_feature_rows(
        self,
        feature_names: list[str],
        weights: np.ndarray,
        positive: bool,
    ) -> tuple[VGroup, dict[int, VGroup]]:
        selected = [idx for idx, weight in enumerate(weights) if (weight >= 0) == positive]
        selected = sorted(selected, key=lambda idx: abs(weights[idx]), reverse=True)[:3]
        rows = []
        row_map: dict[int, VGroup] = {}
        for idx in selected:
            color = POSITIVE if positive else NEGATIVE
            label = body_text(pretty_name(feature_names[idx]), font_size=15, color=WHITE)
            bar_width = 0.35 + 1.10 * abs(weights[idx]) / max(abs(weights[selected]).max(), 1e-6)
            bar = RoundedRectangle(
                corner_radius=0.05,
                width=bar_width,
                height=0.12,
                stroke_width=0,
                fill_color=color,
                fill_opacity=0.86,
            )
            value = numeric_text(f"{weights[idx]:+.3f}", font_size=15, color=color)
            row = VGroup(label, bar, value).arrange(RIGHT, buff=0.16)
            rows.append(row)
            row_map[idx] = row
        return VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.18), row_map

    def build_split_bar(self, splits: dict) -> VGroup:
        labels = ["train", "validation", "test"]
        colors = [SUCCESS, WARNING, ACCENT]
        total = sum(int(splits[label]) for label in labels)
        width = 8.8
        segments = []
        texts = []
        current_left = -width / 2.0
        for label, color in zip(labels, colors):
            count = int(splits[label])
            segment_width = width * count / total
            box = RoundedRectangle(
                corner_radius=0.05,
                width=segment_width,
                height=0.46,
                fill_color=color,
                fill_opacity=0.85,
                stroke_width=0,
            )
            box.move_to(np.array([current_left + segment_width / 2.0, 0.0, 0.0]))
            text = numeric_text(f"{label}: {count}", font_size=16, color=WHITE)
            text.move_to(box.get_center())
            segments.append(box)
            texts.append(text)
            current_left += segment_width

        title = display_text("Data split used for the final model", font_size=21, color=WHITE)
        body = VGroup(VGroup(*segments), VGroup(*texts)).move_to(ORIGIN)
        foot = numeric_text("train 70%   |   validation 15%   |   test 15%", font_size=16, color=MUTED)
        group = VGroup(title, body, foot).arrange(DOWN, buff=0.18)
        card = make_card(9.4, 1.75)
        card.move_to(group.get_center())
        return VGroup(card, group)

    def build_threshold_gauge(self, threshold: float) -> VGroup:
        rail = Line(LEFT * 1.55, RIGHT * 1.55, color=CARD_STROKE, stroke_width=4)
        low = numeric_text("0", font_size=14, color=MUTED)
        high = numeric_text("1", font_size=14, color=MUTED)
        low.next_to(rail, LEFT, buff=0.14)
        high.next_to(rail, RIGHT, buff=0.14)

        marker_x = -1.55 + 3.10 * threshold
        marker = Line(UP * 0.22, DOWN * 0.22, color=WARNING, stroke_width=4)
        marker.move_to(np.array([marker_x, 0.0, 0.0]))
        marker_label = body_text("tau", font_size=15, color=WARNING)
        marker_label.next_to(marker, UP, buff=0.08)

        gauge = VGroup(rail, marker, marker_label, low, high)
        return gauge

    def build_confusion_card(self, matrix: list[list[int]]) -> VGroup:
        arr = np.array(matrix, dtype=float)
        max_value = max(float(arr.max()), 1.0)
        squares = []
        texts = []
        for row in range(2):
            for col in range(2):
                value = int(arr[row, col])
                opacity = 0.18 + 0.72 * value / max_value
                square = Square(side_length=0.92, stroke_color=CARD_STROKE, stroke_width=1.2)
                square.set_fill(ACCENT, opacity=opacity)
                square.move_to(np.array([col * 1.02, -row * 1.02, 0.0]))
                text = numeric_text(str(value), font_size=22, color=WHITE)
                text.move_to(square.get_center())
                squares.append(square)
                texts.append(text)

        matrix_group = VGroup(*squares, *texts)
        matrix_group.move_to(ORIGIN)
        col_labels = VGroup(
            body_text("not fraud", font_size=16, color=MUTED),
            body_text("fraud", font_size=16, color=MUTED),
        )
        col_labels.arrange(RIGHT, buff=0.52)
        col_labels.next_to(matrix_group, UP, buff=0.22)
        row_labels = VGroup(
            body_text("not fraud", font_size=16, color=MUTED),
            body_text("fraud", font_size=16, color=MUTED),
        )
        row_labels.arrange(DOWN, buff=0.60)
        row_labels.next_to(matrix_group, LEFT, buff=0.26)

        axis_pred = body_text("predicted", font_size=17, color=MUTED)
        axis_pred.next_to(col_labels, UP, buff=0.08)
        axis_actual = body_text("actual", font_size=17, color=MUTED)
        axis_actual.rotate(np.pi / 2)
        axis_actual.next_to(row_labels, LEFT, buff=0.10)

        body = VGroup(axis_pred, col_labels, axis_actual, row_labels, matrix_group)
        return make_card_group("Confusion matrix", body, width=5.8, height=4.2)

    def make_pipeline_block(self, title: str, subtitle: str, width: float, height: float, accent: str) -> VGroup:
        box = RoundedRectangle(
            corner_radius=0.16,
            width=width,
            height=height,
            fill_color=CARD_FILL,
            fill_opacity=0.96,
            stroke_color=accent,
            stroke_width=2.0,
        )
        title_text = display_text(title, font_size=20, color=WHITE)
        subtitle_text = body_text(subtitle, font_size=17, color=MUTED)
        content = VGroup(title_text, subtitle_text).arrange(DOWN, buff=0.10)
        content.move_to(box.get_center())
        return VGroup(box, content)

    def make_metric_pill(self, label: str, value: str, color: str) -> VGroup:
        box = RoundedRectangle(
            corner_radius=0.18,
            width=2.15,
            height=1.05,
            fill_color=CARD_FILL,
            fill_opacity=0.96,
            stroke_color=color,
            stroke_width=2.0,
        )
        title = body_text(label, font_size=18, color=MUTED)
        value_text = numeric_text(value, font_size=24, color=color)
        content = VGroup(title, value_text).arrange(DOWN, buff=0.04)
        content.move_to(box.get_center())
        return VGroup(box, content)
