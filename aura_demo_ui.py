"""Gradio presentation interface for the existing AURA Colab runtime.

Upload this single file beside AURA_Colab_Demo.ipynb. No model fitting, downloads,
or ground-truth labels are used when a visitor submits a message.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from html import escape
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax

# Keep the saved CSV/score identifier compatible with existing exports.
MAIN_POLICY_ID = "signed_cost_gate"
MAIN_POLICY_LABEL = "Cost-aware gate"
POLICIES = {
    "confidence": "Confidence",
    "signed_gate": "Signed lexical gate",
    MAIN_POLICY_ID: MAIN_POLICY_LABEL,
}
COVERAGES = (50, 70, 80, 90, 100)
COLORS = ("#94a3b8", "#6366f1", "#0d9488")
EXAMPLES = [
    ["How do I activate my new card?"],
    ["I do not recognize a card payment on my account."],
    ["My phone was stolen and I cannot access my banking app."],
    ["My transfer says completed, but the recipient has not received it."],
    ["There is a payment problem with my card."],
    ["I topped up but the app declined it"],
    ["How long will the transfer take?"],
    ["My card didn't go through.  It doesn't make sense.  You told me that everything was ok."],
]
CSS = """
.gradio-container {max-width: 1240px !important; margin: auto; color-scheme: light;}
.aura-hero {background: linear-gradient(120deg, #edf8ff, #e8f8f1);
    border: 1px solid #cfe5e7; border-radius: 18px; padding: 25px 30px;
    color: #23475b; margin: 8px 0 18px;}
.aura-hero h1 {color: #23475b; margin: 5px 0; font-size: 30px;}
.aura-hero p {color: #4d6675; margin: 6px 0;}
.aura-eyebrow {color: #176d65; font-size: 12px; letter-spacing: 2px;}
.aura-summary {border: 1px solid #d8e7ed; padding: 20px; border-radius: 14px;
    background: #f3faff; color: #23475b; margin: 12px 0;}
.aura-summary h2 {color: #23475b; margin: 3px 0 10px;}
.aura-cards {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;}
.aura-card {border: 1px solid #d8e7ed; border-radius: 12px; padding: 17px;
    background: white; color: #23475b;}
.aura-card.main {border: 2px solid #60b5a8; background: #f0faf6;}
.aura-card h3 {color: #23475b; font-size: 15px; margin: 0 0 12px;}
.aura-card p {margin: 7px 0; font-size: 13px;}
.aura-decision {font-size: 23px; font-weight: 750;}
.aura-auto {color: #047857;} .aura-defer {color: #b45309;}
@media(max-width: 700px) {.aura-cards {grid-template-columns: 1fr;}}
"""


def light_theme() -> Any:
    """Use the same light surfaces even when the browser prefers dark mode."""
    import gradio as gr

    palette = {
        "body_background_fill": "#f7fafc",
        "body_text_color": "#243746",
        "body_text_color_subdued": "#526577",
        "background_fill_primary": "#ffffff",
        "background_fill_secondary": "#f0f7fa",
        "border_color_primary": "#d8e7ed",
        "color_accent_soft": "#e1f3f2",
        "link_text_color": "#176d65",
        "code_background_fill": "#edf4f8",
        "block_background_fill": "#ffffff",
        "block_border_color": "#d8e7ed",
        "block_label_background_fill": "#e9f5f4",
        "block_label_text_color": "#176d65",
        "block_title_text_color": "#23475b",
        "block_info_text_color": "#526577",
        "panel_background_fill": "#ffffff",
        "panel_border_color": "#d8e7ed",
        "accordion_text_color": "#243746",
        "input_background_fill": "#ffffff",
        "input_background_fill_focus": "#ffffff",
        "input_background_fill_hover": "#f5fbfb",
        "input_border_color": "#d8e7ed",
        "input_placeholder_color": "#647789",
        "table_text_color": "#243746",
        "table_border_color": "#d8e7ed",
        "table_even_background_fill": "#ffffff",
        "table_odd_background_fill": "#f3f8fb",
        "button_primary_background_fill": "#d9f1ec",
        "button_primary_background_fill_hover": "#c2e8df",
        "button_primary_border_color": "#9fd3c7",
        "button_primary_text_color": "#115e59",
        "button_primary_text_color_hover": "#115e59",
        "button_secondary_background_fill": "#f2f7fa",
        "button_secondary_background_fill_hover": "#e6f0f5",
        "button_secondary_text_color": "#23475b",
    }
    variables = {
        f"{key}{suffix}": value for key, value in palette.items() for suffix in ("", "_dark")
    }
    return gr.themes.Soft(primary_hue="teal", secondary_hue="sky", neutral_hue="slate").set(
        **variables
    )


def validation_cutoff(scores: Any, coverage_percent: float) -> float:
    """Return an illustrative order-statistic cutoff, never a test-derived one."""
    if coverage_percent not in COVERAGES:
        raise ValueError("Choose a supported automation target: 50, 70, 80, 90 or 100%.")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Validation scores must be a non-empty finite vector.")
    if coverage_percent == 100:
        return float("inf")  # No deferral, including scores above the validation maximum.
    count = int(
        (Decimal(str(coverage_percent)) * len(values) / 100).to_integral_value(rounding=ROUND_FLOOR)
    )
    return float(np.sort(values)[count - 1]) if count else float("-inf")


@dataclass
class NotebookDemo:
    """Inject notebook functions so the UI uses the existing, encoder-specific logic."""

    encoders: Mapping[str, Any]
    intents: list[str]
    workflows: list[dict[str, Any]]
    costs: np.ndarray
    witness: Any
    predict_logits: Callable[..., Any]
    lexical_features: Callable[..., Any]
    routing_scores: Callable[..., Any]
    routing_reports: Mapping[str, pd.DataFrame]
    classifier_reports: Mapping[str, pd.DataFrame]
    run_id: str

    @classmethod
    def from_namespace(cls, namespace: Mapping[str, Any]) -> "NotebookDemo":
        """Connect only to prepared runtime objects; fail clearly if cells are missing."""
        required = (
            "ENCODER_ROUTING",
            "INTENTS",
            "WORKFLOWS",
            "C_MODERATE",
            "witness",
            "predict_logits",
            "lexical_features",
            "routing_variant_scores",
            "encoder_validation_report",
            "validation_metrics",
        )
        missing = [key for key in required if key not in namespace]
        if missing:
            raise ValueError(
                "Run the notebook preparation cells through section 11B first. Missing: "
                + ", ".join(missing)
            )
        routing = {"Development validation": namespace["encoder_validation_report"]}
        classifiers = {"Development validation": namespace["validation_metrics"]}
        if isinstance(namespace.get("encoder_test_report"), pd.DataFrame):
            routing["Official test"] = namespace["encoder_test_report"]
        if isinstance(namespace.get("test_metrics"), pd.DataFrame):
            classifiers["Official test (already exposed)"] = namespace["test_metrics"]
        return cls(
            namespace["ENCODER_ROUTING"],
            list(namespace["INTENTS"]),
            sorted(namespace["WORKFLOWS"], key=lambda item: item["label"]),
            np.asarray(namespace["C_MODERATE"], dtype=np.float64),
            namespace["witness"],
            namespace["predict_logits"],
            namespace["lexical_features"],
            namespace["routing_variant_scores"],
            routing,
            classifiers,
            str(namespace.get("RUN_ID", "current notebook runtime")),
        )

    def analyze(self, message: str, encoder: str, coverage: float) -> dict[str, Any]:
        """Run one inference and compare three policies on the same predicted intent."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Enter a banking-support message before analyzing it.")
        if len(message) > 4000:
            raise ValueError("Keep the message under 4,000 characters for this demo.")
        if encoder not in self.encoders:
            raise ValueError("Choose one of the loaded encoders.")
        state = self.encoders[encoder]
        cutoffs = {key: validation_cutoff(state["val_scores"][key], coverage) for key in POLICIES}
        logits = np.asarray(
            self.predict_logits(
                [message.strip()],
                show_progress=False,
                trained_model=state["model"],
                trained_tokenizer=state["tokenizer"],
                config=state["config"],
            ),
            dtype=np.float64,
        )
        if logits.shape != (1, len(self.intents)) or not np.isfinite(logits).all():
            raise ValueError("The selected encoder returned invalid intent logits.")
        temperature = float(state["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("The selected encoder needs its own positive fitted temperature.")
        probabilities = softmax(logits / temperature, axis=1)
        features = self.lexical_features(
            logits, self.witness.decision_function([message.strip()]), self.witness.classes_
        )
        scores = self.routing_scores(probabilities, features, gates=state["gates"])
        proposed = int(probabilities[0].argmax())
        p = probabilities[0]
        decisions = []
        for key, label in POLICIES.items():
            score = float(scores[key][0])
            if not np.isfinite(score):
                raise ValueError(f"Non-finite score from {label}; no decision was issued.")
            decisions.append(
                {
                    "policy": key,
                    "name": label,
                    "score": score,
                    "cutoff": cutoffs[key],
                    "decision": "AUTO" if score <= cutoffs[key] else "DEFER",
                }
            )
        alternative = np.arange(len(p)) != proposed
        mass = float(p[alternative].sum())
        risk = float(p @ self.costs[:, proposed])
        if mass <= 0:
            raise ValueError("There is no numerical alternative-intent mass to explain.")
        contributions = (
            pd.DataFrame(
                {
                    "Possible true intent": self.intents,
                    "Probability (%)": p * 100,
                    "Cost of proposed route": self.costs[:, proposed],
                    "Contribution to expected cost": p * self.costs[:, proposed],
                }
            )
            .sort_values("Contribution to expected cost", ascending=False)
            .head(5)
            .round({"Probability (%)": 2, "Contribution to expected cost": 5})
        )
        return {
            "encoder": encoder,
            "coverage": coverage,
            "intent": self.intents[proposed],
            "confidence": float(p[proposed]),
            "workflow": self.workflows[proposed],
            "temperature": temperature,
            "probabilities": p,
            "decisions": decisions,
            "q": float(scores["signed_gate"][0]),
            "risk": risk,
            "mass": mass,
            "severity": risk / mass,
            "contributions": contributions,
        }

    def selected_results(
        self,
        partition: str,
        encoder: str,
        scenario: str,
        coverage: float,
    ) -> pd.DataFrame:
        """Read saved results without fitting, inference, or changing acceptance masks."""
        report = self.routing_reports[partition]
        selected = report[
            (report.encoder == encoder)
            & (report.evaluation_scenario == scenario)
            & np.isclose(report.target_coverage, coverage / 100)
            & report.policy.isin(POLICIES)
        ].copy()
        if len(selected) != len(POLICIES) or selected.policy.duplicated().any():
            raise ValueError("The current run has no complete result for this selection.")
        if selected.accepted_count.nunique() != 1:
            raise ValueError("Selected policy results do not have matched acceptance counts.")
        return selected.set_index("policy").loc[list(POLICIES)].reset_index()


def _cutoff_text(value: float) -> str:
    if value == float("inf"):
        return "all predictions accepted"
    if value == float("-inf"):
        return "no predictions accepted"
    return f"{value:.5f}"


def render_prediction(result: Mapping[str, Any]) -> tuple[str, str, str]:
    """Render model output safely; text is escaped before insertion into HTML."""
    proposed = escape(result["intent"].replace("_", " "))
    workflow = escape(result["workflow"]["required_action"].replace("_", " "))
    summary = (
        '<div class="aura-summary"><small>PROPOSED INTENT</small>'
        f"<h2>{proposed}</h2>Suggested workflow: <b>{workflow}</b><br>"
        f"Calibrated confidence: <b>{result['confidence']:.1%}</b>"
        f" &nbsp; | &nbsp; Encoder: {escape(result['encoder'])}</div>"
    )
    cards = []
    for item in result["decisions"]:
        main = " main" if item["policy"] == MAIN_POLICY_ID else ""
        decision = item["decision"]
        cards.append(
            f'<div class="aura-card{main}"><h3>{escape(item["name"])}</h3>'
            f'<div class="aura-decision aura-{decision.lower()}">{decision}</div>'
            f"<p>{'Automate proposed route' if decision == 'AUTO' else 'Send for human review'}"
            f"</p><p>Score: {item['score']:.5f}<br>"
            f"Cutoff: {_cutoff_text(item['cutoff'])}</p></div>"
        )
    aura = result["decisions"][-1]
    relation = "at or below" if aura["decision"] == "AUTO" else "above"
    explanation = (
        f"**AURA chose {aura['decision']}**: its score ({aura['score']:.5f}) is "
        f"{relation} the validation-derived cutoff ({_cutoff_text(aura['cutoff'])}).\n\n"
        f"1. The encoder proposes **{result['intent']}**. All three policies keep this intent.\n"
        f"2. Temperature scaling uses this encoder's fitted **T = "
        f"{result['temperature']:.4f}**.\n"
        f"3. The signed lexical gate estimates error probability **q = {result['q']:.5f}**.\n"
        f"4. Moderate-scenario expected routing cost **R = {result['risk']:.5f}**; "
        f"alternative-intent probability mass **a = {result['mass']:.5f}**.\n"
        f"5. AURA score **q × (R / a) = {aura['score']:.5f}**.\n\n"
        "The table shows possible alternative intents, not known true labels. The gate is "
        "an estimated error predictor; the consequence score is a relative research proxy. "
        "No actual error, customer harm, or financial loss can be measured "
        "from an unlabelled query."
    )
    return summary, '<div class="aura-cards">' + "".join(cards) + "</div>", explanation


def probability_figure(result: Mapping[str, Any], intents: list[str]) -> Any:
    """Top-five calibrated probabilities; does not renormalize away other intents."""
    from matplotlib.figure import Figure

    p = result["probabilities"]
    indices = np.argsort(-p, kind="stable")[:5]
    figure = Figure(figsize=(8, 3.1), layout="constrained")
    axis = figure.subplots()
    bars = axis.barh(
        [intents[i].replace("_", " ") for i in indices],
        p[indices] * 100,
        color=["#0d9488"] + ["#cbd5e1"] * (len(indices) - 1),
    )
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set(xlim=(0, 110), xlabel="Calibrated probability (%)")
    axis.invert_yaxis()
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def results_figures(
    backend: NotebookDemo,
    partition: str,
    encoder: str,
    scenario: str,
    coverage: float,
) -> tuple[Any, Any, str]:
    """Plot only this run, with the selected cost scenario used for evaluation."""
    from matplotlib.figure import Figure

    selected = backend.selected_results(partition, encoder, scenario, coverage)
    report = backend.routing_reports[partition]
    report = report[
        (report.encoder == encoder)
        & (report.evaluation_scenario == scenario)
        & report.policy.isin(POLICIES)
    ]
    figure = Figure(figsize=(9, 3.9), layout="constrained")
    axes = figure.subplots(1, 2)
    labels = ["Confidence", "Signed gate", "AURA"]
    for axis, column, title in zip(
        axes,
        ["accepted_total_cost", "accepted_errors"],
        ["Relative cost of automated routes", "Errors among automated routes"],
        strict=True,
    ):
        bars = axis.bar(labels, selected[column], color=COLORS, width=0.6)
        axis.bar_label(bars, fmt="%g", padding=4)
        axis.set(title=title, ylim=(0, max(float(selected[column].max()) * 1.22, 1)))
        axis.spines[["top", "right"]].set_visible(False)
    curve = Figure(figsize=(9, 3.4), layout="constrained")
    axis = curve.subplots()
    for (key, label), color in zip(POLICIES.items(), COLORS, strict=True):
        points = report[report.policy == key].sort_values("target_coverage")
        axis.plot(
            points.target_coverage * 100,
            points.accepted_total_cost,
            marker="o",
            label=label,
            color=color,
            linewidth=2,
        )
    axis.axvline(coverage, color="#94a3b8", linestyle=":", linewidth=1)
    axis.set(xlabel="Target automated coverage (%)", ylabel="Total relative research cost")
    axis.set_xticks(COVERAGES)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=9)
    indexed = selected.set_index("policy").to_dict(orient="index")
    aura, confidence, signed = (
        indexed[key] for key in ["signed_cost_gate", "confidence", "signed_gate"]
    )
    change = float(aura["accepted_total_cost"] - confidence["accepted_total_cost"])
    incremental = float(aura["accepted_total_cost"] - signed["accepted_total_cost"])
    note = (
        f"**{encoder} · {partition.removesuffix(' (already exposed)')} · "
        f"{scenario} evaluation scenario · "
        f"{coverage:g}% target**\n\n"
        f"Each policy automates **{int(aura['accepted_count']):,}** messages and defers "
        f"**{int(aura['deferred_count']):,}**. AURA minus confidence cost: **{change:+g}**; "
        f"AURA minus plain signed-gate cost: **{incremental:+g}**. Negative means lower cost.\n\n"
        "These are saved exact-budget batch results from this run. Live queries instead use "
        "validation-derived cutoffs; results here are not a live-query accuracy measurement. "
        "The high-protection selector changes the **evaluation costs**, not AURA's fixed "
        "moderate scoring rule. Observed differences do not establish statistical significance "
        "or uniform superiority."
    )
    return figure, curve, note


def classifier_figure(backend: NotebookDemo, partition: str) -> Any:
    from matplotlib.figure import Figure

    metrics = backend.classifier_reports[partition]
    figure = Figure(figsize=(9, 3.5), layout="constrained")
    axis = figure.subplots()
    bars = axis.barh(metrics.model, metrics.accuracy * 100, color="#0d9488")
    axis.bar_label(bars, fmt="%.2f%%", padding=4)
    axis.set(
        xlim=(0, 110),
        xlabel="Accuracy (%)",
        title=f"Classifiers: {partition.removesuffix(' (already exposed)')}",
    )
    axis.invert_yaxis()
    axis.spines[["top", "right"]].set_visible(False)
    return figure


def build_app(backend: NotebookDemo) -> Any:
    """Create the UI without starting a server or creating a public share link."""
    import gradio as gr

    encoder_names = list(backend.encoders)
    if not encoder_names:
        raise ValueError("No encoder is loaded. Complete notebook section 11B first.")
    default_encoder = "DistilBERT" if "DistilBERT" in encoder_names else encoder_names[0]
    partitions = list(backend.routing_reports)
    official_test = "Official test (already exposed)"
    default_partition = official_test if official_test in partitions else partitions[0]
    default_classifier_partition = (
        official_test
        if official_test in backend.classifier_reports
        else next(iter(backend.classifier_reports))
    )
    initial = results_figures(backend, default_partition, default_encoder, "moderate", 80)

    def analyze(message: str, encoder: str, coverage: float) -> tuple[Any, ...]:
        try:
            result = backend.analyze(message, encoder, coverage)
        except ValueError as error:
            raise gr.Error(str(error)) from error
        summary, cards, explanation = render_prediction(result)
        return (
            summary,
            cards,
            probability_figure(result, backend.intents),
            explanation,
            result["contributions"],
        )

    with gr.Blocks(title="AURA | Banking Intent Routing", analytics_enabled=False) as app:
        gr.HTML(
            '<div class="aura-hero"><div class="aura-eyebrow">CAPSTONE RESEARCH PROTOTYPE</div>'
            "<h1>AURA · Banking Intent Routing</h1>"
            "<p>Classify a customer message. Decide when to automate and when to defer.</p>"
            "<p>Puspita Chowdhury &amp; Farhat</p></div>"
        )
        with gr.Tab("Live routing"):
            with gr.Row():
                encoder = gr.Dropdown(encoder_names, value=default_encoder, label="Encoder")
                coverage = gr.Dropdown(
                    [(f"{c}%", c) for c in COVERAGES],
                    value=80,
                    label="Target automation",
                )
            message = gr.Textbox(
                label="Customer message",
                lines=3,
                max_lines=6,
                placeholder="Enter a banking-support question…",
            )
            run = gr.Button("Analyze query", variant="primary")
            gr.Examples(EXAMPLES, inputs=message, label="Try a message (outputs are not scripted)")
            summary = gr.HTML("<p>Submit a message to see its predicted intent and routing.</p>")
            cards = gr.HTML()
            gr.Markdown("### Top five possible intents")
            probabilities = gr.Plot(show_label=False)
            with gr.Accordion("Why did AURA make this decision?", open=False):
                explanation = gr.Markdown()
                gr.Markdown(
                    r"$$s(x)=q(x)\,\frac{\sum_i p_T(i\mid x)C[i,\hat y]}{1-p_T(\hat y\mid x)}$$"
                )
                contributions = gr.Dataframe(interactive=False, label="Expected-cost contributors")
                gr.Markdown(
                    "Moderate costs: correct route 0; incorrect 1; action mismatch 2; "
                    "omitted correction/compliance 3; omitted protection/unauthorized handling 5. "
                    "Use the largest applicable floor. These ratios are research assumptions."
                )
            run.click(
                analyze,
                inputs=[message, encoder, coverage],
                outputs=[summary, cards, probabilities, explanation, contributions],
                concurrency_limit=1,
                concurrency_id="encoder_inference",
            )
            for control in (message, encoder, coverage):
                control.change(
                    lambda: (
                        "<p>Inputs changed. Press Analyze query to update the decision.</p>",
                        "",
                        None,
                        "",
                        pd.DataFrame(),
                    ),
                    outputs=[summary, cards, probabilities, explanation, contributions],
                    queue=False,
                )
        with gr.Tab("Experimental results"):
            with gr.Row():
                partition = gr.Dropdown(
                    [(name.removesuffix(" (already exposed)"), name) for name in partitions],
                    value=default_partition,
                    label="Result partition",
                )
                result_encoder = gr.Dropdown(
                    encoder_names,
                    value=default_encoder,
                    label="Encoder",
                )
                scenario = gr.Dropdown(
                    ["flat", "moderate", "high_protection"],
                    value="moderate",
                    label="Evaluation cost scenario",
                )
                result_coverage = gr.Dropdown(
                    [(f"{c}%", c) for c in COVERAGES],
                    value=80,
                    label="Target automation",
                )
            result_note = gr.Markdown(initial[2])
            gr.Markdown("### Matched-coverage comparison")
            result_plot = gr.Plot(initial[0], show_label=False)
            gr.Markdown("### Cost across automation targets")
            curve = gr.Plot(initial[1], show_label=False)

            def update_results(part: str, enc: str, sc: str, cov: float) -> tuple[Any, ...]:
                figure, curve, note = results_figures(backend, part, enc, sc, cov)
                return note, figure, curve

            for control in (partition, result_encoder, scenario, result_coverage):
                control.change(
                    update_results,
                    inputs=[partition, result_encoder, scenario, result_coverage],
                    outputs=[result_note, result_plot, curve],
                )
            classifier_partition = gr.Dropdown(
                [
                    (name.removesuffix(" (already exposed)"), name)
                    for name in backend.classifier_reports
                ],
                value=default_classifier_partition,
                label="Classifier comparison partition",
            )
            gr.Markdown("### Classifier performance")
            classifiers = gr.Plot(
                classifier_figure(backend, default_classifier_partition),
                show_label=False,
            )
            classifier_partition.change(
                lambda part: classifier_figure(backend, part),
                inputs=classifier_partition,
                outputs=classifiers,
            )
    return app.queue(default_concurrency_limit=1, max_size=8)


def launch_notebook_demo(namespace: Mapping[str, Any], *, share: bool = True) -> Any:
    """Launch on the Colab GPU runtime; a share link lasts only while it is running."""
    app = build_app(NotebookDemo.from_namespace(namespace))
    app.launch(share=share, inline=True, css=CSS, theme=light_theme(), show_error=False)
    return app
