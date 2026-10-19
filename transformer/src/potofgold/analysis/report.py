"""Render saved encoder measurements as one self-contained HTML report.

Rendering reads measurements; it does not train models or recompute statistics.
Both the published experiment archive and new evaluation records are supported.
"""
from __future__ import annotations

import argparse
import base64
from html import escape
import io
import json
import math
import os
from pathlib import Path
import shlex
import tempfile

from ..encoder.config import TOKENIZER_ALIASES

NAMES = {"linear": "Linear", "patch": "Patch", "convolutional": "Convolutional", "indicators": "Indicators"}
TARGETS = {
    "volatility_difference": "Volatility difference", "volatility_scaled_return": "Volatility-scaled return",
    "upward_excursion": "Upward excursion", "downward_excursion": "Downward excursion",
}
METRICS = {"rankme": "Effective rank (RankMe)", "pr": "Participation ratio", "participation_ratio": "Participation ratio", "rank_ic": "RankIC", "r2": "R²"}
SPLITS = {"train": "Training", "val": "Validation", "test": "Test"}
COLORS = {"linear": "#44779d", "patch": "#a478aa", "convolutional": "#2b938a", "indicators": "#8a929b"}
for previous, current in TOKENIZER_ALIASES.items():
    NAMES[previous] = NAMES[current]
    COLORS[previous] = COLORS[current]


def number(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def statistic(value, digits=3):
    # Archives use "mean"; individual evaluations usually use "value". Accept
    # either shape and format its saved bounds without estimating an interval.
    if not isinstance(value, dict):
        return number(value, digits)
    result = number(value.get("mean", value.get("value", value.get("point"))), digits)
    bounds = value.get("ci")
    if bounds is not None and len(bounds) == 2:
        result += f" [{number(bounds[0], digits)}, {number(bounds[1], digits)}]"
    return result


def table(headers, rows):
    rows = list(rows)
    if not rows:
        return ""
    head = "".join(f"<th scope='col'>{escape(str(v))}</th>" for v in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f"<div class='table-scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def paragraph(text, cls=""):
    return f"<p class='{cls}'>{escape(str(text))}</p>"


def details(title, content):
    return f"<details><summary>{escape(title)}</summary>{content}</details>"


def _plot(rows, metrics, title, caption, group="label"):
    """Plot saved points and intervals, then embed the figure in HTML.

    Seaborn draws only the points; it does not aggregate rows or estimate bounds.
    """
    if not rows:
        return ""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "potofgold-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook", font_scale=.95)
    fig, axes = plt.subplots(1, len(metrics), figsize=(11.3, max(3.6, min(6.2, len(rows) * .28))))
    axes = np.atleast_1d(axes)
    labels = list(dict.fromkeys(row[group] for row in rows))
    tokenizers = list(dict.fromkeys(row["tokenizer"] for row in rows))
    offsets = np.linspace(-.24, .24, len(tokenizers)) if len(tokenizers) > 1 else [0]
    for ax, metric in zip(axes, metrics):
        points = []
        for row in rows:
            value = row.get(metric)
            if value is None:
                continue
            value = value if isinstance(value, dict) else {"mean": value}
            mean = value.get("mean", value.get("value", value.get("point")))
            if mean is None or not math.isfinite(mean):
                continue
            y = labels.index(row[group]) + (offsets[tokenizers.index(row["tokenizer"])] if group != "tokenizer" else 0)
            color = COLORS.get(row["tokenizer"], "#667085")
            bounds = value.get("ci")
            if bounds and all(v is not None and math.isfinite(v) for v in bounds):
                # Draw endpoints directly: a bootstrap percentile interval can
                # exclude the point estimate, so error-bar distances may be negative.
                ax.hlines(y, bounds[0], bounds[1], color=color, linewidth=2)
                ax.plot(bounds, [y, y], "|", color=color, markersize=7)
            points.append({"value": mean, "position": y, "Tokenizer": NAMES.get(row["tokenizer"], row["tokenizer"])})
        if points:
            palette = {NAMES.get(a, a): COLORS.get(a, "#667085") for a in tokenizers}
            sns.scatterplot(data=pd.DataFrame(points), x="value", y="position", hue="Tokenizer", palette=palette,
                            s=65, edgecolor="white", linewidth=.8, ax=ax, legend=(ax is axes[-1]))
        ax.set(yticks=range(len(labels)), yticklabels=[NAMES.get(v, v) for v in labels], ylabel="",
               xlabel=METRICS[metric], title=METRICS[metric])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        if metric in ("rank_ic", "r2"):
            ax.axvline(0, color="#9aa5b1", linewidth=.9, linestyle="--", zorder=0)
        sns.despine(ax=ax, left=True)
        legend = ax.get_legend()
        if legend is not None:
            if group == "tokenizer":
                legend.remove()
            else:
                legend.set_title(None)
                legend.set_loc("best")
                legend.set_frame_on(False)
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#233647")
    fig.tight_layout(rect=(0, 0, 1, .92))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # A data URL keeps the report portable as one file, with no separate images.
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"<figure><img alt='{escape(title)}' src='data:image/png;base64,{encoded}'><figcaption>{escape(caption)}</figcaption></figure>"


def _rankme_split_plot(rows, seed_rows):
    """Plot saved per-seed RankMe paths and means across the data splits."""
    def point(value):
        if isinstance(value, dict):
            return value.get("mean", value.get("value", value.get("point")))
        return value

    means, seeds = [], []
    for source, destination in ((rows, means), (seed_rows, seeds)):
        for row in source:
            value = point(row.get("rankme"))
            if row.get("split") in SPLITS and value is not None and math.isfinite(value):
                destination.append(dict(row, rankme=value, position=list(SPLITS).index(row["split"])))
    splits = [split for split in SPLITS if any(row["split"] == split for row in means)]
    if len(splits) < 2 or not seeds:
        return ""
    # A mean path requires one saved summary per tokenizer and split. Reports
    # without these summaries retain their existing per-run plots and tables.
    if len({(row["tokenizer"], row["split"]) for row in means}) != len(means):
        return ""

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "potofgold-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    import pandas as pd
    import seaborn as sns

    order = list(dict.fromkeys(row["tokenizer"] for row in means))
    reference_colors = {"Linear": "#0072b2", "Patch": "#e69f00", "Convolutional": "#009e73"}
    palette = {name: reference_colors.get(NAMES.get(name, name), COLORS.get(name, "#667085")) for name in order}
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.3,
                  rc={"grid.linewidth": 1.3, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    sns.lineplot(data=pd.DataFrame(seeds), x="position", y="rankme", hue="tokenizer",
                 hue_order=order, palette=palette, units="seed", estimator=None,
                 errorbar=None, alpha=.25, linewidth=1.3, legend=False, ax=ax)
    sns.lineplot(data=pd.DataFrame(means), x="position", y="rankme", hue="tokenizer",
                 hue_order=order, palette=palette, estimator=None, errorbar=None,
                 marker="o", markersize=10, linewidth=3.3, ax=ax)
    ax.set(xlabel="", ylabel="Effective rank (RankMe)",
           xticks=[list(SPLITS).index(split) for split in splits],
           xticklabels=[SPLITS[split] for split in splits])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [NAMES.get(name, name) for name in order],
              loc="center right", frameon=False, title=None, fontsize=13)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    title = "Effective rank (RankMe) across training, validation and test splits"
    caption = "Thin lines show individual training seeds; thick lines and markers show the saved means across seeds. Each split is measured separately using the trained encoders."
    return f"<figure><img alt='{escape(title)}' src='data:image/png;base64,{encoded}'><figcaption>{escape(caption)}</figcaption></figure>"


def _geometry_section(rows, *, caption, plot_split="test", seed_rows=()):
    if not rows:
        return ""
    available = list(dict.fromkeys(r.get("split", "unspecified") for r in rows))
    # Plot the requested split if present; the table below still shows every split.
    selected_split = plot_split if plot_split in available else available[-1]
    preferred = [r for r in rows if r.get("split", "unspecified") == selected_split]
    content = "<h2>Representation structure</h2>"
    content += paragraph("RankMe measures how evenly the singular values are spread across embedding directions. Participation ratio measures how broadly variance is distributed. Higher values alone do not establish useful prediction.")
    content += _rankme_split_plot(rows, seed_rows)
    content += _plot(preferred, ("rankme", "pr"), "Representation structure · " + SPLITS.get(selected_split, selected_split).lower() + " data", caption, group="tokenizer")
    content += table(["Split", "Tokenizer", "Windows", "Dimensions", "RankMe [95% CI]", "Participation ratio [95% CI]"], [
        [SPLITS.get(r.get("split"), r.get("split", "—")), NAMES.get(r["tokenizer"], r["tokenizer"]), r.get("n_windows", "—"), r.get("d_model", "—"), statistic(r.get("rankme"), 2), statistic(r.get("pr"), 2)] for r in rows])
    return content


def _prediction_section(rows, *, title, explanation, caption, plot=True):
    if not rows:
        return ""
    content = f"<h2>{escape(title)}</h2>" + paragraph(explanation)
    if plot:
        content += _plot(rows, ("rank_ic", "r2"), title, caption)
    content += table(["Target", "Tokenizer / inputs", "RankIC [95% CI]", "R² [95% CI]"], [
        [r["label"], r.get("inputs", NAMES.get(r["tokenizer"], r["tokenizer"])), statistic(r.get("rank_ic")), statistic(r.get("r2"))] for r in rows])
    return content


def _training_section(training):
    if not training:
        return ""
    return "<h2>Training records</h2>" + paragraph(
        "Best validation loss combines prediction and regularization terms. Tokenizers use different latent targets, so this is not a directly comparable financial prediction error. Paired-window counts refer to student–teacher training examples, not all exported embeddings."
    ) + table(["Tokenizer", "Seed", "Epochs", "Training pairs", "Validation pairs", "Best validation loss"], [
        [NAMES.get(r.get("tokenizer"), r.get("tokenizer", "—")), r.get("seed", "—"), r.get("epochs", "—"), r.get("train_pairs", "—"), r.get("val_pairs", "—"), number(r.get("best_val_loss"), 4)] for r in training])


def _archive(record):
    """Adapt the schema-1 experiment archive and its supplementary analyses."""
    summary = record.get("across_seed_results", {})
    tokenizers = summary.get("tokenizers", list(summary.get("per_tokenizer", {})))
    seeds = summary.get("seeds", [])
    output = paragraph("Original experiment · saved measurements", "eyebrow")
    output += _reproduction_notice()
    geometry, predictions, seed_rows, extra_predictions, seed_geometry = [], [], [], [], []
    for split in ("train", "val", "test"):
        for tokenizer in tokenizers:
            tokenizer_record = summary["per_tokenizer"][tokenizer]
            # Use results grouped by split when present. The archive also allows
            # test metrics directly under each tokenizer, but no such fallback for other splits.
            result = tokenizer_record.get("windows", {}).get(split, tokenizer_record if split == "test" else {})
            if not result:
                continue
            geometry.append(dict(tokenizer=tokenizer, split=split, n_windows=summary.get("n_eval_by_window", {}).get(split, "—"), d_model=summary.get("d_model"), rankme=result.get("rankme"), pr=result.get("pr")))
            # Each per_seed array follows summary["seeds"] in order.
            for i, seed in enumerate(seeds):
                seed_geometry.append(dict(tokenizer=tokenizer, split=split, seed=seed,
                                          rankme=result.get("rankme", {}).get("per_seed", [])[i]))
                seed_rows.append([SPLITS[split], NAMES.get(tokenizer, tokenizer), seed,
                                  number(result.get("rankme", {}).get("per_seed", [])[i], 2),
                                  number(result.get("pr", {}).get("per_seed", [])[i], 2)])
            for target, label in TARGETS.items():
                metrics = result.get("targets", {}).get(target)
                if metrics:
                    row = dict(tokenizer=tokenizer, label=label, rank_ic=metrics.get("rank_ic"), r2=metrics.get("r2"))
                    if split == "test":
                        predictions.append(row)
                    else:
                        extra_predictions.append([SPLITS[split], label, NAMES.get(tokenizer, tokenizer), statistic(row["rank_ic"]), statistic(row["r2"])])
    ci_caption = "Points: recorded means. Bars: 95% Student-t intervals across training seeds on the same dataset."
    output += _geometry_section(geometry, caption=ci_caption, seed_rows=seed_geometry)
    comparison = summary.get("five_percent_comparison", {}).get("effective_rank", {})
    if comparison:
        output += table(["Effective-rank comparison", "Relative difference [95% CI]"], [[
            "Convolutional versus " + NAMES.get(comparison.get("next_best_tokenizer"), "next-highest tokenizer"),
            statistic({"mean": comparison.get("improvement_pct_mean"), "ci": comparison.get("improvement_pct_ci")}, 2) + "%"]])
    output += details("Representation measurements for each seed", table(["Split", "Tokenizer", "Seed", "RankMe", "Participation ratio"], seed_rows))
    output += _prediction_section(predictions, title="Prediction of residual targets", explanation=
        "Ridge probes predict the part of each financial target left after subtracting a linear indicator model fitted on training data. Inputs to these probes are frozen embeddings. RankIC is the Spearman correlation of predicted and observed ranks. R² compares squared error with prediction by the evaluation target mean; a negative value means larger error. These are probe scores, not trading returns.", caption=ci_caption)
    output += paragraph("The intervals above describe variation across training seeds on the same test period; they do not resample test days. Intervals are target-wise, without a multiple-comparison adjustment.")
    output += details("Residual-target scores on training and validation data", table(["Split", "Target", "Tokenizer", "RankIC [95% CI]", "R² [95% CI]"], extra_predictions))
    output += _indicator_section(record.get("indicator_comparison", {}))
    output += _training_section(record.get("training", {}).get("records", []))
    output += _single_checkpoint_section(record)
    output += "<h2>Measurement methods</h2>" + paragraph(
        "The report preserves recorded point estimates and confidence bounds. It does not recalculate intervals from rounded tables. RankMe and participation ratio are distinct spectrum measures; the nominal embedding dimension is their ceiling. Sample size and concentration affect these measurements. The training archive records best validation summaries, not per-epoch curves."
    )
    return output


def _reproduction_notice():
    return "<aside><strong>What is required to reproduce the original experiment?</strong>" + paragraph(
        "Recomputing the original experimental measurements requires the original processed market dataset and the corresponding trained encoder checkpoints. The licensed original data cannot be redistributed and is not included. Rendering saved measurements recreates their report; synthetic data produces new measurements, not the original experimental results."
    ) + paragraph("All 15 original checkpoints are included: linear, patch and convolutional tokenizers at seeds 0, 42, 4395, 5040 and 6118. Recomputing the archived five-seed comparison also requires the original licensed corpus.") + "</aside>"


def _indicator_section(refit):
    """Render the archive's original-target analysis, separate from residual probes."""
    if not refit:
        return ""
    point, labels = refit["point"], refit["labels"]
    targets, metric_names = refit["targets"], refit["metrics"]
    # Arrays are [input model][target][metric], in the orders named above;
    # metric_ci adds a final [lower, upper] axis. Model zero is indicators only.
    scores, differences = [], []
    for ai, tokenizer in enumerate(labels):
        for ti, target in enumerate(targets):
            row = dict(tokenizer=tokenizer, label=TARGETS.get(target, target), inputs="Indicators only" if tokenizer == "indicators" else "Indicators + " + NAMES.get(tokenizer, tokenizer).lower())
            for mi, metric in enumerate(metric_names):
                row[metric] = {"mean": point[ai][ti][mi], "ci": refit.get("95", {}).get("metric_ci", [])[ai][ti][mi]}
            scores.append(row)
            if ai:
                delta = dict(tokenizer=tokenizer, label=TARGETS.get(target, target))
                # Subtract the saved points, but use the paired bootstrap's saved
                # delta interval; subtracting marginal interval bounds is invalid.
                # "ai_minus_0" selects the models; delta_ci is [target][metric][bound].
                for mi, metric in enumerate(metric_names):
                    delta[metric] = {"mean": point[ai][ti][mi] - point[0][ti][mi], "ci": refit["95"]["comparisons"][f"{ai}_minus_0"]["delta_ci"][ti][mi]}
                differences.append(delta)
    output = _prediction_section(scores, title="Prediction of original financial targets", explanation=
        "This separate analysis compares indicators alone with indicators plus frozen embeddings, predicting the original financial targets. These scores are not directly comparable with the residual-target scores above. Augmented scores average separately fitted encoder runs; they are not ensemble forecasts.", caption="", plot=False)
    output += _plot(differences, ("rank_ic", "r2"), "Additional predictive value beyond indicators", "Points: augmented-model score minus indicator-only score. Bars: paired whole-day and seed bootstrap 95% intervals.")
    output += table(["Target", "Added representation", "RankIC difference [95% CI]", "R² difference [95% CI]"], [
        [r["label"], NAMES.get(r["tokenizer"], r["tokenizer"]), statistic(r["rank_ic"]), statistic(r["r2"])] for r in differences])
    output += paragraph(f"Uncertainty: {refit.get('B', '—'):,} bootstrap resamples of the same {refit.get('n_days', '—')} test-day groups across models, keeping all windows within a day together. Seeds are resampled independently within each tokenizer. No consecutive-day blocks are used. Intervals are conditional on the training dataset and fitted models and are not adjusted across targets.")
    output += details("Indicator fitting and sensitivity intervals", paragraph(refit.get("method", "")) + paragraph(
        "The indicators are sampled at the last bar of each input window. Scaling and Ridge penalty selection use training data only. Alternative 90% and matched-seed 95% intervals remain in the original JSON."
    ) + table(["Input indicators"], [[name] for name in refit.get("indicator_names", [])]))
    return output


def _single_checkpoint_section(record):
    """Keep single-seed diagnostics apart from the archive's seed averages."""
    saved = record.get("checkpoint_evaluation", {})
    probe = record.get("checkpoint_probe", {})
    if not saved:
        return ""
    geometry, scores = [], []
    for tokenizer, result in saved.get("per_tokenizer", {}).items():
        geometry.append([NAMES.get(tokenizer, tokenizer), statistic({"mean": result.get("rankme"), "ci": result.get("rankme_ci")}, 2), statistic({"mean": result.get("pr"), "ci": result.get("pr_ci")}, 2)])
        for target, result_target in result.get("targets", {}).items():
            scores.append([NAMES.get(tokenizer, tokenizer), TARGETS.get(target, target)] + [statistic({"mean": result_target.get(m), "ci": result_target.get(m + "_ci")}) for m in ("rank_ic", "r2")])
    content = paragraph(f"Seed {saved.get('released_seed', '4395')}: {saved.get('n_eval', '—')} windows on {saved.get('n_days', '—')} test days. These intervals use {saved.get('config', {}).get('bootstrap_B', '—')} resamples of whole test-day groups, with fitted probes and encoder weights fixed. They do not include training-seed or probe-fitting uncertainty. Percentile intervals need not contain the original point estimate.")
    content += table(["Tokenizer", "RankMe [95% CI]", "Participation ratio [95% CI]"], geometry)
    content += table(["Tokenizer", "Residual target", "RankIC [95% CI]", "R² [95% CI]"], scores)
    supplementary, spectra = [], []
    for tokenizer, values in probe.get("tokenizers", {}).items():
        if tokenizer not in NAMES:
            continue
        geom = values.get("geometry") or {}
        spectra.append([NAMES[tokenizer], number(geom.get("covariance_decay_exponent")), number(geom.get("vicreg_hinge"))])
        # Skill and shuffle arrays share the positional order in probe["targets"].
        for i, result in enumerate(values.get("skill", [])):
            target = probe.get("targets", [])[i]
            supplementary.append([NAMES[tokenizer], TARGETS.get(target, target), number(result.get("ic")), number(result.get("sign")), number(values.get("shuffled_reference", {}).get("rank_ic", [None] * len(probe.get("targets", [])))[i])])
    content += table(["Tokenizer", "Covariance decay exponent", "Variance hinge"], spectra)
    content += paragraph("The covariance decay exponent is the negative fitted log–log slope of variance against direction rank. The variance hinge averages shortfalls below the per-dimension standard-deviation target; it is a training diagnostic, not a financial score.")
    content += paragraph("Supplementary probe diagnostics: Pearson IC, sign agreement, and the 95th percentile of shuffled-embedding RankIC scores. The shuffle result is a probe sanity check, not an alternative trading strategy.")
    content += table(["Tokenizer", "Residual target", "Pearson IC", "Sign agreement", "Shuffled RankIC (95th pct.)"], supplementary)
    return details("Single-seed measurements (seed 4395)", content)


def _evaluation(record):
    """Render schema-2 records organized by encoder run, split, and metric.

    Geometry-only runs are valid. Prediction, comparison, and training sections
    use whatever the evaluator saved; rendering does not fill in missing metrics.
    """
    if record.get("schema_version") != 2 or not isinstance(record.get("runs"), list):
        raise ValueError("Expected an archived experiment (schema 1) or encoder metrics (schema 2)")
    runs = record["runs"]
    if not runs:
        raise ValueError("The metrics record contains no encoder runs")
    source = record.get("source_kind", "new")
    evaluation = record.get("evaluation", {})
    eval_window = evaluation.get("eval_window", "test")
    source_name = {"synthetic": "Synthetic data", "original": "Original experiment data", "new": "User-supplied data",
                   "archived_measurements": "Original experiment"}.get(source, source)
    output = paragraph(source_name + " · saved measurements", "eyebrow")
    if source == "archived_measurements":
        output += paragraph("The evaluator recalculated representation means, confidence intervals and the 5% effective-rank milestone from archived per-seed measurements. This report displays those recalculated results.", "lead")
    else:
        output += paragraph("This report renders metrics already computed from exported embeddings. Rendering does not train an encoder or run an evaluation.", "lead")
    if source == "synthetic":
        output += "<aside>" + paragraph("These are measurements on generated synthetic data. They demonstrate the workflow and do not reproduce the original market-data results or establish trading performance.") + "</aside>"
    elif source == "original":
        output += _reproduction_notice()
    source_label = "Measurement archive" if source == "archived_measurements" else "Embedding file"
    output += table(["Tokenizer", "Seed", "Dimensions", source_label], [
        [NAMES.get(r.get("tokenizer"), r.get("tokenizer", "—")), r.get("seed", "—"), r.get("d_model", "—"),
         Path(record.get("source", {}).get("path", "—") if source == "archived_measurements" else r.get("embedding_file", "—")).name] for r in runs])
    split_dates = record.get("corpus", {}).get("split_dates", {})
    split_rows = []
    for split in ("train", "val", "test"):
        counts = [r.get("windows", {}).get(split, {}) for r in runs]
        days = sorted({str(c["n_days"]) for c in counts if "n_days" in c})
        windows = sorted({str(c["n_windows"]) for c in counts if "n_windows" in c})
        if counts and any(counts):
            split_rows.append([SPLITS[split], split_dates.get(split + "_start", "—"), split_dates.get(split + "_end", "—"), ", ".join(days) or "—", ", ".join(windows) or "—"])
    output += table(["Split", "Configured start", "Configured end", "Evaluated days", "Evaluated windows"], split_rows)
    output += paragraph("Split dates are configured boundaries. Counts reflect available observations after evaluation filtering; different counts across runs are listed separately.")
    if source == "archived_measurements":
        output += paragraph(f"Evaluation split: {SPLITS.get(eval_window, eval_window).lower()}. Replication unit: encoder seed on the recorded evaluation dates.")
    else:
        output += paragraph(f"Evaluation split: {SPLITS.get(eval_window, eval_window).lower()}. Embargo: {evaluation.get('embargo_days', '—')} trading days. Geometry and residual-probe bootstrap: {evaluation.get('bootstrap_repeats', '—')} resamples; random seed {evaluation.get('bootstrap_seed', '—')}.")
    ci_method = evaluation.get("ci_method", "Confidence intervals, where present, are supplied by the saved evaluation record.")
    if isinstance(ci_method, dict):
        ci_method = "; ".join(f"{k}: {v}" for k, v in ci_method.items())
    output += paragraph("Uncertainty: " + str(ci_method))
    aggregates = record.get("seed_aggregates", {})
    geometry, predictions, individual, individual_predictions, seed_geometry = [], [], [], [], []
    # Prefer saved across-seed summaries when any tokenizer has multiple seeds.
    # Otherwise show per-run metrics and their saved day-bootstrap intervals.
    # In a mixed aggregate report, single-seed groups have no across-seed interval.
    multiple_seeds = any(group.get("n_seeds", 0) > 1 for group in aggregates.values())
    use_aggregates = bool(aggregates) and multiple_seeds
    for run in runs:
        tokenizer, seed = run.get("tokenizer", "unknown"), run.get("seed", "—")
        for split, result in run.get("windows", {}).items():
            geom = result.get("geometry", {})
            seed_geometry.append(dict(tokenizer=tokenizer, split=split, seed=seed, rankme=geom.get("rankme")))
            individual.append([NAMES.get(tokenizer, tokenizer), seed, SPLITS.get(split, split), result.get("n_days", "—"), result.get("n_windows", "—"), statistic(geom.get("rankme"), 2), statistic(geom.get("participation_ratio", geom.get("pr")), 2)])
            if not use_aggregates:
                geometry.append(dict(tokenizer=tokenizer, split=split, n_windows=result.get("n_windows", "—"), d_model=run.get("d_model", "—"), rankme=geom.get("rankme"), pr=geom.get("participation_ratio", geom.get("pr"))))
            for target, value in result.get("residual_predictions", {}).items():
                individual_predictions.append([NAMES.get(tokenizer, tokenizer), seed, SPLITS.get(split, split), TARGETS.get(target, target), statistic(value.get("rank_ic")), statistic(value.get("r2"))])
                if not use_aggregates and split == eval_window:
                    predictions.append(dict(tokenizer=tokenizer, label=TARGETS.get(target, target), rank_ic=value.get("rank_ic"), r2=value.get("r2")))
    if use_aggregates:
        for tokenizer, aggregate in aggregates.items():
            # Aggregates contain statistics; sample counts and dimensions come
            # from a representative run. The per-run table retains every count.
            example = next((r for r in runs if r.get("tokenizer") == tokenizer), {})
            for split, result in aggregate.get("windows", {}).items():
                geom = result.get("geometry", {})
                count = example.get("windows", {}).get(split, {}).get("n_windows", "—")
                geometry.append(dict(tokenizer=tokenizer, split=split, n_windows=count, d_model=example.get("d_model", "—"), rankme=geom.get("rankme"), pr=geom.get("participation_ratio", geom.get("pr"))))
                if split == eval_window:
                    for target, value in result.get("residual_predictions", {}).items():
                        predictions.append(dict(tokenizer=tokenizer, label=TARGETS.get(target, target), rank_ic=value.get("rank_ic"), r2=value.get("r2")))
    caption = ("Points: means across training seeds. Bars: 95% Student-t intervals across seeds; no interval for a single seed." if use_aggregates else
               "Points: computed metrics. Bars, where present: saved 95% intervals from resampling whole evaluation days.")
    output += _geometry_section(geometry, caption=caption, plot_split=eval_window, seed_rows=seed_geometry)
    if record.get("seed_comparisons"):
        comparison_rows = []
        show_threshold = any("threshold_pct" in item for item in record["seed_comparisons"])
        for item in record["seed_comparisons"]:
            if item.get("metric") != "relative_rankme_pct":
                continue
            others = ", ".join(NAMES.get(tokenizer, tokenizer) + " (" + str(count) + ")" for tokenizer, count in item.get("next_best_counts", {}).items())
            row = [NAMES.get(item.get("tokenizer"), item.get("tokenizer", "—")), SPLITS.get(item.get("window"), item.get("window", "—")), ", ".join(map(str, item.get("seeds", []))), others or "Best other selected tokenizer", statistic(item, 2) + "%"]
            if show_threshold:
                passed = item.get("lower_ci_exceeds_threshold")
                row.append("—" if passed is None else f'{"Yes" if passed else "No"} ({number(item["threshold_pct"], 0)}%)')
            comparison_rows.append(row)
        output += paragraph("Relative effective-rank comparisons use the best other selected tokenizer within each matched seed. The percentage differences are formed before averaging across seeds. Brackets show a 95% Student-t interval when more than one matched seed is available.")
        headers = ["Tokenizer", "Split", "Matched seeds", "Comparator (seed count)", "RankMe difference [95% CI]"]
        if show_threshold:
            headers.append("Lower CI bound exceeds threshold?")
        output += table(headers, comparison_rows)
    output += details("Measurements and sample counts for each run", table(["Tokenizer", "Seed", "Split", "Days", "Windows", "RankMe [95% CI]", "Participation ratio [95% CI]"], individual))
    output += _prediction_section(predictions, title="Prediction of residual targets", explanation=
        "Frozen embeddings are used to fit Ridge probes to residual financial targets. The target residuals are formed by subtracting a linear indicator model fitted on training data. RankIC measures agreement between predicted and observed rankings. R² measures squared-error performance relative to predicting the evaluation target mean. These scores do not measure strategy profitability.", caption=caption)
    if individual_predictions:
        output += details("Residual-target scores for each run and split", table(["Tokenizer", "Seed", "Split", "Target", "RankIC [95% CI]", "R² [95% CI]"], individual_predictions))
    elif source == "archived_measurements":
        output += paragraph("This recalculation covers representation geometry and the effective-rank comparison.")
    else:
        output += paragraph("Prediction metrics were not requested. This record evaluates the structure of the exported embeddings only.")
    # Original targets use a separate indicator comparison, not the residual
    # targets summarized above; keep their scores in their own section.
    output += _new_indicator_section(record.get("indicator_comparison"))
    paired_rows = []
    for comparison in record.get("paired_deltas", []):
        left, right = comparison.get("left", "—"), comparison.get("right", "—")
        label = f"{NAMES.get(left, left)} minus {NAMES.get(right, right)}"
        for metric, value in comparison.get("geometry", {}).items():
            paired_rows.append([label, comparison.get("window", eval_window), METRICS.get(metric, metric), statistic(value)])
        for target, values in comparison.get("residual_predictions", {}).items():
            for metric, value in values.items():
                paired_rows.append([label, comparison.get("window", eval_window), TARGETS.get(target, target) + " / " + METRICS.get(metric, metric), statistic(value)])
    if paired_rows:
        output += details("Paired comparisons", paragraph("Differences use the same sampled evaluation days for both runs. Positive values mean the first run's metric is higher. Intervals are conditional on these fitted models and probes.") + table(["Comparison", "Split", "Metric", "Difference [95% CI]"], paired_rows))
    training = [dict(r.get("training", {}), tokenizer=r.get("tokenizer"), seed=r.get("seed")) for r in runs if r.get("training")]
    output += _training_section(training)
    if evaluation.get("residual_probe_method") and predictions:
        output += details("Residual-probe fitting method", paragraph(evaluation["residual_probe_method"]))
    notes = evaluation.get("notes", [])
    if isinstance(notes, str):
        notes = [notes]
    if notes:
        output += "<h2>Evaluation notes</h2>" + "".join(paragraph(note) for note in notes)
    return output


def _new_indicator_section(comparison):
    # Schema 2 names models, targets, and metrics instead of indexing archive arrays.
    if not comparison:
        return ""
    rows = []
    for model, values in comparison.get("models", {}).items():
        for target, metrics in values.get("targets", {}).items():
            inputs = "Indicators only" if model == "indicators" else "Indicators + " + NAMES.get(model, model).lower()
            rows.append(dict(tokenizer=model, label=TARGETS.get(target, target), inputs=inputs,
                             rank_ic=metrics.get("rank_ic"), r2=metrics.get("r2")))
    output = _prediction_section(rows, title="Prediction of original financial targets", explanation=
        "This separate analysis uses indicators alone or indicators plus embeddings to predict the original targets. Its scores have different targets from the residual probes above and must be read separately.", caption="", plot=False)
    deltas = []
    for pair in comparison.get("paired_deltas", []):
        for target, metrics in pair.get("targets", {}).items():
            left, right = pair.get("left", "—"), pair.get("right", "—")
            left_label = "Indicators only" if left == "indicators" else "Indicators + " + NAMES.get(left, left).lower()
            right_label = "indicators only" if right == "indicators" else "indicators + " + NAMES.get(right, right).lower()
            deltas.append([left_label + " minus " + right_label, TARGETS.get(target, target), statistic(metrics.get("rank_ic")), statistic(metrics.get("r2"))])
    output += table(["Comparison", "Target", "RankIC difference [95% CI]", "R² difference [95% CI]"], deltas)
    output += paragraph(comparison.get("method", ""))
    if "bootstrap_repeats" in comparison:
        output += paragraph(f"Original-target bootstrap: {comparison['bootstrap_repeats']} resamples; random seed {comparison.get('bootstrap_seed', '—')}. Evaluation sample: {comparison.get('n_days', '—')} days and {comparison.get('n_windows', '—')} windows.")
    return output


CSS = """
:root{color-scheme:light;--ink:#233647;--muted:#607285;--line:#dce5eb;--teal:#237f78}
*{box-sizing:border-box}body{margin:0;background:#f5f8fa;color:var(--ink);font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
main{max-width:1120px;margin:36px auto;background:white;padding:42px 54px;border:1px solid var(--line);border-radius:14px}
h1{font-size:34px;line-height:1.18;letter-spacing:-.8px;margin:0 0 18px}h2{font-size:23px;margin:36px 0 12px;padding-top:18px;border-top:1px solid var(--line)}
p{margin:12px 0} .lead{font-size:17px;color:var(--muted)} .eyebrow{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1.6px;color:var(--teal)}
aside{background:#edf6f5;border-left:4px solid var(--teal);padding:16px 20px;margin:22px 0;font-size:14px}aside p{margin:8px 0}
.table-scroll{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th{text-align:left;background:#edf2f6;font-weight:650;padding:11px 13px;white-space:nowrap}td{padding:9px 13px;border-bottom:1px solid var(--line)}tr:nth-child(even){background:#f8fafb}td:first-child{font-weight:500}figure{margin:25px 0}img{display:block;width:100%;height:auto}figcaption{font-size:12px;color:var(--muted);text-align:center;margin-top:8px}details{margin:20px 0;padding:12px 16px;border:1px solid var(--line);border-radius:7px}summary{cursor:pointer;font-weight:600}footer{border-top:1px solid var(--line);padding-top:18px;margin-top:36px;font-size:12px;color:var(--muted)}code{font-size:12px}
@media(max-width:700px){main{margin:0;padding:24px 18px;border:0;border-radius:0}h1{font-size:28px}}
@media print{body{background:white}main{margin:0;padding:0;border:0;max-width:none}figure,aside,tr{break-inside:avoid}h2{break-after:avoid}details{break-inside:avoid}summary{cursor:default}}
"""


def render(record, source="metrics.json"):
    if record.get("schema_version") == 1 and "across_seed_results" in record:
        content = _archive(record)
    else:
        content = _evaluation(record)
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Encoder metrics</title><style>{CSS}</style></head><body><main>"
            "<h1>Encoder metrics</h1>" + content + "<footer>"
            f"Source: <code>{escape(Path(source).name)}</code>. All figures are embedded in this HTML file. "
            "Rendering reads saved measurements; it does not recompute them.</footer></main></body></html>")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Metrics JSON from potofgold-evaluate for one or more encoders, or archived experiment.json")
    parser.add_argument("--out", type=Path, required=True, help="One self-contained .html file")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing report")
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".html":
        parser.error("--out must end in .html")
    if not args.results.is_file():
        message = (f"Results file not found: {args.results}\n"
                   f"Paths are relative to: {Path.cwd()}\n"
                   "First run potofgold-evaluate for the encoder(s) you trained, "
                   f"using --out {shlex.quote(str(args.results))}, then rerun this command. "
                   "One encoder is sufficient; see README steps 5–6.")
        geometry = args.results.parent / "geometry.json"
        if geometry.is_file():
            command = ["potofgold-report", "--results", str(geometry), "--out", str(args.out)]
            if args.overwrite:
                command.append("--overwrite")
            message += ("\nExisting geometry-only results are available. To report those, run:\n"
                        + shlex.join(command))
        parser.error(message)
    if args.out.exists() and not args.overwrite:
        parser.error(f"Output already exists: {args.out}; use --overwrite to replace it")
    try:
        record = json.loads(args.results.read_text())
    except (OSError, ValueError) as error:
        parser.error(f"Cannot read results from {args.results}: {error}. Supply the JSON produced by potofgold-evaluate.")
    output = render(record, args.results.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output)
    print(f"Report: {args.out} (rendered saved measurements; no evaluation or training)")


if __name__ == "__main__":
    main()
