"""
Paper §5.1 Fig 5 — Temperature × back-translation consistency plot.

For each (model × language) reads the best-candidate JSONL produced by
`02_translation/translate_find_bestCode_from_back_vllm.py`, groups the back-
translation BLEU / SBERT scores by sampling temperature τ ∈ {0, 0.7, 0.9, 1.1},
and plots:

    - solid line = mean score at τ            ("Average")
    - dashed line = max score at τ            ("Oracle / theoretical upper bound")

The resulting two-panel figure (BLEU left, SBERT right) is Fig 5 in the paper.
The paper's TikZ/pgfplots version of this figure reads the same underlying
numbers; this matplotlib script is the standalone reproducer.

Input file layout (default):
    ./data/LowData/<lang>/trans_<base>/<lang>_python_best_candidate_vllm_<suffix>.jsonl
where <base> ∈ {ds_small3th, ds_mid3th, Llama3th, Seed3th, qwen3th}.

Usage:
    python plot_temperature_picture.py
    python plot_temperature_picture.py --root ./data/LowData --suffix _python_best_candidate_vllm_B0.5_S0.5.jsonl --out fig5.png
"""

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------- Defaults (match the paper's experimental layout) ----------
DEFAULT_ROOT   = "./data/LowData"
DEFAULT_SUFFIX = "_python_best_candidate_vllm_B0.5_S0.5.jsonl"
DEFAULT_OUT    = "fig5_temperature_backtranslation.png"

MODELS_MAPPING = {
    "deepseek-coder-1.3b-instruct": "DS-1.3B",
    "deepseek-coder-6.7b-instruct": "DS-6.7B",
    "Llama-3.1-8B-Instruct":        "Llama-3-8B",
    "Seed-Coder-8B-Instruct":       "Seed-8B",
    "Qwen2.5-Coder-14B-Instruct":   "Qwen-14B",
}

LANGUAGES    = ["julia", "lua", "ocaml", "racket", "r"]
TARGET_TEMPS = [0.0, 0.7, 0.9, 1.1]


# ---------- Folder-name convention (server-side experimental layout) ----------
def _base_subdir(model_folder: str) -> str:
    if "1.3"   in model_folder: return "trans_ds_small3th"
    if "6.7"   in model_folder: return "trans_ds_mid3th"
    if "Llama" in model_folder: return "trans_Llama3th"
    if "Seed"  in model_folder: return "trans_Seed3th"
    if "Qwen"  in model_folder: return "trans_qwen3th"
    return ""


def get_file_path(root: str, model_folder: str, lang: str, suffix: str) -> str:
    return os.path.join(root, lang, _base_subdir(model_folder), f"{lang}{suffix}")


# ---------- Load mean / max BLEU & SBERT per (model, τ) ----------
def load_data_mean_max(root: str, suffix: str) -> pd.DataFrame:
    rows: List[dict] = []
    for model_folder, model_label in MODELS_MAPPING.items():
        for lang in LANGUAGES:
            file_path = get_file_path(root, model_folder, lang, suffix)
            if not os.path.exists(file_path):
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        temp_groups = {t: {"bleu": [], "sbert": []} for t in TARGET_TEMPS}

                        for trans in item.get("translations", []):
                            t = trans.get("temperature")
                            if t is None:
                                continue
                            t = float(t)
                            matched_t = next(
                                (target for target in TARGET_TEMPS if abs(t - target) < 1e-4),
                                None,
                            )
                            if matched_t is None:
                                continue
                            scores = trans.get("scores", {}) or {}
                            temp_groups[matched_t]["bleu"].append(scores.get("bleu", 0.0))
                            temp_groups[matched_t]["sbert"].append(scores.get("jina", 0.0))

                        for t in TARGET_TEMPS:
                            bleus  = temp_groups[t]["bleu"]
                            sberts = temp_groups[t]["sbert"]
                            if bleus:
                                rows.append({
                                    "Model": model_label, "Temperature": t, "Metric": "BLEU",
                                    "Mean": sum(bleus) / len(bleus), "Max": max(bleus),
                                })
                            if sberts:
                                rows.append({
                                    "Model": model_label, "Temperature": t, "Metric": "SBERT",
                                    "Mean": sum(sberts) / len(sberts), "Max": max(sberts),
                                })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return pd.DataFrame(rows)


# ---------- Plotting ----------
def plot_figure(df: pd.DataFrame, out_path: str) -> None:
    df_plot = df.groupby(["Model", "Temperature", "Metric"])[["Mean", "Max"]].mean().reset_index()

    sns.set_style("whitegrid")
    sns.set_context("talk")

    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=True)
    palette = sns.color_palette("tab10", n_colors=len(MODELS_MAPPING))

    def _plot_metric(ax, metric_name: str, title: str) -> None:
        data = df_plot[df_plot["Metric"] == metric_name]

        # Solid: mean across the population at each τ.
        sns.lineplot(
            data=data, x="Temperature", y="Mean", hue="Model",
            palette=palette, style="Model", markers=True, dashes=False,
            linewidth=2.5, ax=ax, alpha=0.7, legend=False,
        )

        # Dashed: oracle upper-bound (max across candidates) at each τ.
        sns.lineplot(
            data=data, x="Temperature", y="Max", hue="Model",
            palette=palette, legend=False, ax=ax, linewidth=2.5,
        )

        # The second sns.lineplot adds n_models more Line2D objects; mark them dashed.
        lines = ax.get_lines()
        n_models = len(data["Model"].unique())
        for line in lines[n_models:]:
            line.set_linestyle("--")
            line.set_alpha(1.0)

        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.set_xticks(TARGET_TEMPS)
        ax.set_xlabel("Temperature", fontsize=14)
        ax.set_ylabel(f"{metric_name} Score", fontsize=14)
        ax.tick_params(labelsize=12)

    _plot_metric(axes[0], "BLEU",  "BLEU: Average (Solid) vs Oracle (Dashed)")
    _plot_metric(axes[1], "SBERT", "SBERT: Average (Solid) vs Oracle (Dashed)")

    # Manual legend (seaborn sorts hue alphabetically; reuse that order).
    sorted_models = sorted(MODELS_MAPPING.values())
    handles, labels = [], []
    for i, model_label in enumerate(sorted_models):
        handles.append(plt.Line2D([0], [0], color=palette[i], linewidth=2.5,
                                  marker="o", markersize=8, label=model_label))
        labels.append(model_label)

    solid_line  = plt.Line2D([0], [0], color="gray", linewidth=2.5, linestyle="-",  label="Average")
    dashed_line = plt.Line2D([0], [0], color="gray", linewidth=2.5, linestyle="--", label="Oracle")

    fig.legend(handles, labels,
               loc="center left", bbox_to_anchor=(0.92, 0.55),
               fontsize=13, frameon=True, title="Model", title_fontsize=14)
    fig.legend([solid_line, dashed_line], ["Average", "Oracle"],
               loc="center left", bbox_to_anchor=(0.92, 0.30),
               fontsize=13, frameon=True, title="Line Type", title_fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",   default=DEFAULT_ROOT,
                        help=f"Root directory holding per-language back-translation JSONL (default: {DEFAULT_ROOT}).")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX,
                        help=f"Best-candidate file suffix (default: {DEFAULT_SUFFIX}).")
    parser.add_argument("--out",    default=DEFAULT_OUT,
                        help=f"Output PNG path (default: {DEFAULT_OUT}).")
    parser.add_argument("--show",   action="store_true",
                        help="Also call plt.show() after saving.")
    args = parser.parse_args()

    print(f"Loading from: {args.root}  (suffix: {args.suffix})")
    df = load_data_mean_max(args.root, args.suffix)
    if df.empty:
        print("No data found — check --root / --suffix arguments.")
        return

    plot_figure(df, args.out)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
