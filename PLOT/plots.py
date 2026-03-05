# === Chapter 6 figure generator ===
# - Uses ONLY matplotlib (no seaborn), one chart per figure.
# - Saves figures to ./figs_ch6 and summary CSVs to /mnt/data.
# - Filenames match the captions in the dissertation text.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path("/mnt/data")
fig_dir = base / "figs_ch6"
fig_dir.mkdir(exist_ok=True)

def read_table(p: Path) -> pd.DataFrame:
    return pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)

# ---------- IoU aggregation ----------
iou_files = {
    "Gear":   base / "Gear_iou_results.xlsx",
    "Flange": base / "flange_iou_results.xlsx",
    "Shaft":  base / "Shaft_iou_results.xlsx",
    "Nut":    base / "Nut_iou_results.xlsx",
}

def normalize_iou_columns(df: pd.DataFrame) -> pd.Series:
    for c in ["iou","IOU","IoU","Iou"]:
        if c in df.columns: return df[c].astype(float)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if num_cols: return df[num_cols[-1]].astype(float)
    raise ValueError("No IoU-like column found.")

frames = []
for cat, path in iou_files.items():
    if not path.exists(): continue
    s = normalize_iou_columns(read_table(path)).clip(0,1)
    frames.append(pd.DataFrame({"category": cat, "iou": s}))
iou_df = pd.concat(frames, ignore_index=True)

iou_summary = (
    iou_df.groupby("category")["iou"]
    .agg(["mean","median","min","max","count"])
    .sort_index()
)
iou_summary.to_csv(base / "iou_summary.csv")

# Figure 6.1 — IoU by part category (boxplot)
plt.figure(figsize=(6,4))
cats = list(iou_summary.index)
data = [iou_df.loc[iou_df["category"]==c,"iou"].dropna().values for c in cats]
plt.boxplot(data, labels=cats, showfliers=True)
plt.ylabel("Intersection over Union (IoU)")
plt.ylim(0,1.05)
plt.title("Figure 6.1 — IoU by part category")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(fig_dir / "Figure6_1_IoU_boxplots.png", dpi=300)
plt.close()

# Figure 6.2 — Reliability curves (share ≥ τ)
thresholds = np.linspace(0.70, 1.00, 7)  # 0.70, 0.75, ..., 1.00
plt.figure(figsize=(6,4))
for c in cats:
    vals = iou_df.loc[iou_df["category"]==c,"iou"].dropna().values
    shares = [(vals >= t).mean() for t in thresholds]
    plt.step(thresholds, shares, where="post", label=c)
plt.xlabel("IoU threshold (τ)")
plt.ylabel("Proportion of models with IoU ≥ τ")
plt.ylim(0,1.05); plt.xlim(thresholds.min(), thresholds.max())
plt.title("Figure 6.2 — Reliability of geometric fidelity")
plt.legend(frameon=False)
plt.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(fig_dir / "Figure6_2_Reliability_curves.png", dpi=300)
plt.close()

# ---------- JSON plan validity & compactness (all 9 categories) ----------
json_files = {
    "Arduino":   base / "Arduino_json_eval_recursive_metrics.xlsx",
    "Battery":   base / "Battery_holder_json_eval_recursive_metrics.xlsx",
    "Sensor":    base / "Sensor_Housing_json_eval_recursive_metrics.xlsx",
    "Switch":    base / "Switch_Box_json_eval_recursive_metrics.xlsx",
    "WallMount": base / "Wall_mount_json_eval_recursive_metrics.xlsx",
    "Flange":    base / "Flange_json_eval_recursive_metrics.xlsx",
    "Gear":      base / "Gear_json_eval_recursive_metrics.xlsx",
    "Nut":       base / "Nut_json_eval_recursive_metrics.xlsx",
    "Shaft":     base / "Shaft_json_eval_recursive_metrics.xlsx",
}

rows = []
for name, path in json_files.items():
    if not path.exists(): continue
    df = read_table(path)
    rows.append({
        "category": name,
        "ValidJSONRate":       df["valid_json"].mean(),
        "ConsistencyOKRate":   df["consistency_ok"].mean(),
        "ConsistencyErrors":   df["consistency_errors"].mean(),
        "MeanTreeDepth":       df["tree_depth"].mean(),
        "MeanJSONBytes":       df["minified_bytes"].mean(),
        "MeanCompactness":     df["compactness"].mean(),
        "n": len(df),
    })
json_summary = pd.DataFrame(rows).set_index("category").sort_index()
json_summary.to_csv(base / "json_summary.csv")

# Figure 6.3a — Mean tree depth
plt.figure(figsize=(6,4))
plt.bar(json_summary.index, json_summary["MeanTreeDepth"].values)
plt.ylabel("Mean plan tree depth")
plt.title("Figure 6.3a — Plan compactness: depth by category")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(fig_dir / "Figure6_3a_TreeDepth.png", dpi=300)
plt.close()

# Figure 6.3b — Mean JSON size (bytes)
plt.figure(figsize=(6,4))
plt.bar(json_summary.index, json_summary["MeanJSONBytes"].values)
plt.ylabel("Mean minified JSON size (bytes)")
plt.title("Figure 6.3b — Plan compactness: bytes by category")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(fig_dir / "Figure6_3b_JSONBytes.png", dpi=300)
plt.close()

# ---------- Synthetic prompt metrics (5 scenarios) ----------
prompt_files = {
    "Arduino":   base / "05_ARDUINO_CASE_PROMPT_metrics.csv",
    "Battery":   base / "06_BATTERY_HOLDER_PROMPTS_metrics.csv",
    "Sensor":    base / "07_SENSOR_HOUING_PROMPT_metrics.csv",
    "Switch":    base / "08_SWITCH_BOX_PROMPTS_metrics.csv",
    "WallMount": base / "09_WALL_MOUNT_PROMPTS_metrics.csv",
}
prom = []
for name, path in prompt_files.items():
    if not path.exists(): continue
    df = read_table(path)
    prom.append({
        "scenario": name,
        "words": df["words"].mean(),
        "fk_grade": df["readability_fkgl"].mean(),
        "ttr": df["lexdiv_ttr"].mean(),
        "coherence": df["coherence_adjacent_cos"].mean(),
        "grammar_err_rate": df["grammar_error_rate"].mean(),
        "perplexity": df["perplexity"].mean(),
    })
ps = pd.DataFrame(prom).set_index("scenario").sort_index()
ps.to_csv(base / "prompt_summary.csv")

def simple_bar(y, ylabel, title, filename):
    plt.figure(figsize=(6,4))
    plt.bar(ps.index, ps[y].values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(fig_dir / filename, dpi=300)
    plt.close()

simple_bar("words", "Average words", "Figure 6.4a — Prompt length by scenario", "Figure6_4a_PromptLength.png")
simple_bar("fk_grade", "Flesch–Kincaid grade", "Figure 6.4b — Readability by scenario", "Figure6_4b_FKGrade.png")
simple_bar("ttr", "Type–Token Ratio (TTR)", "Figure 6.4c — Lexical diversity by scenario", "Figure6_4c_TTR.png")
simple_bar("coherence", "Adjacent sentence cosine", "Figure 6.4d — Coherence by scenario", "Figure6_4d_Coherence.png")
simple_bar("grammar_err_rate", "Grammar error rate", "Figure 6.4e — Grammar errors by scenario", "Figure6_4e_Grammar.png")
simple_bar("perplexity", "Perplexity (model-dependent)", "Figure 6.4f — Prompt perplexity by scenario", "Figure6_4f_Perplexity.png")

# ---------- Optional: Case-study panel (supply image paths when you have them) ----------
# Example usage:
# from pathlib import Path
# imgs = [Path("/path/to/easy.png"), Path("/path/to/medium.png"), Path("/path/to/hard.png"),
#         Path("/path/to/easy2.png"), Path("/path/to/medium2.png"), Path("/path/to/hard2.png")]
# make_case_study_panel(imgs, fig_dir / "Figure6_5_CaseStudy.png", ncols=3)

def make_case_study_panel(image_paths, out_path, ncols=3):
    import matplotlib.image as mpimg, math
    n = len(image_paths)
    if n == 0: return
    nrows = int(math.ceil(n / ncols))
    plt.figure(figsize=(4*ncols, 3*nrows))
    for i, p in enumerate(image_paths):
        ax = plt.subplot(nrows, ncols, i+1)
        if Path(p).exists():
            ax.imshow(mpimg.imread(p)); ax.axis("off"); ax.set_title(Path(p).stem, fontsize=9)
        else:
            ax.text(0.5, 0.5, f"Missing:\n{Path(p).name}", ha="center", va="center"); ax.axis("off")
    plt.suptitle("Figure 6.5 — Case-study panel", y=0.98)
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()
