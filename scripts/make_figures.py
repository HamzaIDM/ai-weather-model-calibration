"""Slide-sized figures for the internship defense.

Every number comes from the files the report was built from:
results/tables/*.csv and data/dataset_fusionnée.csv (AROME removed, as in the report).
Figures are exported as PDF at the exact size they occupy on the slide,
so the text in them stays readable on a projector.

Run from the repository root:  python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "defense"
OUT.mkdir(parents=True, exist_ok=True)

# Fira Sans, the slide font, shipped with MiKTeX
FIRA = Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/fira"
for f in FIRA.glob("FiraSans-*.otf"):
    font_manager.fontManager.addfont(str(f))

MM = 1 / 25.4
NAVY, BLUE, TEXT, GREY = "#10348A", "#165EFF", "#1F2937", "#6B7280"
LIGHT, CONTEXT = "#D1D5DB", "#8C94A0"
# Okabe-Ito based, colour-blind safe, one fixed colour per time-driven model
MODEL_COL = {"aifs": "#D55E00", "dlwp": "#AA3377", "graphcast": "#009E73"}
GROUP_COL = {"Coastal": "#2B7BBA", "Inland plains & plateaus": "#9A7B2F",
             "Mountain & pre-Saharan": "#5B2C0F"}
NAMES = {"aifs": "AIFS", "aurora": "Aurora", "dlwp": "DLWP", "fengwu": "FengWu",
         "fourcastnet": "FourCastNet", "fuxi": "FuXi", "graphcast": "GraphCast",
         "pangu6": "Pangu6"}

mpl.rcParams.update({
    "font.family": "Fira Sans", "font.size": 8.5, "axes.titlesize": 8.5,
    "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT,
    "ytick.color": TEXT, "axes.edgecolor": GREY, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 3, "savefig.transparent": True,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("saved", name)


# 1. Mean t2m bias by valid hour (report Fig. 4.3)
def hourly():
    h = pd.read_csv(ROOT / "results/tables/bias_hourly.csv")
    h = h[h.variable_name == "t2m"]
    fig, ax = plt.subplots(figsize=(74 * MM, 56 * MM), layout="constrained")
    ax.axhline(0, color=GREY, lw=0.8)
    for m in [m for m in NAMES if m not in MODEL_COL]:
        d = h[h.model == m].sort_values("hour")
        ax.plot(d.hour, d.biais, color=CONTEXT, lw=1.4, marker="o", ms=2.5, zorder=1)
    ax.text(18.7, 1.0, "5 other\nmodels", color=GREY, va="center", fontsize=7.5,
            linespacing=0.95)
    markers = {"aifs": "o", "dlwp": "s", "graphcast": "^"}
    label_y = {"aifs": 2.75, "dlwp": -0.6, "graphcast": -1.9}
    for m, c in MODEL_COL.items():
        d = h[h.model == m].sort_values("hour")
        ax.plot(d.hour, d.biais, color=c, lw=2.2, marker=markers[m], ms=4.5, zorder=3)
        ax.text(18.7, label_y[m], NAMES[m], color=c, va="center",
                fontweight="semibold")
    ax.set_xticks([0, 6, 12, 18], ["00", "06", "12", "18"])
    ax.set_xlim(-1, 25)
    ax.set_xlabel("valid hour (UTC)")
    ax.set_ylabel("mean t2m bias (°C)")
    ax.set_ylim(-7.2, 6.7)
    ax.text(7.0, -6.35, "−6.35", color=MODEL_COL["aifs"], fontsize=8, va="center")
    ax.text(1.0, 6.25, "+5.77", color=MODEL_COL["dlwp"], fontsize=8, va="center")
    save(fig, "hourly_bias_t2m")


# 2. t2m RMSE per station, grouped as in report Table 3.5 (report Fig. 4.7)
GROUPS = {
    "Coastal": ["AGADIR", "AIT-MELLOUL", "CASA-ANFA", "DAKHLA", "EL-HOCEIMA",
                "EL-JADIDA", "ESSAOUIRA-PORT", "KENITRA", "LAAYOUNE", "LARACHE",
                "MOHAMMEDIA", "RABAT-SALE", "SAFI", "SIDI-IFNI", "TANGER-AERO",
                "TANTAN", "TETOUAN"],
    "Inland plains & plateaus": ["BOUARFA", "FES-SAISS", "GUELMIM", "KHOURIBGA-AERO",
                                 "MARRAKECH", "MEKNES", "NOUASSEUR", "OUJDA", "SETTAT",
                                 "SIDI-SLIMANE", "SMARA", "TAROUDANT", "TIZNIT"],
    "Mountain & pre-Saharan": ["ERRACHIDIA", "IFRANE", "MIDELT", "OUARZAZATE"],
}


def load_t2m():
    df = pd.read_csv(ROOT / "data" / "dataset_fusionnée.csv")
    df = df[df.model.str.lower() != "arome"]
    return df[df.variable_name == "t2m"]


def station_rmse(t2m):
    rmse = t2m.groupby("station").biais.apply(lambda b: np.sqrt((b ** 2).mean()))
    rmse = rmse.sort_values()
    group_of = {s: g for g, ss in GROUPS.items() for s in ss}
    assert set(rmse.index) == set(group_of), "station list differs from Table 3.5"
    fig, ax = plt.subplots(figsize=(74 * MM, 50 * MM), layout="constrained")
    x = np.arange(len(rmse))
    ax.bar(x, rmse.values, width=0.78,
           color=[GROUP_COL[group_of[s]] for s in rmse.index])
    ax.set_xticks([])
    ax.set_xlim(-0.8, len(rmse) - 0.2)
    ax.set_ylim(0, 6.3)
    ax.set_ylabel("t2m RMSE (°C), 8 models")
    ax.set_xlabel("34 stations, sorted from easiest to hardest")
    ax.text(0, rmse.iloc[0] + 0.15, f"Mohammedia\n{rmse.iloc[0]:.2f}", ha="left",
            va="bottom", fontsize=7.5)
    ax.text(len(rmse) - 1, rmse.iloc[-1] + 0.12, f"Ifrane {rmse.iloc[-1]:.2f}",
            ha="right", va="bottom", fontsize=7.5)
    ax.annotate("", xy=(-0.4, 3.75), xytext=(13.4, 3.75),
                arrowprops=dict(arrowstyle="<->", color=GROUP_COL["Coastal"], lw=1))
    ax.text(6.5, 3.9, "14 best: all coastal", ha="center", va="bottom", fontsize=7.5,
            color=GROUP_COL["Coastal"], fontweight="semibold")
    handles = [Patch(color=c, label=f"{g} ({len(GROUPS[g])})")
               for g, c in GROUP_COL.items()]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.5,
              handlelength=1.0, handleheight=0.8, borderaxespad=0.1)
    save(fig, "station_rmse_t2m")


# 3. Range compression: Aurora vs GraphCast (report Fig. 4.6b)
def compression(t2m):
    fig, axes = plt.subplots(1, 2, figsize=(64 * MM, 40 * MM), sharey=True,
                             layout="constrained")
    fig.get_layout_engine().set(w_pad=0.06)
    for ax, m in zip(axes, ["aurora", "graphcast"]):
        d = t2m[t2m.model == m]
        ax.scatter(d.obs_value, d.pred_value, s=1.2, color=NAVY, alpha=0.06,
                   lw=0, rasterized=True)
        ax.plot([-8, 44], [-8, 44], ls="--", color=TEXT, lw=1.0)
        ax.set_xlim(-8, 46)
        ax.set_ylim(-8, 46)
        ax.set_xticks([0, 20, 40])
        ax.set_yticks([0, 20, 40])
        ax.set_title(NAMES[m], fontweight="semibold")
        ax.set_aspect("equal")
    axes[0].set_ylabel("predicted (°C)")
    fig.supxlabel("observed (°C)", fontsize=8.5)
    axes[1].axhline(28, color=GREY, lw=0.7, ls=":")
    axes[1].text(-6, 30.5, "28 °C", color=GREY, fontsize=7.5)
    save(fig, "compression_t2m")


# 4. Mutual information heatmap (report Fig. 4.10).  The CSV rows have no
# label; their order (lead_time, hour, month, station, lat, lon) was matched
# against the bars of results_figures/feature_importance_per_model_t2m.png.
def mutual_information():
    mi = pd.read_csv(ROOT / "results/tables/feature_importance_per_model_t2m.csv")
    mi.index = ["lead time", "hour", "month", "station", "lat", "lon"]
    rows = ["aifs", "dlwp", "graphcast", "aurora", "fengwu", "fourcastnet", "fuxi",
            "pangu6"]
    cols = ["hour", "lead time", "station", "lat", "lon", "month"]
    col_lab = ["hour", "lead\ntime", "station", "lat", "lon", "month"]
    M = mi.loc[cols, rows].T.values
    fig, ax = plt.subplots(figsize=(75 * MM, 50 * MM), layout="constrained")
    ax.imshow(M, cmap="Blues", vmin=0, vmax=0.40, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > 0.22 else TEXT)
    ax.set_xticks(range(len(cols)), col_lab)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [NAMES[r] for r in rows])
    for lab, r in zip(ax.get_yticklabels(), rows):
        if r in MODEL_COL:
            lab.set_color(MODEL_COL[r])
            lab.set_fontweight("semibold")
    ax.axhline(2.5, color="white", lw=3)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for y, lab in [(1.5, "time-driven"), (5.5, "location-driven")]:
        ax.text(1.02, 1.0 - y / 8, lab, transform=ax.transAxes, rotation=270,
                ha="left", va="center", fontsize=7.5, color=TEXT,
                fontweight="semibold")
    save(fig, "mutual_information_t2m")


# 5. Raw vs corrected RMSE, Config A, best algorithm (report Table 4.2)
def dumbbell():
    c = pd.read_csv(ROOT / "results/tables/calibration_results.csv")
    c = c[(c.config == "single") & (c.variable == "t2m")]
    best = c.loc[c.groupby("weather_model").corrected_RMSE.idxmin()]
    best = best.sort_values("RMSE_improvement_pct")
    algo = {"MLP": "MLP", "RandomForest": "Random Forest", "LGBM": "LightGBM",
            "SVR": "SVR"}
    fig, ax = plt.subplots(figsize=(88 * MM, 54 * MM), layout="constrained")
    y = np.arange(len(best))
    for yi, (_, r) in zip(y, best.iterrows()):
        ax.plot([r.corrected_RMSE, r.raw_RMSE], [yi, yi], color=LIGHT, lw=3,
                zorder=1, solid_capstyle="round")
        ax.scatter(r.raw_RMSE, yi, s=34, facecolor="white", edgecolor=GREY, lw=1.4,
                   zorder=2)
        ax.scatter(r.corrected_RMSE, yi, s=38, color=NAVY, zorder=3)
        ax.text(r.raw_RMSE + 0.18, yi, f"−{r.RMSE_improvement_pct:.1f} %",
                va="center", fontsize=8, color=NAVY, fontweight="semibold")
        ax.text(7.0, yi, algo[r.algorithm], va="center", fontsize=7.5, color=GREY)
    ax.set_yticks(y, [NAMES[m] for m in best.weather_model])
    for lab, m in zip(ax.get_yticklabels(), best.weather_model):
        if m in MODEL_COL:
            lab.set_color(MODEL_COL[m])
            lab.set_fontweight("semibold")
    ax.set_xlim(1.0, 8.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("t2m RMSE on the test period (°C)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(7.0, len(best) - 0.35, "best algorithm", fontsize=7.5, color=GREY,
            va="bottom")
    ax.scatter([], [], s=34, facecolor="white", edgecolor=GREY, lw=1.4, label="raw")
    ax.scatter([], [], s=38, color=NAVY, label="corrected")
    ax.legend(loc="lower left", frameon=False, fontsize=7.5, handletextpad=0.2,
              bbox_to_anchor=(0.43, 0.0))
    ax.set_ylim(-0.6, len(best) - 0.1)
    save(fig, "dumbbell_configA_t2m")


# ---------- backup figures ----------
ORDER = ["aifs", "dlwp", "graphcast", "aurora", "fengwu", "fourcastnet", "fuxi",
         "pangu6"]


# 6. Feature importance, Random Forest vs LightGBM (report Figs. 4.12 and 4.13)
def importance():
    rf = pd.read_csv(ROOT / "results/tables/rf_feature_importance_single_t2m.csv",
                     index_col=0)[ORDER]
    lg = pd.read_csv(ROOT / "results/tables/lgbm_feature_importance_single_t2m.csv",
                     index_col=0)[ORDER]
    lg = lg / lg.sum()  # split counts -> share of splits per model
    feats = ["pred_value", "lat", "lon", "lead_time", "hour_sin", "hour_cos",
             "is_afternoon", "is_night"]
    fig, axes = plt.subplots(1, 2, figsize=(144 * MM, 47 * MM), layout="constrained")
    for ax, M, title in [(axes[0], rf.loc[feats], "Random Forest: impurity importance"),
                         (axes[1], lg.loc[feats], "LightGBM: share of splits")]:
        v = M.values
        ax.imshow(v, cmap="Blues", vmin=0, vmax=0.36, aspect="auto")
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, f"{v[i, j]:.2f}", ha="center", va="center", fontsize=6.8,
                        color="white" if v[i, j] > 0.2 else TEXT)
        ax.set_xticks(range(len(ORDER)), [NAMES[m] for m in ORDER], rotation=35,
                      ha="right", fontsize=7.5)
        ax.set_yticks(range(len(feats)), feats, fontsize=7.5)
        for lab, m in zip(ax.get_xticklabels(), ORDER):
            if m in MODEL_COL:
                lab.set_color(MODEL_COL[m])
                lab.set_fontweight("semibold")
        ax.axvline(2.5, color="white", lw=3)
        ax.set_title(title, fontweight="semibold", fontsize=8.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, "importance_rf_lgbm_t2m")


# 7. Mean t2m bias against lead time (t2m panel of report Fig. 4.5)
def leadtime(t2m):
    lt = t2m.groupby(["model", "lead_time"]).biais.mean().reset_index()
    fig, ax = plt.subplots(figsize=(72 * MM, 50 * MM), layout="constrained")
    ax.axhline(0, color=GREY, lw=0.8)
    for m in ORDER:
        d = lt[lt.model == m]
        c = MODEL_COL.get(m, CONTEXT)
        ax.plot(d.lead_time, d.biais, color=c, lw=1.8 if m in MODEL_COL else 1.0,
                zorder=3 if m in MODEL_COL else 1)
    for m, y in [("dlwp", 5.6), ("graphcast", 3.6), ("aifs", -5.6)]:
        ax.text(73, y, NAMES[m], color=MODEL_COL[m], fontsize=7.5,
                fontweight="semibold", va="center")
    ax.set_xticks([0, 24, 48, 72])
    ax.set_xlim(-2, 88)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("mean t2m bias (°C)")
    save(fig, "leadtime_bias_t2m")


# 8. Mean t2m bias per month (report Fig. 4.8)
def monthly():
    mo = pd.read_csv(ROOT / "results/tables/bias_monthly.csv")
    mo = mo[mo.variable_name == "t2m"]
    months = sorted(mo.iloc[:, 2].unique())
    col = mo.columns[2]
    fig, ax = plt.subplots(figsize=(67 * MM, 50 * MM), layout="constrained")
    ax.axhline(0, color=GREY, lw=0.8)
    for m in ORDER:
        d = mo[mo.model == m].sort_values(col)
        x = [months.index(v) for v in d[col]]
        c = MODEL_COL.get(m, CONTEXT)
        ax.plot(x, d.biais, color=c, lw=1.8 if m in MODEL_COL else 1.0, marker="o",
                ms=3 if m in MODEL_COL else 1.5, zorder=3 if m in MODEL_COL else 1)
    for m, y in [("dlwp", 1.6), ("graphcast", -1.05), ("aifs", -0.55)]:
        ax.text(6.25, y, NAMES[m], color=MODEL_COL[m], fontsize=7.5,
                fontweight="semibold", va="center")
    labels = [pd.Timestamp(v + "-01").strftime("%b") for v in months]
    ax.set_xticks(range(len(months)), labels)
    ax.set_xlim(-0.3, 7.6)
    ax.set_ylabel("mean t2m bias (°C)")
    ax.set_xlabel("Sep 2025 to Mar 2026")
    save(fig, "monthly_bias_t2m")


if __name__ == "__main__":
    hourly()
    mutual_information()
    dumbbell()
    importance()
    monthly()
    t2m = load_t2m()
    station_rmse(t2m)
    compression(t2m)
    leadtime(t2m)
