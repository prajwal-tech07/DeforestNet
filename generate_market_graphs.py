"""
DeforestNet - Professional Market & Competitor Analysis Graphs
Generates high-quality PNG charts for investor/industry pitch deck.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "docs" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

# ── Global style ──────────────────────────────────────────────
BG       = "#0f172a"
CARD     = "#1e293b"
TEXT     = "#e2e8f0"
GRID     = "#334155"
GREEN    = "#10b981"
RED      = "#ef4444"
AMBER    = "#f59e0b"
BLUE     = "#3b82f6"
PURPLE   = "#8b5cf6"
CYAN     = "#06b6d4"
PINK     = "#ec4899"
LIME     = "#84cc16"
ORANGE   = "#f97316"
TEAL     = "#14b8a6"
PALETTE  = [GREEN, BLUE, PURPLE, AMBER, RED, CYAN, PINK, LIME, ORANGE, TEAL]

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   CARD,
    "axes.edgecolor":   GRID,
    "axes.labelcolor":  TEXT,
    "text.color":       TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "grid.color":       GRID,
    "grid.alpha":       0.3,
    "font.family":      "sans-serif",
    "font.size":        11,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] {name}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 1 – Market Growth Forecast (2024-2032)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_market_growth():
    fig, ax = plt.subplots(figsize=(12, 6))

    years = np.arange(2024, 2033)

    # Satellite EO market (19.9% CAGR from $11.83B)
    sat_eo = 11.83 * (1.199 ** (years - 2024))
    # GeoAI market (11.1% CAGR from $32.38B)
    geoai = 32.38 * (1.111 ** (years - 2024))
    # Forest monitoring niche (20% CAGR from $1.8B)
    forest = 1.8 * (1.20 ** (years - 2024))

    ax.fill_between(years, geoai, alpha=0.15, color=BLUE)
    ax.fill_between(years, sat_eo, alpha=0.15, color=GREEN)
    ax.fill_between(years, forest, alpha=0.25, color=PURPLE)

    ax.plot(years, geoai,  "o-", color=BLUE,   lw=2.5, ms=7, label=f"GeoAI / Geospatial Analytics  (11.1% CAGR)")
    ax.plot(years, sat_eo, "s-", color=GREEN,  lw=2.5, ms=7, label=f"Satellite EO Services  (19.9% CAGR)")
    ax.plot(years, forest, "D-", color=PURPLE, lw=2.5, ms=7, label=f"Forest Monitoring Niche  (~20% CAGR)")

    # Annotations
    ax.annotate(f"${geoai[-1]:.0f}B", (2032, geoai[-1]), textcoords="offset points",
                xytext=(10, 0), fontsize=12, fontweight="bold", color=BLUE)
    ax.annotate(f"${sat_eo[-1]:.0f}B", (2032, sat_eo[-1]), textcoords="offset points",
                xytext=(10, 0), fontsize=12, fontweight="bold", color=GREEN)
    ax.annotate(f"${forest[-1]:.1f}B", (2032, forest[-1]), textcoords="offset points",
                xytext=(10, -15), fontsize=12, fontweight="bold", color=PURPLE)

    # EUDR deadline marker
    ax.axvline(2026, color=RED, ls="--", lw=1.5, alpha=0.7)
    ax.text(2026.08, ax.get_ylim()[1] * 0.92, "EUDR\nDeadline", fontsize=9,
            color=RED, fontweight="bold", va="top")

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Market Size (USD Billions)", fontsize=12)
    ax.set_title("Satellite & Forest Monitoring Market Growth Forecast",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.3, edgecolor=GRID)
    ax.grid(True, axis="y")
    ax.set_xticks(years)

    save(fig, "01_market_growth_forecast.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 2 – Feature Comparison Heatmap (DeforestNet vs Competitors)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_feature_heatmap():
    competitors = [
        "DeforestNet", "GFW/GLAD", "Planet", "Satelligence",
        "Kayrros", "SarVision", "MapBiomas"
    ]
    features = [
        "SAR Integration",
        "Optical Data",
        "Multi-Sensor Fusion",
        "Cloud Penetration",
        "6-Class Cause ID",
        "Deep Learning",
        "10m Resolution",
        "EUDR Compliance",
        "Real-Time Alerts",
        "Free Data Source",
        "Explainable AI",
        "Open Methodology",
    ]

    # Score: 0=No, 0.5=Partial, 1=Yes
    data = np.array([
        # DeforestNet  GFW   Planet Satel  Kayrros SarV  MapBio
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],  # SAR
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],  # Optical
        [1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],  # Fusion
        [1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.0],  # Cloud
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 6-class
        [1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.5],  # DL
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],  # 10m
        [1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0],  # EUDR
        [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5],  # RT alerts
        [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0],  # Free data
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Explainable
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # Open
    ])

    fig, ax = plt.subplots(figsize=(13, 8))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("custom", ["#1e293b", "#334155", GREEN])

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(competitors)))
    ax.set_xticklabels(competitors, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=10)

    # Add text labels
    for i in range(len(features)):
        for j in range(len(competitors)):
            val = data[i, j]
            label = "YES" if val == 1 else ("PARTIAL" if val == 0.5 else "NO")
            color = "#ffffff" if val >= 0.5 else "#64748b"
            fontw = "bold" if val == 1 else "normal"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color=color, fontweight=fontw)

    # Highlight DeforestNet column
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, len(features),
                                fill=False, edgecolor=GREEN, lw=3))

    ax.set_title("Feature Comparison: DeforestNet vs Industry Competitors",
                 fontsize=15, fontweight="bold", pad=15)

    # Score totals at bottom
    totals = data.sum(axis=0)
    for j, total in enumerate(totals):
        ax.text(j, len(features) - 0.15, f"Score: {total:.0f}/12",
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=GREEN if j == 0 else AMBER)

    fig.tight_layout()
    save(fig, "02_feature_comparison_heatmap.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 3 – Radar/Spider Chart: Technical Capabilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_radar_chart():
    categories = [
        "Spectral\nBands", "Spatial\nResolution", "Cloud\nPenetration",
        "Cause\nClassification", "AI/ML\nSophistication", "Real-Time\nCapability",
        "Cost\nEfficiency", "Explainability"
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Scores out of 10
    deforestnet = [10, 8, 10, 10, 9, 8, 10, 9]
    gfw         = [4,  3,  1,  1,  3, 5,  10, 3]
    satelligence= [7,  8,  8,  2,  8, 8,  2,  2]
    planet      = [6,  10, 1,  1,  5, 9,  3,  2]

    deforestnet += deforestnet[:1]
    gfw         += gfw[:1]
    satelligence+= satelligence[:1]
    planet      += planet[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD)

    ax.fill(angles, deforestnet, alpha=0.25, color=GREEN)
    ax.plot(angles, deforestnet, "o-", color=GREEN, lw=2.5, ms=8, label="DeforestNet")

    ax.fill(angles, satelligence, alpha=0.1, color=BLUE)
    ax.plot(angles, satelligence, "s--", color=BLUE, lw=1.5, ms=6, label="Satelligence")

    ax.fill(angles, planet, alpha=0.1, color=AMBER)
    ax.plot(angles, planet, "D--", color=AMBER, lw=1.5, ms=6, label="Planet Labs")

    ax.fill(angles, gfw, alpha=0.1, color=PURPLE)
    ax.plot(angles, gfw, "^--", color=PURPLE, lw=1.5, ms=6, label="GFW / GLAD")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 11)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color=GRID)
    ax.grid(color=GRID, alpha=0.3)

    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05),
              fontsize=11, framealpha=0.3, edgecolor=GRID)
    ax.set_title("Technical Capability Comparison",
                 fontsize=16, fontweight="bold", pad=30, y=1.05)

    save(fig, "03_radar_capability_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 4 – Annual Data Cost Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_cost_comparison():
    fig, ax = plt.subplots(figsize=(11, 6))

    systems = [
        "DeforestNet\n(Sentinel Free)",
        "GFW / GLAD\n(Landsat Free)",
        "MapBiomas\n(Landsat Free)",
        "SarVision\n(Enterprise)",
        "Kayrros\n(Enterprise)",
        "Satelligence\n(Enterprise)",
        "Planet Labs\n(Commercial)",
        "Maxar/Vantor\n(VHR Tasking)"
    ]
    costs = [0, 0, 0, 150, 250, 200, 300, 500]
    colors = [GREEN, PURPLE, PURPLE, AMBER, AMBER, BLUE, AMBER, RED]

    bars = ax.barh(range(len(systems)), costs, color=colors, height=0.6,
                   edgecolor="#ffffff10", linewidth=0.5)

    # Add cost labels
    for i, (bar, cost) in enumerate(zip(bars, costs)):
        if cost == 0:
            ax.text(8, i, "FREE (ESA Open Data)", va="center", fontsize=11,
                    fontweight="bold", color=GREEN)
        else:
            ax.text(cost + 8, i, f"~${cost}K/year", va="center", fontsize=11,
                    fontweight="bold", color=colors[i])

    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=10)
    ax.set_xlabel("Estimated Annual Data + Platform Cost (USD Thousands)", fontsize=11)
    ax.set_title("Annual Cost Comparison: Satellite Data & Analytics",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(0, 580)
    ax.grid(True, axis="x", alpha=0.2)
    ax.invert_yaxis()

    # Green highlight for DeforestNet
    ax.get_yticklabels()[0].set_color(GREEN)
    ax.get_yticklabels()[0].set_fontweight("bold")

    save(fig, "04_cost_comparison.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 5 – Resolution vs Capability Bubble Chart
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_resolution_bubble():
    fig, ax = plt.subplots(figsize=(12, 7))

    # x = resolution (m), y = capability score (out of 10), size = data cost inverse
    competitors = {
        "DeforestNet":   {"res": 10,  "cap": 9.5, "cost_inv": 900, "color": GREEN},
        "GFW / GLAD":    {"res": 30,  "cap": 4.0, "cost_inv": 900, "color": PURPLE},
        "MapBiomas":     {"res": 30,  "cap": 5.0, "cost_inv": 900, "color": TEAL},
        "SarVision":     {"res": 10,  "cap": 6.5, "cost_inv": 300, "color": CYAN},
        "Kayrros":       {"res": 10,  "cap": 6.0, "cost_inv": 200, "color": ORANGE},
        "Satelligence":  {"res": 10,  "cap": 7.5, "cost_inv": 250, "color": BLUE},
        "Planet Labs":   {"res": 3.5, "cap": 5.5, "cost_inv": 150, "color": AMBER},
        "Maxar/Vantor":  {"res": 0.3, "cap": 3.0, "cost_inv": 80,  "color": RED},
    }

    for name, d in competitors.items():
        ax.scatter(d["res"], d["cap"], s=d["cost_inv"], color=d["color"],
                   alpha=0.7, edgecolors="white", linewidth=1.5, zorder=3)
        offset_x = 1.5 if d["res"] > 5 else 0.3
        offset_y = 0.35
        if name == "MapBiomas":
            offset_y = -0.45
        if name == "Kayrros":
            offset_y = -0.45
        ax.annotate(name, (d["res"], d["cap"]),
                    xytext=(offset_x, offset_y), textcoords="offset fontsize",
                    fontsize=10, fontweight="bold", color=d["color"])

    ax.set_xscale("log")
    ax.set_xticks([0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.3m", "1m", "3m", "10m", "30m"])
    ax.set_xlabel("Spatial Resolution (log scale, lower = finer)", fontsize=12)
    ax.set_ylabel("Overall Capability Score (10 = best)", fontsize=12)
    ax.set_title("Resolution vs Capability vs Cost Efficiency",
                 fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(1.5, 11)

    # Legend for bubble size
    for size, label in [(900, "Free/Low Cost"), (300, "Medium"), (80, "Expensive")]:
        ax.scatter([], [], s=size, color=GRID, alpha=0.5, edgecolors="white",
                   linewidth=1, label=label)
    ax.legend(title="Bubble Size = Cost Efficiency", loc="lower left",
              fontsize=9, title_fontsize=10, framealpha=0.3, edgecolor=GRID)

    # Highlight DeforestNet zone
    from matplotlib.patches import FancyBboxPatch
    rect = FancyBboxPatch((6, 8.8), 8, 1.6, boxstyle="round,pad=0.3",
                           facecolor=GREEN, alpha=0.08, edgecolor=GREEN, lw=1.5, ls="--")
    ax.add_patch(rect)
    ax.text(7, 10.55, "OPTIMAL ZONE", fontsize=9, color=GREEN, fontweight="bold", alpha=0.7)

    save(fig, "05_resolution_vs_capability.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 6 – Deforestation Crisis: Annual Tropical Forest Loss
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_deforestation_crisis():
    fig, ax = plt.subplots(figsize=(12, 6))

    years = list(range(2015, 2025))
    # Approximate data from WRI Global Forest Review (Mha tropical primary forest loss)
    loss = [3.1, 3.0, 3.9, 3.6, 3.8, 4.2, 3.8, 4.1, 3.7, 6.7]

    colors = [RED if v > 4.0 else AMBER if v > 3.5 else BLUE for v in loss]
    bars = ax.bar(years, loss, color=colors, width=0.7, edgecolor="#ffffff10", linewidth=0.5)

    # Highlight 2024
    bars[-1].set_edgecolor(RED)
    bars[-1].set_linewidth(2.5)
    ax.annotate("RECORD\n6.7 Mha\n(+80% YoY)", (2024, 6.7),
                xytext=(0, 15), textcoords="offset points",
                fontsize=12, fontweight="bold", color=RED,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    # Glasgow target line
    ax.axhline(y=0, color=GREEN, ls="--", lw=1.5, alpha=0.5)
    # Target: reach zero by 2030, so declining from 2021 levels
    target_years = np.array([2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030])
    target_loss = np.linspace(3.8, 0, len(target_years))
    ax.plot(target_years, target_loss, "--", color=GREEN, lw=2, alpha=0.6,
            label="Glasgow 2030 Target Path")
    ax.fill_between(target_years, target_loss, alpha=0.05, color=GREEN)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Tropical Primary Forest Loss (Million Hectares)", fontsize=12)
    ax.set_title("Global Deforestation Crisis: Why Better Monitoring is Urgent",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=10, framealpha=0.3, edgecolor=GRID)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_xticks(years)
    ax.set_ylim(0, 8.5)

    # Add CO2 annotation
    ax.text(2015.3, 7.8, "2024: 3.1 Gt CO2e emissions from tropical deforestation alone",
            fontsize=10, color=AMBER, fontstyle="italic")

    save(fig, "06_deforestation_crisis.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 7 – Binary vs Multi-Class: The Gap DeforestNet Fills
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_classification_gap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: What competitors see (binary)
    ax = axes[0]
    labels_bin = ["Forest\n(No Change)", "Non-Forest\n(Deforestation)"]
    sizes_bin = [65, 35]
    colors_bin = [GREEN, RED]
    wedges, texts, autotexts = ax.pie(
        sizes_bin, labels=labels_bin, colors=colors_bin, autopct="%1.0f%%",
        startangle=90, textprops={"fontsize": 12, "color": TEXT},
        wedgeprops={"edgecolor": BG, "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(14)
    ax.set_title("Industry Standard\n(Binary Classification)", fontsize=14,
                 fontweight="bold", pad=15, color=RED)
    ax.text(0, -1.35, '"WHAT happened"\nbut not WHY',
            ha="center", fontsize=11, fontstyle="italic", color="#94a3b8")

    # Right: What DeforestNet sees (6-class)
    ax = axes[1]
    labels_mc = ["Forest", "Logging", "Mining", "Agriculture", "Fire", "Infrastructure"]
    sizes_mc = [50, 12, 10, 15, 8, 5]
    colors_mc = [GREEN, ORANGE, RED, AMBER, "#dc2626", PURPLE]
    explode = [0, 0.05, 0.05, 0.05, 0.05, 0.05]
    wedges, texts, autotexts = ax.pie(
        sizes_mc, labels=labels_mc, colors=colors_mc, autopct="%1.0f%%",
        startangle=90, explode=explode,
        textprops={"fontsize": 10, "color": TEXT},
        wedgeprops={"edgecolor": BG, "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(11)
    ax.set_title("DeforestNet\n(6-Class Cause Identification)", fontsize=14,
                 fontweight="bold", pad=15, color=GREEN)
    ax.text(0, -1.35, '"WHAT happened AND WHY"\nActionable intelligence for EUDR, carbon credits, enforcement',
            ha="center", fontsize=10, fontstyle="italic", color="#94a3b8")

    fig.suptitle("The Classification Gap: Why Cause Identification Matters",
                 fontsize=16, fontweight="bold", y=1.02)

    fig.tight_layout()
    save(fig, "07_classification_gap.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 8 – Cloud Cover Problem: Why SAR Matters
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_cloud_cover():
    fig, ax = plt.subplots(figsize=(12, 6))

    regions = [
        "Amazon\nBasin", "Congo\nBasin", "SE Asia\n(Borneo)", "Western\nGhats (India)",
        "Central\nAmerica", "West\nAfrica"
    ]
    cloud_pct = [87, 80, 75, 70, 72, 68]
    usable_optical = [13, 20, 25, 30, 28, 32]

    x = np.arange(len(regions))
    w = 0.35

    bars1 = ax.bar(x - w/2, cloud_pct, w, color=RED, alpha=0.8, label="Cloud-Blocked (%)")
    bars2 = ax.bar(x + w/2, usable_optical, w, color=BLUE, alpha=0.8, label="Usable Optical (%)")

    # SAR line at 100%
    ax.axhline(y=95, color=GREEN, ls="-", lw=2.5, alpha=0.8)
    ax.text(len(regions) - 0.5, 96.5, "SAR Availability: ~95-100%",
            fontsize=12, fontweight="bold", color=GREEN, ha="right")

    for bar, val in zip(bars1, cloud_pct):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val}%",
                ha="center", fontsize=10, fontweight="bold", color=RED)
    for bar, val in zip(bars2, usable_optical):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val}%",
                ha="center", fontsize=10, fontweight="bold", color=BLUE)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10)
    ax.set_ylabel("Percentage of Observations (%)", fontsize=11)
    ax.set_title("The Cloud Cover Problem in Tropical Forest Monitoring",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=10, loc="center right", framealpha=0.3, edgecolor=GRID)
    ax.grid(True, axis="y", alpha=0.15)

    ax.text(0.5, -0.13, "Optical-only systems (GFW, Planet, MapBiomas) lose 68-87% of observations. "
            "DeforestNet's SAR+Optical fusion ensures continuous monitoring.",
            transform=ax.transAxes, ha="center", fontsize=10, color="#94a3b8",
            fontstyle="italic")

    save(fig, "08_cloud_cover_problem.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 9 – 11-Band Architecture Diagram
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_band_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.5, "DeforestNet: 11-Band Multi-Sensor Input Architecture",
            ha="center", fontsize=16, fontweight="bold", color=TEXT)

    # Sentinel-2 box
    s2_box = mpatches.FancyBboxPatch((0.5, 5.5), 4.5, 3.2, boxstyle="round,pad=0.2",
                                       facecolor="#1e3a5f", edgecolor=BLUE, lw=2)
    ax.add_patch(s2_box)
    ax.text(2.75, 8.3, "SENTINEL-2 (Optical)", ha="center", fontsize=12,
            fontweight="bold", color=BLUE)

    s2_bands = [
        ("B2", "Blue", "490nm", "#3b82f6"),
        ("B3", "Green", "560nm", "#22c55e"),
        ("B4", "Red", "665nm", "#ef4444"),
        ("B8", "NIR", "842nm", "#a855f7"),
    ]
    for i, (band, name, wl, color) in enumerate(s2_bands):
        y = 7.6 - i * 0.5
        rect = mpatches.FancyBboxPatch((0.8, y - 0.18), 3.9, 0.36,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, alpha=0.3, edgecolor=color, lw=1)
        ax.add_patch(rect)
        ax.text(1.0, y, f"{band}", fontsize=10, fontweight="bold", color=color, va="center")
        ax.text(2.0, y, f"{name}", fontsize=9, color=TEXT, va="center")
        ax.text(4.4, y, f"{wl}", fontsize=9, color="#94a3b8", va="center", ha="right")

    # Sentinel-1 box
    s1_box = mpatches.FancyBboxPatch((0.5, 3.3), 4.5, 1.8, boxstyle="round,pad=0.2",
                                       facecolor="#3a1e3a", edgecolor=PINK, lw=2)
    ax.add_patch(s1_box)
    ax.text(2.75, 4.8, "SENTINEL-1 (SAR C-Band)", ha="center", fontsize=12,
            fontweight="bold", color=PINK)

    s1_bands = [
        ("VV", "Co-Polarized", "5.405 GHz", PINK),
        ("VH", "Cross-Polarized", "5.405 GHz", PURPLE),
    ]
    for i, (band, name, freq, color) in enumerate(s1_bands):
        y = 4.2 - i * 0.5
        rect = mpatches.FancyBboxPatch((0.8, y - 0.18), 3.9, 0.36,
                                        boxstyle="round,pad=0.05",
                                        facecolor=color, alpha=0.3, edgecolor=color, lw=1)
        ax.add_patch(rect)
        ax.text(1.0, y, f"{band}", fontsize=10, fontweight="bold", color=color, va="center")
        ax.text(2.0, y, f"{name}", fontsize=9, color=TEXT, va="center")
        ax.text(4.4, y, f"{freq}", fontsize=9, color="#94a3b8", va="center", ha="right")

    # Derived indices box
    d_box = mpatches.FancyBboxPatch((0.5, 0.5), 4.5, 2.4, boxstyle="round,pad=0.2",
                                      facecolor="#1e3a2a", edgecolor=GREEN, lw=2)
    ax.add_patch(d_box)
    ax.text(2.75, 2.6, "DERIVED INDICES", ha="center", fontsize=12,
            fontweight="bold", color=GREEN)

    derived = [
        ("NDVI", "(NIR-Red)/(NIR+Red)", GREEN),
        ("EVI",  "Enhanced Veg Index", LIME),
        ("SAVI", "Soil-Adjusted VI", TEAL),
        ("VV/VH", "SAR Ratio", CYAN),
        ("RVI",  "Radar Veg Index", AMBER),
    ]
    for i, (name, desc, color) in enumerate(derived):
        y = 2.2 - i * 0.35
        ax.text(1.0, y, f"{name}", fontsize=10, fontweight="bold", color=color, va="center")
        ax.text(2.3, y, f"{desc}", fontsize=8, color="#94a3b8", va="center")

    # Arrow to fusion box
    ax.annotate("", xy=(6.5, 5), xytext=(5.2, 5),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2.5))

    # Fusion box
    fusion_box = mpatches.FancyBboxPatch((6.5, 3.5), 3, 3, boxstyle="round,pad=0.3",
                                           facecolor="#1e293b", edgecolor=GREEN, lw=3)
    ax.add_patch(fusion_box)
    ax.text(8, 6.1, "11-BAND STACK", ha="center", fontsize=14,
            fontweight="bold", color=GREEN)
    ax.text(8, 5.5, "[B, 11, 256, 256]", ha="center", fontsize=11,
            color=AMBER, fontfamily="monospace")
    ax.text(8, 4.9, "4 Optical + 2 SAR", ha="center", fontsize=10, color=TEXT)
    ax.text(8, 4.5, "+ 5 Derived Indices", ha="center", fontsize=10, color=TEXT)
    ax.text(8, 3.9, "10m Resolution", ha="center", fontsize=10, color=CYAN)

    # Arrow to model
    ax.annotate("", xy=(10.8, 5), xytext=(9.7, 5),
                arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=2.5))

    # Model output box
    model_box = mpatches.FancyBboxPatch((10.8, 2.5), 2.8, 5, boxstyle="round,pad=0.3",
                                          facecolor="#1e293b", edgecolor=AMBER, lw=3)
    ax.add_patch(model_box)
    ax.text(12.2, 7.1, "U-Net +\nResNet-34", ha="center", fontsize=13,
            fontweight="bold", color=AMBER)
    ax.text(12.2, 6.2, "24.4M params", ha="center", fontsize=10, color="#94a3b8")

    # Output classes
    classes = [
        ("Forest", GREEN),
        ("Logging", ORANGE),
        ("Mining", RED),
        ("Agriculture", AMBER),
        ("Fire", "#dc2626"),
        ("Infrastructure", PURPLE),
    ]
    ax.text(12.2, 5.5, "6-Class Output:", ha="center", fontsize=10,
            fontweight="bold", color=TEXT)
    for i, (cls, color) in enumerate(classes):
        y = 5.0 - i * 0.4
        ax.plot(11.3, y, "s", color=color, ms=8)
        ax.text(11.6, y, cls, fontsize=9, color=color, va="center", fontweight="bold")

    save(fig, "09_band_architecture.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH 10 – DeforestNet Competitive Advantage Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def graph_competitive_advantage():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")

    ax.text(6, 9.5, "DeforestNet: Competitive Advantage Summary",
            ha="center", fontsize=18, fontweight="bold", color=GREEN)

    advantages = [
        ("ONLY 6-CLASS CAUSE IDENTIFICATION",
         "No competitor identifies WHY deforestation occurs. All others = binary forest/non-forest.",
         GREEN, "INDUSTRY FIRST"),
        ("11-BAND SAR + OPTICAL FUSION",
         "Cloud-penetrating SAR + spectral optical in unified deep learning model. Most use optical only.",
         BLUE, "TECHNICAL EDGE"),
        ("ZERO DATA COST",
         "100% free ESA Sentinel data. Competitors charge $30K-$500K/year for satellite imagery.",
         PURPLE, "COST ADVANTAGE"),
        ("10m RESOLUTION ON FREE DATA",
         "Matches commercial resolution. GFW/GLAD limited to 30m Landsat. 9x more detail per pixel.",
         CYAN, "RESOLUTION"),
        ("EUDR COMPLIANCE READY",
         "Cause-specific classification directly enables EU regulation compliance for 400K+ operators.",
         AMBER, "MARKET TIMING"),
        ("EXPLAINABLE AI (GradCAM)",
         "Visual explanations of predictions. Builds trust with regulators and auditors.",
         PINK, "TRANSPARENCY"),
    ]

    for i, (title, desc, color, badge) in enumerate(advantages):
        y = 8.2 - i * 1.35

        # Badge
        badge_box = mpatches.FancyBboxPatch((0.3, y - 0.05), 2.2, 0.45,
                                              boxstyle="round,pad=0.1",
                                              facecolor=color, alpha=0.2,
                                              edgecolor=color, lw=1.5)
        ax.add_patch(badge_box)
        ax.text(1.4, y + 0.17, badge, ha="center", fontsize=8,
                fontweight="bold", color=color)

        # Title + description
        ax.text(2.8, y + 0.2, title, fontsize=12, fontweight="bold", color=color)
        ax.text(2.8, y - 0.25, desc, fontsize=9.5, color="#94a3b8", wrap=True)

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)

    save(fig, "10_competitive_advantage.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    print("Generating professional market & competitor graphs...\n")

    graph_market_growth()
    graph_feature_heatmap()
    graph_radar_chart()
    graph_cost_comparison()
    graph_resolution_bubble()
    graph_deforestation_crisis()
    graph_classification_gap()
    graph_cloud_cover()
    graph_band_architecture()
    graph_competitive_advantage()

    print(f"\nAll graphs saved to: {OUT}")
    print(f"Total: {len(list(OUT.glob('*.png')))} PNG files")
