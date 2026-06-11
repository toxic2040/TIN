#!/usr/bin/env python3
"""Phase 3 Forensic Analysis — Visual Booklet (PDF)"""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

DATA = "runs/epyc_results/phase3"
CONFIGS = ["Exp1", "Exp2", "Exp3", "Exp6", "Cambridge"]
POLICIES = ["greedy", "no_retry", "oracle", "random"]
P_EFFS = ["0.05", "0.1", "0.3", "0.5"]
P_EFFS_F = [0.05, 0.1, 0.3, 0.5]

# ── load data ──────────────────────────────────────────────────────


def main():
    ri_data = {}
    for cfg in CONFIGS:
        with open(f"{DATA}/routing_independence_{cfg}_results.json") as f:
            ri_data[cfg] = json.load(f)

    with open(f"{DATA}/vehicular_gamma_results.json") as f:
        vg = json.load(f)

    # ── colours / style ───────────────────────────────────────────────
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    POL_COLORS = {
        "greedy": "#2196F3",
        "no_retry": "#F44336",
        "oracle": "#4CAF50",
        "random": "#FF9800",
    }
    PEFF_COLORS = {0.05: "#7B1FA2", 0.1: "#1976D2", 0.3: "#388E3C", 0.5: "#F57C00"}

    OUT = "phase3_forensic_booklet.pdf"

    with PdfPages(OUT) as pdf:
        # ════════════════════════════════════════════════════════════════
        # PAGE 1: Title + summary stats
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        title_text = (
            "PHASE 3 FORENSIC ANALYSIS\n"
            "────────────────────────────────────\n"
            "TIN Classification Theorem — Data Booklet\n"
            "7–8 March 2026  ·  192-core EPYC  ·  13.1 min\n"
        )
        ax.text(
            0.5,
            0.88,
            title_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=14,
            fontweight="bold",
            fontfamily="monospace",
        )

        summary = (
            "EXPERIMENT A: Routing Independence\n"
            "  5 CRAWDAD traces × 4 policies × 4 p_eff = 80 gamma values\n"
            "  79/80 positive (98.75%)\n"
            "  Sole outlier: Cambridge no_retry p=0.5, γ = −0.086 (R² = 0.002)\n"
            "\n"
            "EXPERIMENT B: Vehicular GPS (SF Cab)\n"
            "  200 nodes, 197,938 contacts, 24h trace\n"
            "  Classification: CLUSTER (unanimous, all 4 p_eff)\n"
            "  γ range: +0.94 to +1.00\n"
            "\n"
            "KEY FINDINGS\n"
            "  1. Cambridge sign flip is noise (R² = 0.002)\n"
            "  2. no_retry has systematic survivorship bias\n"
            "  3. Oracle preserves topological signal under heavy erasure\n"
            "  4. γ trends toward 1.0 with network size\n"
            "  5. Φ > 1 in 100% of CLUSTER configs (all policies)\n"
            "  6. η_lyap perfectly deterministic (zero seed variance)\n"
            "  7. SF Cab: Φ spans 7 orders of magnitude\n"
            "  8. Exp3: greedy beats oracle at high E[H] (multi-copy)\n"
            "  9. Self-averaging FAILS at long paths (CV grows with E[H])\n"
            " 10. Encounter rate best predicts |γ| (r = +0.91)\n"
        )
        ax.text(
            0.08,
            0.68,
            summary,
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(facecolor="#F5F5F5", edgecolor="#BDBDBD", boxstyle="round,pad=0.6"),
        )
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 2: Gamma heatmap (config × policy × p_eff)
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(11, 7))
        fig.suptitle("GAMMA VALUES: sign(γ) is routing-invariant", y=0.96)

        # Left: gamma matrix
        ax = axes[0]
        rows = []
        row_labels = []
        for cfg in CONFIGS:
            gbp = ri_data[cfg]["gamma_by_policy"]
            for pol in POLICIES:
                vals = [gbp[pol][p]["gamma"] for p in P_EFFS]
                rows.append(vals)
                row_labels.append(f"{cfg}\n{pol}")

        mat = np.array(rows)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-0.2, vmax=1.1, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"p={p}" for p in P_EFFS], fontsize=8)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)
        for i in range(len(rows)):
            for j in range(4):
                v = mat[i, j]
                color = "white" if v < 0.3 else "black"
                ax.text(
                    j,
                    i,
                    f"{v:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                    fontweight="bold",
                )
        ax.set_title("γ values", fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.6, label="gamma")

        # Horizontal lines between configs
        for i in range(1, 5):
            ax.axhline(i * 4 - 0.5, color="white", linewidth=2)

        # Right: vehicular gamma
        ax = axes[1]
        vg_gammas = [vg["gamma"][p]["gamma"] for p in P_EFFS]
        vg_npts = [vg["gamma"][p]["n_points"] for p in P_EFFS]
        bars = ax.bar(
            range(4), vg_gammas, color=[PEFF_COLORS[p] for p in P_EFFS_F], edgecolor="black"
        )
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"p={p}" for p in P_EFFS])
        ax.set_ylabel("γ")
        ax.set_ylim(0, 1.15)
        ax.set_title("SF Cab (N=200)", fontsize=10)
        for i, (g, n) in enumerate(zip(vg_gammas, vg_npts)):
            ax.text(i, g + 0.02, f"{g:.3f}\nn={n}", ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=1)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 3: R² quality heatmap
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.suptitle("FIT QUALITY: R² of ln(Φ) ~ E[H]", y=0.96)

        r2_mat = []
        r2_labels = []
        for cfg in CONFIGS:
            results = ri_data[cfg]["results"]
            for pol, phi_key in [
                ("greedy", "phi_greedy"),
                ("no_retry", "phi_noretry"),
                ("oracle", "phi_oracle"),
                ("random", "phi_random"),
            ]:
                eta_key = phi_key.replace("phi_", "eta_")
                row = []
                for p in P_EFFS_F:
                    subset = [
                        r
                        for r in results
                        if r["p_eff"] == p and r.get(eta_key, 0) > 0 and r.get(phi_key, 0) > 0
                    ]
                    if len(subset) < 3:
                        row.append(0)
                        continue
                    ehs = np.array([r["E_H"] for r in subset])
                    lp = np.log(np.array([r[phi_key] for r in subset]))
                    if np.std(lp) < 1e-10:
                        row.append(0)
                        continue
                    sl, ic = np.polyfit(ehs, lp, 1)
                    pred = sl * ehs + ic
                    ss_r = np.sum((lp - pred) ** 2)
                    ss_t = np.sum((lp - np.mean(lp)) ** 2)
                    row.append(1 - ss_r / ss_t if ss_t > 0 else 0)
                r2_mat.append(row)
                r2_labels.append(f"{cfg}\n{pol}")

        mat = np.array(r2_mat)
        im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"p={p}" for p in P_EFFS], fontsize=9)
        ax.set_yticks(range(len(r2_labels)))
        ax.set_yticklabels(r2_labels, fontsize=6)
        for i in range(len(r2_mat)):
            for j in range(4):
                v = mat[i, j]
                color = "white" if v > 0.6 else "black"
                marker = "" if v > 0.1 else " !"
                ax.text(
                    j,
                    i,
                    f"{v:.3f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                    fontweight="bold",
                )
        for i in range(1, 5):
            ax.axhline(i * 4 - 0.5, color="white", linewidth=2)
        plt.colorbar(im, ax=ax, shrink=0.6, label="R²")
        ax.set_title("R² < 0.1 marked with !  —  Cambridge/no_retry/p=0.5 = 0.002", fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 4: Survivorship bias — active fraction heatmap
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 7))
        fig.suptitle("SURVIVORSHIP BIAS: Active points / 300 by policy × p_eff", y=0.96)

        surv_mat = []
        surv_labels = []
        for cfg in CONFIGS:
            gbp = ri_data[cfg]["gamma_by_policy"]
            for pol in POLICIES:
                row = [gbp[pol][p]["n_points"] / 300.0 for p in P_EFFS]
                surv_mat.append(row)
                surv_labels.append(f"{cfg}\n{pol}")

        mat = np.array(surv_mat)
        im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"p={p}" for p in P_EFFS], fontsize=9)
        ax.set_yticks(range(len(surv_labels)))
        ax.set_yticklabels(surv_labels, fontsize=6)
        for i in range(len(surv_mat)):
            for j in range(4):
                v = mat[i, j]
                n = int(v * 300)
                color = "white" if v < 0.4 else "black"
                ax.text(
                    j,
                    i,
                    f"{n}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                )
        for i in range(1, 5):
            ax.axhline(i * 4 - 0.5, color="white", linewidth=2)
        plt.colorbar(im, ax=ax, shrink=0.6, label="fraction active")
        ax.set_title("no_retry loses 70–97% of data at p=0.05", fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 5: Gamma vs network size + encounter rate
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle("WHAT PREDICTS |γ|?", y=0.97)

        trace_info = [
            ("Exp1", 9, 2.0),
            ("Exp2", 12, 3.3),
            ("Cambridge", 36, 1.4),
            ("Exp3", 41, 7.4),
            ("Exp6", 98, 16.3),
            ("SF_Cab", 200, 41.2),
        ]

        # Oracle mean gamma
        gammas_oracle = []
        for cfg, n, cpnh in trace_info:
            if cfg == "SF_Cab":
                gs = [vg["gamma"][p]["gamma"] for p in P_EFFS]
            else:
                gs = [ri_data[cfg]["gamma_by_policy"]["oracle"][p]["gamma"] for p in P_EFFS]
            gammas_oracle.append(np.mean(gs))

        ns = [t[1] for t in trace_info]
        cpnhs = [t[2] for t in trace_info]
        names = [t[0] for t in trace_info]

        # Left: gamma vs log(N)
        ax = axes[0]
        ax.scatter(ns, gammas_oracle, s=80, c="#1976D2", zorder=5, edgecolor="black")
        for name, n, g in zip(names, ns, gammas_oracle):
            ax.annotate(f" {name}", (n, g), fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_xlabel("Network size N (log scale)")
        ax.set_ylabel("Mean γ (oracle)")
        ax.set_title(f"γ vs N    r = {np.corrcoef(np.log(ns), gammas_oracle)[0, 1]:+.3f}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylim(0.5, 1.05)
        ax.grid(alpha=0.3)

        # Right: gamma vs encounter rate
        ax = axes[1]
        ax.scatter(cpnhs, gammas_oracle, s=80, c="#F57C00", zorder=5, edgecolor="black")
        for name, c, g in zip(names, cpnhs, gammas_oracle):
            ax.annotate(f" {name}", (c, g), fontsize=8, va="center")
        ax.set_xscale("log")
        ax.set_xlabel("Contacts / node / hour (log scale)")
        ax.set_ylabel("Mean γ (oracle)")
        r_cpnh = np.corrcoef(np.log(cpnhs), gammas_oracle)[0, 1]
        ax.set_title(f"γ vs encounter rate    r = {r_cpnh:+.3f}")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylim(0.5, 1.05)
        ax.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 6: Vehicular ln(Φ) vs E[H] scatter
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("SF CAB: ln(Φ) vs E[H] — the exponential relationship", y=0.97)

        for idx, p in enumerate(P_EFFS_F):
            ax = axes[idx // 2][idx % 2]
            subset = [r for r in vg["results"] if r["p_eff"] == p and r.get("eta_greedy", 0) > 0]
            ehs = np.array([r["E_H"] for r in subset])
            log_phis = np.log(np.array([r["phi_greedy"] for r in subset]))

            ax.scatter(ehs, log_phis, s=12, alpha=0.5, c=PEFF_COLORS[p], edgecolor="none")

            # Fit line
            sl, ic = np.polyfit(ehs, log_phis, 1)
            xs = np.linspace(min(ehs), max(ehs), 100)
            ax.plot(xs, sl * xs + ic, "k--", linewidth=1.5, alpha=0.7)

            lam = np.log(p)
            gamma = sl / lam
            pred = sl * ehs + ic
            ss_r = np.sum((log_phis - pred) ** 2)
            ss_t = np.sum((log_phis - np.mean(log_phis)) ** 2)
            r2 = 1 - ss_r / ss_t

            ax.set_xlabel("E[H]")
            ax.set_ylabel("ln(Φ)")
            ax.set_title(f"p_eff = {p}    γ = {abs(gamma):.4f}    R² = {r2:.4f}", fontsize=9)
            ax.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 7: Vehicular Phi distributions (log scale)
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle("SF CAB: Φ distributions by p_eff (log₁₀ scale)", y=0.97)

        for p in P_EFFS_F:
            subset = [r for r in vg["results"] if r["p_eff"] == p and r.get("phi_greedy", 0) > 0]
            log_phis = np.log10(np.array([r["phi_greedy"] for r in subset]))
            ax.hist(
                log_phis,
                bins=30,
                alpha=0.55,
                color=PEFF_COLORS[p],
                edgecolor="black",
                linewidth=0.3,
                label=f"p={p} (n={len(subset)}, med={10 ** np.median(log_phis):.0f})",
            )

        ax.set_xlabel("log₁₀(Φ)")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title("7 orders of magnitude at p=0.05, compresses to ~1.5 at p=0.5")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 8: Oracle anomaly — R² by policy across p_eff
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 5, figsize=(11, 4.5), sharey=True)
        fig.suptitle("ORACLE ANOMALY: R² holds at high p_eff while others collapse", y=0.99)

        for ci, cfg in enumerate(CONFIGS):
            ax = axes[ci]
            results = ri_data[cfg]["results"]

            for pol, phi_key in [
                ("greedy", "phi_greedy"),
                ("no_retry", "phi_noretry"),
                ("oracle", "phi_oracle"),
                ("random", "phi_random"),
            ]:
                eta_key = phi_key.replace("phi_", "eta_")
                r2s = []
                for p in P_EFFS_F:
                    sub = [
                        r
                        for r in results
                        if r["p_eff"] == p and r.get(eta_key, 0) > 0 and r.get(phi_key, 0) > 0
                    ]
                    if len(sub) < 3:
                        r2s.append(np.nan)
                        continue
                    ehs = np.array([r["E_H"] for r in sub])
                    lp = np.log(np.array([r[phi_key] for r in sub]))
                    if np.std(lp) < 1e-10:
                        r2s.append(0)
                        continue
                    sl, ic = np.polyfit(ehs, lp, 1)
                    pred = sl * ehs + ic
                    ss_r = np.sum((lp - pred) ** 2)
                    ss_t = np.sum((lp - np.mean(lp)) ** 2)
                    r2s.append(1 - ss_r / ss_t if ss_t > 0 else 0)

                ax.plot(
                    P_EFFS_F,
                    r2s,
                    "o-",
                    color=POL_COLORS[pol],
                    label=pol,
                    markersize=5,
                    linewidth=1.5,
                )

            ax.set_title(cfg, fontsize=9)
            ax.set_xlabel("p_eff")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.1, color="red", linestyle=":", alpha=0.4)
            ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel("R²")
            if ci == 4:
                ax.legend(fontsize=6, loc="center right")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 9: Greedy vs Oracle — the Exp3 anomaly
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle("EXP3 ANOMALY: Greedy beats Oracle at high E[H]", y=0.97)

        # Left: scatter of eta_oracle vs eta_greedy at p=0.3
        ax = axes[0]
        for cfg in CONFIGS:
            results = ri_data[cfg]["results"]
            sub = [
                r
                for r in results
                if r["p_eff"] == 0.3 and r.get("eta_greedy", 0) > 0 and r.get("eta_oracle", 0) > 0
            ]
            eg = [r["eta_greedy"] for r in sub]
            eo = [r["eta_oracle"] for r in sub]
            alpha = 0.7 if cfg == "Exp3" else 0.15
            s = 20 if cfg == "Exp3" else 8
            ax.scatter(eg, eo, s=s, alpha=alpha, label=cfg, edgecolor="none")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="oracle = greedy")
        ax.set_xlabel("η_greedy")
        ax.set_ylabel("η_oracle")
        ax.set_title("p_eff = 0.3, all configs")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 0.85)
        ax.set_ylim(0, 0.85)
        ax.grid(alpha=0.3)

        # Right: oracle/greedy ratio vs E[H] for Exp3
        ax = axes[1]
        results = ri_data["Exp3"]["results"]
        for p in [0.1, 0.3, 0.5]:
            sub = [
                r
                for r in results
                if r["p_eff"] == p and r.get("eta_greedy", 0) > 0 and r.get("eta_oracle", 0) > 0
            ]
            ehs = [r["E_H"] for r in sub]
            ratios = [r["eta_oracle"] / r["eta_greedy"] for r in sub]
            ax.scatter(
                ehs, ratios, s=12, alpha=0.5, c=PEFF_COLORS[p], edgecolor="none", label=f"p={p}"
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("E[H]")
        ax.set_ylabel("η_oracle / η_greedy")
        ax.set_title("Exp3: ratio vs path length")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 2.5)
        ax.grid(alpha=0.3)
        ax.annotate("greedy wins", (3.0, 0.3), fontsize=9, color="red", fontweight="bold")
        ax.annotate("oracle wins", (3.0, 1.8), fontsize=9, color="green", fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 10: Gamma stability — CV across p_eff
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle("GAMMA STABILITY: CV(γ) across p_eff by config × policy", y=0.97)

        x_pos = 0
        x_ticks = []
        x_labels = []
        for cfg in CONFIGS:
            gbp = ri_data[cfg]["gamma_by_policy"]
            for pol in POLICIES:
                gammas = [gbp[pol][p]["gamma"] for p in P_EFFS]
                cv = np.std(gammas) / abs(np.mean(gammas)) if abs(np.mean(gammas)) > 1e-6 else 1
                color = POL_COLORS[pol]
                bar = ax.bar(x_pos, cv, color=color, edgecolor="black", linewidth=0.5, width=0.8)
                if cv > 0.5:
                    ax.text(
                        x_pos,
                        cv + 0.01,
                        f"{cv:.2f}",
                        ha="center",
                        fontsize=6,
                        color="red",
                        fontweight="bold",
                    )
                x_ticks.append(x_pos)
                x_labels.append(f"{pol[:3]}")
                x_pos += 1
            x_pos += 0.5  # gap between configs

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=6, rotation=45)
        ax.set_ylabel("CV(γ)")
        ax.axhline(0.1, color="green", linestyle="--", alpha=0.4, label="CV=0.10 (stable)")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="CV=0.50 (unstable)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

        # Config labels — place above the axes so they never overlap bars
        group_starts = [0, 4.5, 9, 13.5, 18]
        total_width = group_starts[-1] + 4  # rightmost bar position
        for gs, cfg in zip(group_starts, CONFIGS):
            x_frac = (gs + 1.5) / total_width
            fig.text(
                0.09 + x_frac * 0.82,
                0.90,
                cfg,
                ha="center",
                fontsize=9,
                fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
            )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 11: Self-averaging failure — CV(η) vs E[H]
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
        fig.suptitle("SELF-AVERAGING FAILURE: CV(η_greedy) increases with E[H]", y=0.97)

        # Left: vehicular
        ax = axes[0]
        for p in [0.1, 0.3, 0.5]:
            groups = defaultdict(list)
            ehs_u = {}
            for r in vg["results"]:
                if r["p_eff"] == p and r.get("eta_greedy", 0) > 0:
                    groups[(r["source"], r["dest"])].append(r["eta_greedy"])
                    ehs_u[(r["source"], r["dest"])] = r["E_H"]

            pair_cvs = []
            for key, etas in groups.items():
                if len(etas) > 1:
                    pair_cvs.append((ehs_u[key], np.std(etas) / np.mean(etas)))

            if pair_cvs:
                ehs_arr = [x[0] for x in pair_cvs]
                cvs_arr = [x[1] for x in pair_cvs]
                ax.scatter(
                    ehs_arr,
                    cvs_arr,
                    s=25,
                    alpha=0.6,
                    c=PEFF_COLORS[p],
                    edgecolor="none",
                    label=f"p={p}",
                )
                # Trend line
                sl, ic = np.polyfit(ehs_arr, cvs_arr, 1)
                xs = np.linspace(min(ehs_arr), max(ehs_arr), 50)
                ax.plot(xs, sl * xs + ic, color=PEFF_COLORS[p], linestyle="--", alpha=0.7)

        ax.set_xlabel("E[H]")
        ax.set_ylabel("CV(η_greedy)")
        ax.set_title("SF Cab")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Right: all CRAWDAD at p=0.3
        ax = axes[1]
        for cfg in CONFIGS:
            results = ri_data[cfg]["results"]
            groups = defaultdict(list)
            ehs_u = {}
            for r in results:
                if r["p_eff"] == 0.3 and r.get("eta_greedy", 0) > 0:
                    groups[(r["source"], r["dest"])].append(r["eta_greedy"])
                    ehs_u[(r["source"], r["dest"])] = r["E_H"]

            pair_cvs = []
            for key, etas in groups.items():
                if len(etas) > 1:
                    pair_cvs.append((ehs_u[key], np.std(etas) / np.mean(etas)))

            if pair_cvs:
                ehs_arr = [x[0] for x in pair_cvs]
                cvs_arr = [x[1] for x in pair_cvs]
                ax.scatter(ehs_arr, cvs_arr, s=20, alpha=0.5, edgecolor="none", label=cfg)

        ax.set_xlabel("E[H]")
        ax.set_ylabel("CV(η_greedy)")
        ax.set_title("CRAWDAD traces at p=0.3")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 12: Policy hierarchy — mean η by policy
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle("POLICY HIERARCHY at p_eff = 0.3", y=0.97)

        x = np.arange(len(CONFIGS))
        width = 0.2
        offsets = [-1.5, -0.5, 0.5, 1.5]

        for pi, pol in enumerate(["oracle", "greedy", "random", "no_retry"]):
            eta_key = f"eta_{pol}" if pol != "no_retry" else "eta_noretry"
            means = []
            for cfg in CONFIGS:
                results = ri_data[cfg]["results"]
                sub = [r for r in results if r["p_eff"] == 0.3]
                vals = [r[eta_key] for r in sub if r.get(eta_key, 0) > 0]
                means.append(np.mean(vals) if vals else 0)

            ax.bar(
                x + offsets[pi] * width,
                means,
                width,
                color=POL_COLORS[pol],
                edgecolor="black",
                linewidth=0.5,
                label=pol,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(CONFIGS)
        ax.set_ylabel("Mean η")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

        # Annotate Exp3
        ax.annotate(
            "greedy > oracle\n(multi-copy)",
            xy=(2, 0.46),
            fontsize=8,
            color="red",
            fontweight="bold",
            ha="center",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 13: Cross-policy Phi correlation heatmaps
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        fig.suptitle("CROSS-POLICY Φ CORRELATION (log scale, p=0.3)", y=0.97)

        phi_keys = ["phi_greedy", "phi_noretry", "phi_oracle", "phi_random"]
        pol_short = ["greedy", "no_retry", "oracle", "random"]

        for ci, cfg in enumerate(CONFIGS):
            ax = axes[ci // 3][ci % 3]
            results = ri_data[cfg]["results"]
            sub = [
                r for r in results if r["p_eff"] == 0.3 and all(r.get(pk, 0) > 0 for pk in phi_keys)
            ]

            if len(sub) < 5:
                ax.text(0.5, 0.5, "< 5 active", transform=ax.transAxes, ha="center")
                ax.set_title(cfg)
                continue

            log_phis = {pk: np.log(np.array([r[pk] for r in sub])) for pk in phi_keys}

            corr_mat = np.ones((4, 4))
            for i in range(4):
                for j in range(4):
                    corr_mat[i, j] = np.corrcoef(log_phis[phi_keys[i]], log_phis[phi_keys[j]])[0, 1]

            im = ax.imshow(corr_mat, cmap="RdYlGn", vmin=0, vmax=1)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["gr", "nr", "or", "rn"], fontsize=7)
            ax.set_yticks(range(4))
            ax.set_yticklabels(["gr", "nr", "or", "rn"], fontsize=7)
            for i in range(4):
                for j in range(4):
                    ax.text(
                        j,
                        i,
                        f"{corr_mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if corr_mat[i, j] < 0.4 else "black",
                    )
            ax.set_title(f"{cfg} (n={len(sub)})", fontsize=9)

        # Hide unused subplot
        axes[1][2].axis("off")
        axes[1][2].text(
            0.5,
            0.5,
            "greedy-oracle\ncorrelation > 0.90\nin all configs\n\n"
            "no_retry-random\ncorrelation as low\nas 0.08 (Exp1)\nand 0.12 (Cambridge)",
            transform=axes[1][2].transAxes,
            ha="center",
            va="center",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(facecolor="#FFF9C4", edgecolor="#F9A825"),
        )

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 14: Vehicular — Φ vs E[H] by source node
        # ════════════════════════════════════════════════════════════════
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.suptitle("SF CAB: Φ vs E[H] coloured by source node (p=0.3)", y=0.97)

        sub = [r for r in vg["results"] if r["p_eff"] == 0.3 and r.get("phi_greedy", 0) > 0]

        sources = sorted(set(r["source"] for r in sub))
        src_colors = plt.cm.tab10(np.linspace(0, 1, len(sources)))

        for si, src in enumerate(sources):
            pts = [r for r in sub if r["source"] == src]
            ehs = [r["E_H"] for r in pts]
            log_phis = [np.log10(r["phi_greedy"]) for r in pts]
            ax.scatter(
                ehs,
                log_phis,
                s=25,
                c=[src_colors[si]],
                alpha=0.6,
                edgecolor="none",
                label=src.replace("new_", ""),
            )

        ax.set_xlabel("E[H]")
        ax.set_ylabel("log₁₀(Φ)")
        ax.legend(fontsize=7, ncol=2, title="source node")
        ax.grid(alpha=0.3)
        ax.set_title("Each source node traces a clean exponential — no outlier sources")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ════════════════════════════════════════════════════════════════
        # PAGE 15: no_retry collapse — gamma vs p_eff by config
        # ════════════════════════════════════════════════════════════════
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("no_retry COLLAPSE vs oracle STABILITY", y=0.97)

        for panel, pol in enumerate(["no_retry", "oracle"]):
            ax = axes[panel]
            for cfg in CONFIGS:
                gbp = ri_data[cfg]["gamma_by_policy"]
                gammas = [gbp[pol][p]["gamma"] for p in P_EFFS]
                ax.plot(P_EFFS_F, gammas, "o-", label=cfg, markersize=5)

            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("p_eff")
            ax.set_ylabel("γ")
            ax.set_title(pol, fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.2, 1.15)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Booklet written to {OUT}")
    print("  15 pages, all Phase 3 forensic findings visualized")


if __name__ == "__main__":
    main()
