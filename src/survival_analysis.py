"""
survival_analysis.py
Kaplan-Meier and Cox Proportional Hazards survival analysis for GBM radiomic habitats.
Author: Sylvester KT (@SylvesterKT)
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts

logger = logging.getLogger(__name__)


class SurvivalAnalyzer:
    """
    Survival analysis pipeline for GBM radiomic habitat clusters.

    Performs Kaplan-Meier estimation, log-rank tests, and Cox PH regression
    to assess survival differences across radiomic habitat groups.

    Args:
        duration_col:  Column name for survival time (days/months)
        event_col:     Column name for event indicator (1=death, 0=censored)
        alpha:         Significance level for statistical tests
    """

    def __init__(
        self,
        duration_col: str = "survival_days",
        event_col: str = "vital_status",
        alpha: float = 0.05,
    ):
        self.duration_col = duration_col
        self.event_col = event_col
        self.alpha = alpha
        self.kmf_fitters: Dict[str, KaplanMeierFitter] = {}
        self.cox_fitter: Optional[CoxPHFitter] = None

    def fit_kaplan_meier(
        self,
        df: pd.DataFrame,
        group_col: str = "habitat",
    ) -> Dict[str, KaplanMeierFitter]:
        """
        Fit Kaplan-Meier survival curves per habitat group.

        Args:
            df:        DataFrame with survival data and habitat labels
            group_col: Column containing group labels (e.g. habitat cluster IDs)

        Returns:
            Dict mapping group label -> fitted KaplanMeierFitter
        """
        groups = df[group_col].unique()
        self.kmf_fitters = {}

        for group in sorted(groups):
            mask = df[group_col] == group
            kmf = KaplanMeierFitter(label=f"Habitat-{group}")
            kmf.fit(
                durations=df.loc[mask, self.duration_col],
                event_observed=df.loc[mask, self.event_col],
            )
            self.kmf_fitters[str(group)] = kmf
            median_os = kmf.median_survival_time_
            logger.info(f"Habitat-{group}: n={mask.sum()}, median OS={median_os:.1f}")

        return self.kmf_fitters

    def logrank_test(
        self,
        df: pd.DataFrame,
        group_col: str = "habitat",
    ) -> pd.DataFrame:
        """
        Pairwise log-rank tests between all habitat groups.

        Returns:
            pd.DataFrame with columns: group_a, group_b, test_statistic, p_value, significant
        """
        groups = sorted(df[group_col].unique())
        results = []

        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                mask1 = df[group_col] == g1
                mask2 = df[group_col] == g2

                result = logrank_test(
                    durations_A=df.loc[mask1, self.duration_col],
                    durations_B=df.loc[mask2, self.duration_col],
                    event_observed_A=df.loc[mask1, self.event_col],
                    event_observed_B=df.loc[mask2, self.event_col],
                )
                results.append({
                    "group_a": f"Habitat-{g1}",
                    "group_b": f"Habitat-{g2}",
                    "test_statistic": round(result.test_statistic, 4),
                    "p_value": round(result.p_value, 4),
                    "significant": result.p_value < self.alpha,
                })

        results_df = pd.DataFrame(results)
        n_sig = results_df["significant"].sum()
        logger.info(f"Log-rank tests: {n_sig}/{len(results_df)} pairs significant (p<{self.alpha})")
        return results_df

    def fit_cox(
        self,
        df: pd.DataFrame,
        covariates: List[str],
        penalizer: float = 0.1,
    ) -> CoxPHFitter:
        """
        Fit a Cox Proportional Hazards model.

        Args:
            df:          DataFrame with survival data and covariate columns
            covariates:  List of covariate column names (e.g. radiomic features + habitat)
            penalizer:   L2 regularisation strength

        Returns:
            Fitted CoxPHFitter
        """
        cox_df = df[[self.duration_col, self.event_col] + covariates].dropna()
        self.cox_fitter = CoxPHFitter(penalizer=penalizer)
        self.cox_fitter.fit(
            cox_df,
            duration_col=self.duration_col,
            event_col=self.event_col,
        )
        self.cox_fitter.print_summary()
        logger.info(f"Cox PH model fitted | Concordance: {self.cox_fitter.concordance_index_:.3f}")
        return self.cox_fitter

    def plot_km_curves(
        self,
        save_path: Optional[str] = None,
        title: str = "Kaplan-Meier Survival Curves by Radiomic Habitat",
    ) -> plt.Figure:
        """
        Plot Kaplan-Meier survival curves for all habitat groups.

        Args:
            save_path: Optional path to save figure
            title:     Plot title

        Returns:
            matplotlib Figure
        """
        if not self.kmf_fitters:
            raise RuntimeError("Run fit_kaplan_meier() first.")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.kmf_fitters)))

        for (label, kmf), color in zip(self.kmf_fitters.items(), colors):
            kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

        ax.set_xlabel("Time (days)", fontsize=12)
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"KM plot saved to {save_path}")

        return fig

    def plot_cox_forest(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Forest plot of Cox PH hazard ratios.

        Returns:
            matplotlib Figure
        """
        if self.cox_fitter is None:
            raise RuntimeError("Run fit_cox() first.")

        fig, ax = plt.subplots(figsize=(10, 8))
        self.cox_fitter.plot(ax=ax)
        ax.set_title("Cox PH Hazard Ratios (95% CI)", fontsize=13, fontweight="bold")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Forest plot saved to {save_path}")

        return fig

    def summary(
        self,
        df: pd.DataFrame,
        group_col: str = "habitat",
    ) -> pd.DataFrame:
        """
        Generate a per-habitat survival summary table.

        Returns:
            pd.DataFrame with n, events, median OS, and 95% CI per habitat
        """
        rows = []
        for label, kmf in self.kmf_fitters.items():
            mask = df[group_col] == int(label)
            rows.append({
                "Habitat": f"Habitat-{label}",
                "N": int(mask.sum()),
                "Events": int(df.loc[mask, self.event_col].sum()),
                "Median OS (days)": round(kmf.median_survival_time_, 1),
                "95% CI Lower": round(kmf.confidence_interval_["KM_estimate_lower_0.95"].iloc[0], 1),
                "95% CI Upper": round(kmf.confidence_interval_["KM_estimate_upper_0.95"].iloc[0], 1),
            })
        return pd.DataFrame(rows)
