"""Reusable academic framing for VN30 comprehensive model-universe outputs."""

from __future__ import annotations

PAPER_TITLE = (
    "Comparing Machine Learning Models for VN30 Equity Directional Forecasting: "
    "Evidence from Walk-Forward Testing and Vietnamese Market Index Benchmarks"
)

RESEARCH_GAP_PARAGRAPHS = [
    (
        "Existing financial forecasting studies show that machine learning can be applied to equity "
        "prediction, but the resulting evidence is difficult to compare when studies differ in target "
        "definition, model scope, forecast horizon, validation design, and reporting granularity. A result "
        "from one algorithm, one index, one forecast horizon, or one final testing window does not establish "
        "general model-family behavior in stock-level VN30 directional forecasting. This paper therefore "
        "treats comparability, validation discipline, and stock-level diagnostic granularity as the central "
        "evidence gap rather than treating a single high final-window score as sufficient evidence."
    ),
    (
        "The gap matters because narrow evidence can overstate model superiority. A model family may look "
        "strong at h40 but weaker at h20, h60, or h80; ensemble or cooperation mechanisms may improve one "
        "diagnostic slice while failing to transfer consistently; and index-level results can hide "
        "stock-level heterogeneity. The paper is therefore positioned as an evidence-based benchmark analysis "
        "with bounded interpretation, not as a leaderboard, trading system, profitability study, investment "
        "recommendation, or live-deployment claim."
    ),
]

BENCHMARK_RESPONSE_PARAGRAPHS = [
    (
        "This paper addresses that gap with a controlled comprehensive VN30 stock-level model-universe "
        "benchmark across h20, h40, h60, and h80. The benchmark preserves full 30-stock headline coverage, "
        "uses a common directional target and horizon structure, selects models and thresholds from "
        "validation evidence only, and treats the final window as scoring-only. The model universe includes "
        "naive reference rules, technical reference rules, linear and generalized linear models, SVM and "
        "kernel models, KNN and distance-based models, probabilistic classifiers, tree-based models, boosting models including "
        "CatBoost, neural and deep models, ensembles and stacking, calibration variants, regime-aware "
        "models, and statistical direction models; GARCH is reported only as a diagnostic, not as a headline "
        "direction classifier. Headline interpretation excludes ticker subsets, confidence abstention, and "
        "top-k ranking substitutions. Where Vietnamese market index evidence appears, it is used as "
        "market-context evidence only and does not replace stock-level VN30 results."
    ),
]

CONTRIBUTION_PARAGRAPHS = [
    (
        "The first contribution is an evidence contribution: broad stock-level VN30 model-universe evidence "
        "under a common directional target, horizon structure, and diagnostic framework. The second "
        "contribution is methodological: validation-only selection, final scoring-only evaluation, explicit "
        "claim boundaries, and model-family comparison rather than leaderboard selection. The third "
        "contribution is diagnostic: horizon dependence, validation-final transfer, model-family "
        "heterogeneity, and regime, cooperation, and KNN-support diagnostics. These contributions support "
        "bounded interpretation and do not establish trading readiness."
    )
]

MAIN_CLAIM_BOUNDARY_BULLETS = [
    "- The bounded h40 result is Logistic L2 using the C-closest reference feature set, with validation-selected threshold 0.55 and final accuracy 61.63% under full 30-stock coverage.",
    "- The bull_bear_sideway_router h40 fixed 0.50 final accuracy 63.33% row is descriptive final-window context only and is not claim-eligible.",
    "- The soft-voting final accuracy 62.00% cooperation row is descriptive context only and is not claim-eligible.",
    "- Rows with high validation and poor final transfer, including stacking_xgboost_meta diagnostics, are interpreted as validation-final transfer or overfit failures rather than as main results.",
    "- KNN-support rows are diagnostic support experiments and do not replace the main claim.",
    "- GARCH is a volatility diagnostic only and not a direct headline direction classifier.",
    "- Market-index evidence is market-context evidence and cannot substitute for stock-level VN30 evidence.",
    "- No ticker subset, confidence abstention, or top-k/ranking substitute is used for headline accuracy.",
    "- No trading readiness, profitability, investment advice, final65, live deployment, or generalization beyond the reported VN30 evidence is claimed.",
]

FAIR_TUNING_ROLE_PARAGRAPHS = [
    (
        "The fair tuning run supports the comprehensive paper as a controlled model-family diagnostic under "
        "the common h40 directional target and shared feature-family framework. It broadens model coverage "
        "and reports validation-final transfer, but it does not create an independent route for final-window "
        "claim selection."
    ),
    (
        "Rows that exceed the current h40 result only on the final-window leaderboard are descriptive because "
        "the final window is scoring-only. High-validation stacking_xgboost_meta rows are interpreted as "
        "validation-final transfer failures when their final accuracy deteriorates."
    ),
]

COOPERATION_ROLE_PARAGRAPHS = [
    (
        "The cooperation audit evaluates soft voting, model-as-feature, error-correction, mixture, "
        "calibration, and feature-selection cooperation mechanisms as diagnostics of transfer behavior. "
        "These mechanisms help assess whether cooperation improves robust evidence or merely shifts final "
        "window rankings."
    ),
    (
        "The soft-voting row with final accuracy near 62.00% is descriptive context only because it is not the "
        "validation-selected h40 paper result. Model-as-feature rows with high validation accuracy and poor "
        "final transfer are discussed as overfit or transfer-failure evidence, not as headline results."
    ),
]

KNN_SUPPORT_ROLE_PARAGRAPHS = [
    (
        "The KNN-support experiment is a diagnostic extension of the model universe. It tests whether "
        "similarity-based signals can support other model families under the same validation-only and "
        "final-scoring-only boundary."
    ),
    (
        "KNN-support evidence is not a replacement for the fixed h40 paper claim. It is used to diagnose "
        "whether distance-based information improves transfer and stock-level robustness without changing "
        "the headline VN30 claim boundary."
    ),
]

FULL_HORIZON_ROLE_PARAGRAPHS = [
    (
        "The full-horizon diagnostics show why the evidence gap cannot be resolved with a single forecast "
        "horizon. Reporting h20, h40, h60, and h80 under common row rules makes horizon dependence visible "
        "while preserving the separate h40 paper claim."
    ),
    (
        "The horizon tables are diagnostic model-family evidence, not a horizon-selection mechanism. They "
        "support bounded interpretation by showing where validation-selected behavior transfers across "
        "horizons and where it weakens."
    ),
]


def section_lines(title: str, paragraphs: list[str], heading_level: str = "##") -> list[str]:
    """Return a Markdown prose section with blank lines between paragraphs."""
    lines = [f"{heading_level} {title}", ""]
    for paragraph in paragraphs:
        lines.extend([paragraph, ""])
    return lines[:-1]


def research_gap_lines(heading_level: str = "##") -> list[str]:
    return section_lines("Research Gap and Evidence Positioning", RESEARCH_GAP_PARAGRAPHS, heading_level)


def benchmark_response_lines(heading_level: str = "##") -> list[str]:
    return section_lines("Benchmark Response", BENCHMARK_RESPONSE_PARAGRAPHS, heading_level)


def contribution_lines(heading_level: str = "##") -> list[str]:
    return section_lines("Contribution Framing", CONTRIBUTION_PARAGRAPHS, heading_level)


def main_claim_boundary_lines(heading_level: str = "##") -> list[str]:
    return [f"{heading_level} Claim Boundary for Paper Interpretation", "", *MAIN_CLAIM_BOUNDARY_BULLETS]


def academic_role_lines(title: str, paragraphs: list[str], heading_level: str = "##") -> list[str]:
    return section_lines(title, paragraphs, heading_level)
