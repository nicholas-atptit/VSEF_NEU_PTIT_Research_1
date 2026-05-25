# VN30 Daily 2015 Target60 Next-Step Decision

- Created at UTC: 2026-05-17.
- Track: VN30 daily 2015 benchmark.
- Branch: `research/vn100-evidence-hardening-v1`.
- Evidence source: `reports/generated/vn30_daily_2015_target60_postmortem/daily_target60_failure_postmortem.md`.

## Decision

Recommended option: **A. Stop daily tuning for now and lock daily result as failed 60.**

This does not prevent using the daily track as robustness context later, but the daily target60 track should not run a broad v3 sweep unless a new validation-only diagnostic identifies a specific fix before final scoring.

## Options Considered

| Option | Decision | Reason |
|---|---|---|
| A. Stop daily tuning for now and lock daily result as failed 60. | Recommended | V2 widened the search but did not improve over the h=40 daily record. The remaining gap is 2.42pp to 60, and no validation-only fix is clear. |
| B. Run daily v3 focused only if diagnostics show a clear fix. | Not recommended now | Diagnostics show drag and mismatch, but not a narrow pre-registered fix. A v3 would risk becoming another broad tuning sweep. |
| C. Change daily target definition, but label it as a separate target. | Not recommended for this track | A target change would answer a different question and must be labeled separately. It should not be used to rescue the current target60 claim. |
| D. Use daily track only as robustness context for the hourly available-window result. | Secondary use | This is acceptable as a claim boundary, but the immediate daily-track decision is still to stop tuning and record failed 60. |

## Basis

- Canonical best daily result remains LightGBM `daily_cross` h=40 with 57.58% final accuracy on 8,880 rows.
- Best v2 final result was LightGBM `volatility_normalized` h=50 at 57.45%, which is 0.13pp below h=40.
- The h=40 result is 2.42pp below the 60% target.
- The h=40 final positive rate is 56.25%, and the model is only 1.33pp above the final-period majority-class diagnostic baseline.
- Worst ticker drag is concentrated in VIC, VJC, VIB, ACB, BVH, VPB, GAS, and SSB.
- Worst monthly drag is concentrated in 2025-09, 2025-10, 2025-11, 2025-03, 2026-01, and 2025-02.
- Worst quarterly drag is 2025-Q4, followed by 2025-Q1.
- Validation-final mismatch is material: v2 stability-best final accuracy was 53.92%, while v2 final-best was 57.45%; stability score Spearman correlation with final accuracy was -0.132.

## Claim Boundary

- Daily target60 passed: no.
- Current daily claim: best recorded daily final accuracy is 57.58%, below 60%.
- Daily v3 recommended now: no.
- Daily-only track: yes.
- Hourly data used: no.
- Daily-to-hourly resampling used: no.
- Hourly claims made from daily data: no.
- Paper or DOCX generated: no.
- Trading, profitability, or live-deployment claim: no.
