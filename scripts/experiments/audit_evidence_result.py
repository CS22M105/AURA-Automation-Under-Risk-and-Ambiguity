"""Post-hoc Monte Carlo and rare-case robustness audit; never read official test."""

import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from aura.data.manifests import calculate_sha256
from aura.evaluation.evidence_bootstrap import bootstrap_policy_contrasts
from aura.policies.coverage import select_at_coverage


def main() -> None:
    """Repeat the entire 18-contrast family once with a larger fixed resample count."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/exploratory/evidence_routing_v2")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/exploratory/evidence_routing_robustness_v1")
    )
    parser.add_argument("--resamples", type=int, default=100_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Robustness audits must not overwrite earlier results")
    summary_path = args.source / "summary.json"
    summary = json.loads(summary_path.read_text())
    for name in ["scores.csv", "comparison.csv", "crossfit.csv"]:
        if calculate_sha256(args.source / name) != summary["output_hashes"][name]:
            raise ValueError(f"Research artifact changed: {name}")
    audit_path = Path("artifacts/audits/learned_cost_data_v1/row_support.csv")
    if calculate_sha256(audit_path) != summary["input_hashes"]["audited_rows"]:
        raise ValueError("Audited cost data changed")
    saved = pd.read_csv(args.source / "scores.csv").set_index("row_id")
    rows = pd.read_csv(audit_path).set_index("row_id").loc[saved.index]
    if set(rows.partition) != {"policy_validation"} or rows.overlaps_model_training.any():
        raise ValueError("Expected the declared filtered validation population")
    scores = {name: saved[name].to_numpy() for name in saved.columns}
    candidates = [
        "signed_tilted_cost",
        "retrieval_gap",
        "evidence_gate",
        "evidence_severity_gate",
        "evidence_tilted_cost",
        "pooled_posterior_risk",
    ]
    contrasts = [
        (candidate, reference)
        for candidate in candidates
        for reference in ["confidence", "signed_gate", "signed_cost_gate"]
    ]
    intervals = bootstrap_policy_contrasts(
        scores,
        rows.moderate_cost.to_numpy(),
        saved.index.tolist(),
        contrasts,
        resamples=args.resamples,
        family_size=len(contrasts),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    intervals.to_csv(args.output / "intervals.csv", index=False)
    sensitivity = []
    # Remove one maximum-cost example at a time and rerank on the remaining rows.
    high = rows.index[rows.moderate_cost == rows.moderate_cost.max()]
    for omitted in high:
        keep = saved.index != omitted
        ids = saved.index.to_numpy()[keep]
        costs = rows.moderate_cost.to_numpy()[keep]
        candidate = select_at_coverage(scores["evidence_severity_gate"][keep], ids, 0.8)
        for reference in ["confidence", "signed_gate", "signed_cost_gate"]:
            baseline = select_at_coverage(scores[reference][keep], ids, 0.8)
            sensitivity.append(
                {
                    "omitted_row_id": int(omitted),
                    "reference": reference,
                    "candidate_cost": float(costs[candidate].sum()),
                    "reference_cost": float(costs[baseline].sum()),
                    "difference": float(costs[candidate].sum() - costs[baseline].sum()),
                    "accepted_count": int(candidate.sum()),
                    "evaluation_count": int(keep.sum()),
                }
            )
    pd.DataFrame(sensitivity).to_csv(args.output / "leave_one_high_cost_out.csv", index=False)
    metadata = {
        "stage": "post_hoc_exploratory_robustness",
        "official_test_used": False,
        "source_summary_sha256": calculate_sha256(summary_path),
        "resamples": args.resamples,
        "seed": 42,
        "contrasts": len(contrasts),
        "reason": "Check Monte Carlo stability of a near-zero adjusted upper endpoint; no retuning",
        "implementation_sha256": calculate_sha256(
            Path("src/aura/evaluation/evidence_bootstrap.py")
        ),
        "script_sha256": calculate_sha256(Path(__file__)),
        "output_hashes": {
            p.name: calculate_sha256(p) for p in args.output.iterdir() if p.is_file()
        },
    }
    (args.output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    reported = intervals[intervals.candidate == "evidence_severity_gate"].copy()
    reported[["difference", "lower", "upper"]] *= 1000
    print(reported.to_string(index=False))
    print(pd.DataFrame(sensitivity).groupby("reference").difference.agg(["min", "max"]).to_string())
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
