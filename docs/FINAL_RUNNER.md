# Running the Frozen Final Evaluation

The runner implements `config/final_evaluation.json` without changing the original
protocol, model, gate, calibration, costs, or policy helpers. Its additional code
fingerprint is `reports/protocols/aura-final-runner-v1.json`, which also binds the
original protocol lock. Commit this implementation and fingerprint before scoring.

## Safe Preflight

From the AURA root with the existing environment active:

```bash
python scripts/evaluate_final_routing.py --check-only
```

Running without flags also performs only preflight. It verifies the original
frozen inputs, runner code, and fitted gates' scikit-learn version. It does not
open the test dataset, load model weights into memory, or deserialize the gates.
The integrity checks do read model files as bytes to compute their checksums.

## Explicit Final Run

```bash
python scripts/evaluate_final_routing.py --run-final
```

Use `--device cpu` if needed; `auto` uses the existing device selection logic.
Record any known earlier test exposure with `--test-exposure-note "..."`. The
default honestly records that historical exposure was not independently verified.
The runner does not prove that the test was never inspected in earlier work.
There are no dataset, coverage, scenario, or bootstrap overrides on this command.

The final dataset is checked for the official 3,080-row, 77-intent structure,
40 messages per intent, canonical labels, and original row IDs. Duplicate texts
are permitted and audited, not silently removed. All nine frozen policies and
two guard matched-count controls are evaluated at five coverages under three
scenarios: 165 comparison rows.

Models are only loaded and used for inference. No `fit` call or threshold search
occurs. Scores and acceptance masks are saved before labeled metrics are computed.
The primary paired bootstrap is cost-aware lexical gate minus plain signed gate,
at 80% coverage under moderate costs, with the configured 10,000 replicates and
seed 42. Its point estimate must reconcile with the comparison table. Secondary
tables are descriptive; no additional secondary intervals are computed.

## Outputs

The fixed output directory is `artifacts/final_evaluation/aura-final-evaluation-v1/`:

- `run_state.json`: started/completed/failed status and whether test access began.
- `protocol.json`, both locks, and `environment.json`: reproducibility metadata.
- `predictions.npz`: row IDs, class order, logits, raw/calibrated probabilities.
- `scores.csv`, `decisions.csv`: label-free policy scores and acceptance masks.
- `labeled_predictions.csv`, `comparison.csv`: labeled outcomes and all metrics.
- `primary_result.json`, `primary_interval.csv`: the predeclared primary result.
- `overlap_audit.csv`, `overlap_summary.json`: normalized training-pool overlaps
  and duplicate-test flags, with no primary exclusions.
- `summary.json`: test checksum, input-lock hashes, output hashes, and limitations.

`summary.json` is written only on success. Existing output directories, including
failed partial runs, are never overwritten. A failure preserves its error and
exposure status. Do not delete a failed run to hide prior test access: document
the problem and version any corrective rerun as required by the protocol.

The test/training checksums and frozen-input checks are repeated before completion.
These checks detect file drift, not adversarial tampering. Artifacts remain local
and Git-ignored. Research cost units, offline ranking, and conditional bootstrap
intervals are not a deployment safety certificate or measured banking losses.

## Implementation Verification

Tests use synthetic messages and stub inference. They exercise label-free
selection, guard shortfalls, null metrics, primary interpretation, overlap flags,
write-once behavior, and a full synthetic output workflow. They do not open the
real test dataset. The existing bootstrap tests cover the resampling calculation;
the end-to-end runner test stubs that calculation while checking its fixed settings.

```bash
python -m pytest tests/unit/evaluation/test_final_routing.py tests/unit/evaluation/test_final_runner.py -q
```
