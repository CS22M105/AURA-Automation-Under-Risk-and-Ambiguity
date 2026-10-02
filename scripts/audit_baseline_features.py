"""Generate a feature and input-data audit for the linear baseline."""

import csv
import json
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from aura.data.loading import load_banking77_csv
from aura.data.manifests import (
    calculate_sha256,
    load_development_split_manifest,
    validate_manifest_source,
)
from aura.data.validation import BANKING77_TRAIN_PROFILE, validate_dataset_profile
from aura.evaluation.feature_audit import audit_baseline_features
from aura.models.artifacts import load_model_artifact


def build_parser() -> ArgumentParser:
    """Create the command-line argument parser."""
    parser = ArgumentParser(description="Audit baseline inputs and TF-IDF features.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/train.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/splits/development_seed_42.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/models/tfidf_logistic_regression.joblib"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/audits/tfidf_logistic_regression"),
    )
    parser.add_argument("--top-features-per-intent", type=int, default=15)
    return parser


def write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write dictionaries to a CSV file with stable columns."""
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Audit the training partition and fitted baseline representation."""
    args = build_parser().parse_args()
    dataset_path: Path = args.dataset
    manifest_path: Path = args.manifest
    model_path: Path = args.model
    output_directory: Path = args.output_directory
    top_features_per_intent: int = args.top_features_per_intent

    frame = load_banking77_csv(dataset_path)
    validate_dataset_profile(frame, BANKING77_TRAIN_PROFILE)
    manifest = load_development_split_manifest(manifest_path)
    validate_manifest_source(manifest, dataset_path, len(frame))
    model_artifact = load_model_artifact(model_path)

    manifest_hash = calculate_sha256(manifest_path)
    if model_artifact.metadata.get("dataset_sha256") != manifest.source_sha256:
        raise ValueError("Model artifact was trained from a different dataset")
    if model_artifact.metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Model artifact was trained from a different split manifest")

    vectorizer = model_artifact.pipeline.named_steps["tfidf"]
    classifier = model_artifact.pipeline.named_steps["classifier"]
    if not isinstance(vectorizer, TfidfVectorizer):
        raise TypeError("Baseline pipeline does not contain a TF-IDF vectorizer")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Baseline pipeline does not contain Logistic Regression")

    training_frame = frame.loc[list(manifest.model_training)]
    intent_names = {
        int(label): str(intent)
        for label, intent in frame[["label", "label_text"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    audit = audit_baseline_features(
        training_frame["text"].tolist(),
        training_frame["label"].tolist(),
        intent_names,
        vectorizer,
        classifier,
        top_features_per_intent=top_features_per_intent,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_directory / "summary.json"
    per_intent_path = output_directory / "per_intent_text.csv"
    top_features_path = output_directory / "top_features.csv"

    write_csv(
        per_intent_path,
        [
            "label",
            "intent",
            "sample_count",
            "mean_word_count",
            "median_word_count",
            "percentile_95_word_count",
            "mean_character_count",
        ],
        [profile.to_dict() for profile in audit.per_intent],
    )
    write_csv(
        top_features_path,
        [
            "label",
            "intent",
            "rank",
            "feature",
            "coefficient",
            "document_frequency",
            "inverse_document_frequency",
        ],
        [feature.to_dict() for feature in audit.top_features],
    )

    summary: dict[str, object] = {
        "schema_version": 1,
        "partition": "model_training",
        "model_type": model_artifact.metadata.get("model_type"),
        "input_variables": ["text"],
        "derived_representation": "word_tfidf_unigrams_and_bigrams",
        "overall_text": {
            "sample_count": audit.overall_text.sample_count,
            "word_count": audit.overall_text.word_count.to_dict(),
            "character_count": audit.overall_text.character_count.to_dict(),
        },
        "class_balance": asdict(audit.class_balance),
        "vocabulary": {
            **asdict(audit.vocabulary),
            "document_frequency": audit.vocabulary.document_frequency.to_dict(),
            "inverse_document_frequency": (audit.vocabulary.inverse_document_frequency.to_dict()),
        },
        "outputs": {
            "per_intent_text_file": per_intent_path.name,
            "top_features_file": top_features_path.name,
            "top_features_per_intent": top_features_per_intent,
        },
        "provenance": {
            "dataset_sha256": manifest.source_sha256,
            "manifest_sha256": manifest_hash,
            "model_sha256": calculate_sha256(model_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Training messages: {audit.overall_text.sample_count}")
    print(f"Classes: {audit.class_balance.class_count}")
    print(f"Features: {audit.vocabulary.feature_count}")
    print(f"Unigrams: {audit.vocabulary.unigram_count}")
    print(f"Bigrams: {audit.vocabulary.bigram_count}")
    print(f"Mean words per message: {audit.overall_text.word_count.mean:.2f}")
    print(f"Audit: {summary_path}")


if __name__ == "__main__":
    main()
