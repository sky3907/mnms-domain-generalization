from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/content/mnms-domain-generalization")

INPUT = ROOT / "prompt_comparison/patient_level_results.csv"
OUTPUT = ROOT / "prompt_comparison/metric_audit"

METRICS = [
    "Dice_LV",
    "Dice_Myocardium",
    "Dice_RV",
    "HD95_LV",
    "HD95_Myocardium",
    "HD95_RV",
]

MODELS = [
    "source_only",
    "generic_prompt",
    "latent_domain_prompt",
]

DOMAINS = ["in-domain", "unseen"]

EXPECTED_COUNTS = {
    "in-domain": 28,
    "unseen": 40,
}

def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    required = {
        "model", "patient", "phase",
        "vendor", "domain", "Mean_Dice", "Mean_HD95",
        *METRICS,
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(df) != 204:
        raise ValueError(f"Expected 204 records, found {len(df)}")

    if df.duplicated(["model", "patient", "phase"]).any():
        raise ValueError("Duplicate patient-phase-model rows.")

    if set(df["model"]) != set(MODELS):
        raise ValueError("Unexpected model names.")

    if set(df["domain"]) != set(DOMAINS):
        raise ValueError("Unexpected domain names.")

    if df[["model", "patient", "phase", "vendor", "domain"]].isna().any().any():
        raise ValueError("Missing identifiers.")

    if not (
        ((df["vendor"].isin(["A", "B"])) & (df["domain"] == "in-domain"))
        | ((df["vendor"].isin(["C", "D"])) & (df["domain"] == "unseen"))
    ).all():
        raise ValueError("Vendor/domain mismatch.")

    # Each patient-phase must appear once for each of the three models.
    groups = df.groupby(["patient", "phase"])["model"].agg(
        lambda x: set(x)
    )
    if len(groups) != 68 or not groups.apply(lambda x: x == set(MODELS)).all():
        raise ValueError("Model pairing is incomplete.")

    rows = []
    summary_rows = []
    issues = []

    for model in MODELS:
        for domain in DOMAINS:
            group = df[
                (df["model"] == model)
                & (df["domain"] == domain)
            ]

            expected = EXPECTED_COUNTS[domain]

            if len(group) != expected:
                raise ValueError(
                    f"{model}/{domain}: expected {expected}, got {len(group)}"
                )

            metric_means = {}

            for metric in METRICS + ["Mean_Dice", "Mean_HD95"]:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(
                    dtype=float
                )

                finite = np.isfinite(values)
                nan_count = int(np.isnan(values).sum())
                inf_count = int(np.isinf(values).sum())

                finite_values = values[finite]
                mean_value = (
                    float(finite_values.mean())
                    if len(finite_values)
                    else np.nan
                )

                metric_means[metric] = mean_value

                rows.append({
                    "model": model,
                    "domain": domain,
                    "metric": metric,
                    "total_volumes": len(values),
                    "valid_count": int(finite.sum()),
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "mean_of_finite_values": mean_value,
                    "minimum": float(finite_values.min()) if len(finite_values) else np.nan,
                    "maximum": float(finite_values.max()) if len(finite_values) else np.nan,
                })

                if not finite.all():
                    for record, valid in zip(group.to_dict("records"), finite):
                        if not valid:
                            issues.append({
                                "model": model,
                                "domain": domain,
                                "patient": record["patient"],
                                "phase": record["phase"],
                                "vendor": record["vendor"],
                                "metric": metric,
                                "value": record[metric],
                            })

            # Reproduce the manuscript-level summaries from saved Mean_ columns.
            summary_rows.append({
                "model": model,
                "domain": domain,
                "N_patient_phase_volumes": len(group),
                "Mean_Dice": metric_means["Mean_Dice"],
                "Mean_HD95_mm": metric_means["Mean_HD95"],
            })

    counts = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    problems = pd.DataFrame(
        issues,
        columns=["model", "domain", "patient", "phase", "vendor", "metric", "value"]
    )

    counts.to_csv(OUTPUT / "metric_validity_counts.csv", index=False)
    summary.to_csv(OUTPUT / "reproduced_domain_summary.csv", index=False)
    problems.to_csv(OUTPUT / "nonfinite_metric_records.csv", index=False)

    print("\n=== METRIC VALIDITY COUNTS ===")
    print(
        counts[
            ["model", "domain", "metric", "total_volumes",
             "valid_count", "nan_count", "inf_count"]
        ].to_string(index=False)
    )

    print("\n=== REPRODUCED DOMAIN SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== NON-FINITE METRIC RECORDS ===")
    if problems.empty:
        print("None in the saved patient-level results.")
    else:
        print(problems.to_string(index=False))

    print("\nIMPORTANT:")
    print(
        "This audit examines saved CSV values only. "
        "It cannot independently verify which ground-truth masks "
        "were empty, whether HD95 exceptions occurred, or whether "
        "the evaluator omitted an undefined class metric."
    )

    print("\nSaved audit files:", OUTPUT)
    print("No models trained. No inference performed.")


if __name__ == "__main__":
    main()
