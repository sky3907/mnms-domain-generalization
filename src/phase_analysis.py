import pandas as pd

# -------------------------------------------------------
# Load Results
# -------------------------------------------------------

CSV_PATH = "/content/patient_level_results.csv"

df = pd.read_csv(CSV_PATH)

print("\nLoaded", len(df), "patient-phase evaluations")

# -------------------------------------------------------
# Group by ED / ES
# -------------------------------------------------------

summary = (
    df.groupby("phase")
      .agg({

          "Dice_LV": "mean",
          "Dice_Myocardium": "mean",
          "Dice_RV": "mean",

          "HD95_LV": "mean",
          "HD95_Myocardium": "mean",
          "HD95_RV": "mean",

      })
)

summary["Mean_Dice"] = (

    summary[
        [
            "Dice_LV",
            "Dice_Myocardium",
            "Dice_RV"
        ]
    ].mean(axis=1)

)

summary["Mean_HD95"] = (

    summary[
        [
            "HD95_LV",
            "HD95_Myocardium",
            "HD95_RV"
        ]
    ].mean(axis=1)

)

summary = summary.round(4)

print("\n==============================")
print("ED vs ES Results")
print("==============================")

print(summary)

summary.to_csv(
    "phase_summary.csv"
)

print(
    "\nSaved:",
    "phase_summary.csv"
)