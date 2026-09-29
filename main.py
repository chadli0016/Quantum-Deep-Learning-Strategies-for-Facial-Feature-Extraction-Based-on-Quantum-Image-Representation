from image_processing import (
    face_crop,
    QIP,
    feature_extraction,
    FaceDatasetBuilder,
    FeatureDatasetBuilder,
)
import os
import cv2
import numpy as np
import pandas as pd
from machine_learning_second import train_nn_classifier, run_experiment
import matplotlib.pyplot as plt

# Ask the user for the folder containing the face dataset.
# Remove surrounding spaces and any quotation marks copied with the path.
ROOT_DIR = input("Enter the path to your face dataset: ").strip().strip('"')


builder = FaceDatasetBuilder(ROOT_DIR)
df = builder.build()

selected_features = ["ALEXnet", "HOG", "QFSD", "quantum_inspired"]

try:
    feature_builder = FeatureDatasetBuilder(
        face_df=df,
        selected_features=selected_features
    )

    feature_df = feature_builder.build()
    scores, pca_dim = run_experiment(feature_df)

    row = {
        "combination_id": 1,
        "features_used": " + ".join(selected_features),
        "num_methods": len(selected_features),
        "num_samples": len(feature_df),
        "pca_components": pca_dim,
        "SVM_accuracy": scores.get("SVM_accuracy", np.nan),
        "Logistic_accuracy": scores.get("Logistic_accuracy", np.nan),
        "NN_accuracy": scores.get("NN_accuracy", np.nan),
        "XGBoost_accuracy": scores.get("XGBoost_accuracy", np.nan)
    }

except Exception as e:
    print(f"FAILED COMBINATION {selected_features}: {e}")

    row = {
        "combination_id": 1,
        "features_used": " + ".join(selected_features),
        "num_methods": len(selected_features),
        "num_samples": 0,
        "pca_components": 0,
        "SVM_accuracy": np.nan,
        "Logistic_accuracy": np.nan,
        "NN_accuracy": np.nan,
        "XGBoost_accuracy": np.nan,
        "error": str(e)
    }


# =========================================================
# FINAL RESULTS DATASET
# =========================================================

results_df = pd.DataFrame([row])

accuracy_cols = [
    "SVM_accuracy",
    "Logistic_accuracy",
    "NN_accuracy",
    "XGBoost_accuracy"
]

existing_accuracy_cols = [
    col for col in accuracy_cols if col in results_df.columns
]

results_df["best_accuracy"] = results_df[existing_accuracy_cols].max(axis=1)
results_df["best_classifier"] = results_df[existing_accuracy_cols].idxmax(axis=1)

results_df = results_df.sort_values(
    by="best_accuracy",
    ascending=False
)




results_df.to_csv(
    "feature_combination_classifier_results.csv",
    index=False
)



# Save a bar plot of the classifier accuracies.
plot_scores = results_df.iloc[0][accuracy_cols].dropna().astype(float)

if not plot_scores.empty:
    labels = [name.replace("_accuracy", "") for name in plot_scores.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, plot_scores.values, color="steelblue")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classifier Accuracy: " + " + ".join(selected_features))
    ax.set_ylim(0, max(100, plot_scores.max() * 1.1))
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    fig.tight_layout()
    fig.savefig("classifier_accuracy_barplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved as: classifier_accuracy_barplot.png")
else:
    print("No classifier accuracies available to plot.")


print("\nDONE")

# Main entry point for the face recognition experiment.
