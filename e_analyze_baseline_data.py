import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from program_params import (
    ANALYSIS_OUTPUT_FOLDER,
    BASELINE_COMBINED_METADATA_FILE,
    BASELINE_COMBINED_WORD_DATA_FILE,
    BASELINE_COMPARISON_GRAPH_FILE,
    BASELINE_DATA_FOLDER,
    COUNTRY_CODE_COLUMN_TITLE,
    DOCUMENT_COMPARISON_ITERATIONS,
    RANDOM_SEED,
)


def analyze_baseline_data():
    print("Beginning to analyze baseline data...")
    baseline_word_data = get_combined_baseline_word_data()
    baseline_meta_data = get_combined_baseline_metadata()

    baseline_data_num_documents = baseline_word_data.shape[0]
    if baseline_data_num_documents != len(baseline_meta_data):
        raise ValueError(
            "The number of TF-IDF vectors does not match the number of metadata rows."
        )

    if COUNTRY_CODE_COLUMN_TITLE not in baseline_meta_data.columns:
        raise ValueError(
            f"Country column '{COUNTRY_CODE_COLUMN_TITLE}' not found. "
            f"Available columns: {list(baseline_meta_data.columns)}"
        )

    random.seed(RANDOM_SEED)
    sample_results = []

    for i in range(DOCUMENT_COMPARISON_ITERATIONS):
        index_1, index_2 = random.sample(range(baseline_data_num_documents), 2)

        distance = euclidean_distances(
            baseline_word_data[index_1],
            baseline_word_data[index_2],
        )[0, 0]

        country1 = baseline_meta_data.iloc[index_1][COUNTRY_CODE_COLUMN_TITLE]
        country2 = baseline_meta_data.iloc[index_2][COUNTRY_CODE_COLUMN_TITLE]

        register = "_".join(sorted([str(country1), str(country2)]))
        status = "Same" if country1 == country2 else "Different"

        sample_results.append([register, distance, status])

    results_df = pd.DataFrame(
        sample_results,
        columns=["Register", "Distance", "Status"],
    )

    results_df.sort_values(
        ["Status", "Register"],
        inplace=True,
    )

    scaler = StandardScaler()
    results_df.loc[:, "Distance"] = scaler.fit_transform(
        results_df[["Distance"]]
    ).ravel()

    figure(figsize=(10, 5))
    ax = sns.stripplot(
        data=results_df,
        x="Register",
        y="Distance",
        hue="Status",
        jitter=True,
        size=3,
    )
    plt.xticks(rotation=45)
    ax.set_ylim([-3, 3])
    plt.tight_layout()

    os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)
    output_path = Path(ANALYSIS_OUTPUT_FOLDER, BASELINE_COMPARISON_GRAPH_FILE)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to {output_path}")
    print("Finsihed analyzing baseline data...")


def get_combined_baseline_word_data():
    baseline_word_data_path = Path(
        BASELINE_DATA_FOLDER,
        BASELINE_COMBINED_WORD_DATA_FILE,
    )
    return load_npz(baseline_word_data_path)


def get_combined_baseline_metadata():
    baseline_metadata_path = Path(
        BASELINE_DATA_FOLDER,
        BASELINE_COMBINED_METADATA_FILE,
    )
    return pd.read_parquet(baseline_metadata_path)


if __name__ == "__main__":
    analyze_baseline_data()
