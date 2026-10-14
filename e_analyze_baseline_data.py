import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from program_params import (
    ANALYSIS_OUTPUT_FOLDER,
    BASELINE_COMBINED_METADATA_FILE,
    BASELINE_COMBINED_WORD_DATA_FILE,
    BASELINE_COMPARISON_GRAPH_FILE,
    BASELINE_DATA_FOLDER,
    BASELINE_SIMILARITY_FIGURES_FILE,
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

    similarities_dict = {"Same": [], "Different": [], "Total": []}

    for i in range(DOCUMENT_COMPARISON_ITERATIONS):
        index_1, index_2 = random.sample(range(baseline_data_num_documents), 2)

        similarity = cosine_similarity(
            baseline_word_data[index_1],
            baseline_word_data[index_2],
        )[0, 0]

        country1 = baseline_meta_data.iloc[index_1][COUNTRY_CODE_COLUMN_TITLE]
        country2 = baseline_meta_data.iloc[index_2][COUNTRY_CODE_COLUMN_TITLE]

        register = "_".join(sorted([str(country1), str(country2)]))
        status = "Same" if country1 == country2 else "Different"
        similarities_dict[status].append(similarity)
        similarities_dict["Total"].append(similarity)

        sample_results.append([register, similarity, status])

    output_similarity_data(similarities_dict, BASELINE_SIMILARITY_FIGURES_FILE)

    results_df = pd.DataFrame(
        sample_results,
        columns=["Register", "Similarity", "Status"],
    )

    results_df.sort_values(
        ["Status", "Similarity"],
        inplace=True,
    )

    figure(figsize=(10, 5))
    sns.stripplot(
        data=results_df,
        x="Register",
        y="Similarity",
        hue="Status",
        jitter=True,
        size=3,
    )
    plt.xticks(rotation=45)
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


def output_similarity_data(similarities_dict: dict, figures_file_name: str):
    output = f"{'Register':<20} {'T1 Score':>10} {'Std Dev':>10}"
    output += f"\n{'-' * 42}"

    for k, v in similarities_dict.items():
        mean = np.mean(v)
        std = np.std(v)
        output += f"\n{k:<20} {mean:>10.4f} {std:>10.4f}"

    output_path = Path(ANALYSIS_OUTPUT_FOLDER, figures_file_name)
    with open(output_path, "w") as f:
        f.write(output)
    print(f"Output similarity data figures to {output_path}")


if __name__ == "__main__":
    analyze_baseline_data()
