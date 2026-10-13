import json
import os
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from d_create_combined_baseline_data import prune_and_load
from e_analyze_baseline_data import (
    get_combined_baseline_metadata,
    get_combined_baseline_word_data,
)
from program_params import (
    ADDITIONAL_COMPARISON_GRAPH_FILE_AFFIX,
    ADDITIONAL_COUNTRIES_TO_COMPARE,
    ANALYSIS_OUTPUT_FOLDER,
    BASELINE_DATA_FOLDER,
    BASELINE_TF_IDF_TRANSFORMER_FILE,
    BASELINE_VOCABULARY_FILE,
    COUNTRY_CODE_COLUMN_TITLE,
    COUNTRY_WORD_FREQUENCY_DATA_FOLDER,
    DOCUMENT_COMPARISON_ITERATIONS,
    META_COLS,
    RANDOM_SEED,
)


def analyze_additional_data():
    print("Beginning to analyze baseline data...")
    baseline_word_data = get_combined_baseline_word_data()
    baseline_meta_data = get_combined_baseline_metadata()
    baseline_transformer = get_combined_baseline_transformer()
    baseline_vocabulary = get_combined_baseline_vocabulary()

    def transform_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        # convert additional country vocab to that of the baseline df
        vocab_transformed_dataframe = dataframe.reindex(
            columns=baseline_vocabulary,
            fill_value=0,
        )

        sparse_df = vocab_transformed_dataframe.astype(pd.SparseDtype("float64", 0))
        sparse_matrix = sparse_df.sparse.to_coo().tocsr()

        tf_idf_transformed_dataframe = baseline_transformer.transform(sparse_matrix)
        return tf_idf_transformed_dataframe

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

    country_word_data_path = Path(COUNTRY_WORD_FREQUENCY_DATA_FOLDER)
    country_files = sorted(country_word_data_path.glob("*_word_counts.parquet"))

    def find_country_code(country_file_name):
        return country_file_name.split("_")[0]

    additional_country_files = [
        country_file
        for country_file in country_files
        if find_country_code(country_file.name) in ADDITIONAL_COUNTRIES_TO_COMPARE
    ]

    additional_country_dfs = []
    for file in additional_country_files:
        print(f"Loading {file.name}")
        country_df = prune_and_load(file)
        additional_country_dfs.append(country_df)

    random.seed(RANDOM_SEED)
    for country_df in additional_country_dfs:
        sample_results = []

        comparison_country = country_df.iloc[0][COUNTRY_CODE_COLUMN_TITLE]
        word_cols = country_df.columns.difference(META_COLS)
        tf_idf_transformed_word_data = transform_dataframe(country_df[word_cols])
        tf_idf_transformed_num_documents = tf_idf_transformed_word_data.shape[0]

        for i in range(DOCUMENT_COMPARISON_ITERATIONS):
            baseline_index = random.randrange(baseline_data_num_documents)
            comparison_index = random.randrange(tf_idf_transformed_num_documents)

            distance = euclidean_distances(
                baseline_word_data[baseline_index],
                tf_idf_transformed_word_data[comparison_index],
            )[0, 0]

            baseline_country = baseline_meta_data.iloc[baseline_index][
                COUNTRY_CODE_COLUMN_TITLE
            ]

            register = "_".join([str(baseline_country), str(comparison_country)])

            sample_results.append([register, distance])

        results_df = pd.DataFrame(
            sample_results,
            columns=["Register", "Distance"],
        )

        results_df.sort_values(
            ["Register"],
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
            hue="Register",
            jitter=True,
            size=3,
        )
        plt.xticks(rotation=45)
        ax.set_ylim([-3, 3])
        plt.tight_layout()

        os.makedirs(ANALYSIS_OUTPUT_FOLDER, exist_ok=True)
        output_path = Path(
            ANALYSIS_OUTPUT_FOLDER,
            str(comparison_country) + ADDITIONAL_COMPARISON_GRAPH_FILE_AFFIX,
        )
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Saved comparison plot to {output_path}")
    print("Finished analyzing comparison countries data...")


def get_combined_baseline_vocabulary():
    baseline_vocabulary_path = Path(BASELINE_DATA_FOLDER, BASELINE_VOCABULARY_FILE)

    with open(baseline_vocabulary_path, encoding="utf-8") as f:
        baseline_vocabulary = json.load(f)
    return baseline_vocabulary


def get_combined_baseline_transformer():
    baseline_transformer_path = Path(
        BASELINE_DATA_FOLDER, BASELINE_TF_IDF_TRANSFORMER_FILE
    )
    return joblib.load(baseline_transformer_path)


if __name__ == "__main__":
    analyze_additional_data()
