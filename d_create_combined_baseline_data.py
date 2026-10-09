import json
import os
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfTransformer

from program_params import (
    BASELINE_COMBINED_METADATA_FILE,
    BASELINE_COMBINED_WORD_DATA_FILE,
    BASELINE_COMPARISON_COUNTRIES,
    BASELINE_DATA_FOLDER,
    BASELINE_TF_IDF_TRANSFORMER_FILE,
    BASELINE_VOCABULARY_FILE,
    COUNTRY_WORD_FREQUENCY_DATA_FOLDER,
    META_COLS,
    MIN_DOC_FREQ,
)


def prune_and_load(file: Path) -> pd.DataFrame:
    print(f"Pruning dataset {file}")
    df = pd.read_parquet(file)
    word_cols = df.columns.difference(META_COLS)

    doc_freq = (df[word_cols] > 0).sum(axis=0)
    keep_cols = doc_freq[doc_freq >= MIN_DOC_FREQ].index

    df = df[list(keep_cols) + META_COLS]

    print(
        f"{file.name}: {df.shape[1] - len(META_COLS)} words kept (of {len(word_cols)})"
    )
    return df


def apply_tf_idf_transformation(dataframe: pd.DataFrame):
    tf_idf_transformer = TfidfTransformer()

    sparse_df = dataframe.astype(pd.SparseDtype("float64", 0))
    sparse_matrix = sparse_df.sparse.to_coo().tocsr()

    transformed_data = tf_idf_transformer.fit_transform(sparse_matrix)

    tf_idf_transformer_path = Path(
        BASELINE_DATA_FOLDER, BASELINE_TF_IDF_TRANSFORMER_FILE
    )
    joblib.dump(tf_idf_transformer, tf_idf_transformer_path)
    print(f"Saved tf idf transformer to {tf_idf_transformer_path}")

    return transformed_data


def create_combined_baseline_data():
    print(
        f"Beginning to create baseline data based on countries: {', '.join(BASELINE_COMPARISON_COUNTRIES)}"
    )

    os.makedirs(BASELINE_DATA_FOLDER, exist_ok=True)

    country_word_data_path = Path(COUNTRY_WORD_FREQUENCY_DATA_FOLDER)
    country_files = sorted(country_word_data_path.glob("*_word_counts.parquet"))

    def find_country_code(country_file_name):
        return country_file_name.split("_")[0]

    baseline_country_files = [
        country_file
        for country_file in country_files
        if find_country_code(country_file.name) in BASELINE_COMPARISON_COUNTRIES
    ]

    baseline_country_dfs = []
    for file in baseline_country_files:
        print(f"Loading {file.name}")
        country_df = prune_and_load(file)
        baseline_country_dfs.append(country_df)

    print("Combining baseline country data frames...")
    combined_baseline_country_df = pd.concat(baseline_country_dfs, axis=0)
    combined_baseline_country_df = combined_baseline_country_df.reset_index(drop=True)

    word_cols = combined_baseline_country_df.columns.difference(META_COLS)
    # Fill in missing vocabulary words as 0s
    print("Filling in NaN values...")
    combined_baseline_country_df[word_cols] = (
        combined_baseline_country_df[word_cols].fillna(0).astype(int)
    )

    combined_baseline_country_df_metadata = combined_baseline_country_df[META_COLS]
    combined_baseline_country_word_data = combined_baseline_country_df[word_cols]

    transformed_word_data = apply_tf_idf_transformation(
        combined_baseline_country_word_data
    )

    word_data_path = Path(BASELINE_DATA_FOLDER, BASELINE_COMBINED_WORD_DATA_FILE)
    save_npz(
        word_data_path,
        transformed_word_data,
    )
    print(f"Saved baseline tf-idf transformed word data to {word_data_path}")

    meta_data_path = Path(BASELINE_DATA_FOLDER, BASELINE_COMBINED_METADATA_FILE)
    combined_baseline_country_df_metadata.to_parquet(meta_data_path)
    print(f"Saved baseline metadata to {meta_data_path}")

    vocabulary_path = Path(BASELINE_DATA_FOLDER, BASELINE_VOCABULARY_FILE)
    with open(vocabulary_path, "w", encoding="utf-8") as f:
        json.dump(word_cols.tolist(), f, ensure_ascii=False, indent=2)
        print(f"Saved baseline data vocabulary to {vocabulary_path}")

    print("Finished creating baseline combined data")


if __name__ == "__main__":
    create_combined_baseline_data()
