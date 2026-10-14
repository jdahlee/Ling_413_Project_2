import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
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
    PSEUDO_DOC_SIZE,
    RANDOM_SEED,
    SUBLINEAR_TF,
)


def aggregate_documents(df: pd.DataFrame, word_cols: list, size: int) -> pd.DataFrame:
    n_groups = len(df) // size

    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(df))[: n_groups * size]

    counts = (
        df[word_cols]
        .to_numpy()[order]
        .reshape(n_groups, size, len(word_cols))
        .sum(axis=1)
    )
    pseudo_df = pd.DataFrame(counts, columns=word_cols)
    meta = df[META_COLS].iloc[order[::size]].reset_index(drop=True)

    print(
        f"Merged {len(df)} documents into {n_groups} pseudo-documents of {size} "
        f"({len(df) - n_groups * size} left over and dropped)"
    )
    return pd.concat([pseudo_df, meta], axis=1)


def prune_and_load(file: Path) -> pd.DataFrame:
    print(f"Pruning dataset {file}")
    df = pd.read_parquet(file)
    word_cols = df.columns.difference(META_COLS)

    doc_freq = (df[word_cols] > 0).sum(axis=0)
    keep_cols = list(doc_freq[doc_freq >= MIN_DOC_FREQ].index)

    df = df[keep_cols + META_COLS]

    print(f"{file.name}: {len(keep_cols)} words kept (of {len(word_cols)})")

    if PSEUDO_DOC_SIZE > 1:
        df = aggregate_documents(df, keep_cols, PSEUDO_DOC_SIZE)
    return df


def build_counts():
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
        baseline_country_dfs.append(prune_and_load(file))

    print("Combining baseline country data frames...")
    combined_baseline_country_df = pd.concat(baseline_country_dfs, axis=0).reset_index(
        drop=True
    )

    word_cols = combined_baseline_country_df.columns.difference(META_COLS)

    # Fill in missing vocabulary words as 0s
    print("Filling in NaN values...")
    combined_baseline_country_df[word_cols] = (
        combined_baseline_country_df[word_cols].fillna(0).astype(int)
    )

    metadata = combined_baseline_country_df[META_COLS]
    counts = csr_matrix(combined_baseline_country_df[word_cols].to_numpy())
    return counts, metadata, list(word_cols)


def apply_tf_idf_transformation(dataframe: pd.DataFrame):
    tf_idf_transformer = TfidfTransformer(sublinear_tf=SUBLINEAR_TF)

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

    counts, metadata, vocabulary = build_counts()
    transformer = TfidfTransformer(sublinear_tf=SUBLINEAR_TF)
    tfidf = transformer.fit_transform(counts).tocsr()

    transformer_path = Path(BASELINE_DATA_FOLDER, BASELINE_TF_IDF_TRANSFORMER_FILE)
    joblib.dump(transformer, transformer_path)
    print(f"Saved tf idf transformer to {transformer_path}")

    word_data_path = Path(BASELINE_DATA_FOLDER, BASELINE_COMBINED_WORD_DATA_FILE)
    save_npz(word_data_path, tfidf)
    print(f"Saved baseline tf-idf transformed word data to {word_data_path}")

    meta_data_path = Path(BASELINE_DATA_FOLDER, BASELINE_COMBINED_METADATA_FILE)
    metadata.to_parquet(meta_data_path)
    print(f"Saved baseline metadata to {meta_data_path}")

    vocabulary_path = Path(BASELINE_DATA_FOLDER, BASELINE_VOCABULARY_FILE)
    with open(vocabulary_path, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
        print(f"Saved baseline data vocabulary to {vocabulary_path}")

    print("Finished creating baseline combined data")
    return tfidf, metadata


if __name__ == "__main__":
    create_combined_baseline_data()
