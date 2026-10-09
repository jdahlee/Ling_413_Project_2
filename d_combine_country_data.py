from pathlib import Path

import pandas as pd

from program_params import (
    COMBINED_WORD_DATA_FILE,
    COUNTRY_WORD_FREQUENCY_DATA_FOLDER,
    INTIAL_COUNTRIES_TO_COMPARE,
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


def combine_country_data():
    print(
        f"Beginning to combine country data for countries: {', '.join(INTIAL_COUNTRIES_TO_COMPARE)}"
    )

    data_path = Path(COUNTRY_WORD_FREQUENCY_DATA_FOLDER)
    output_path = Path(COMBINED_WORD_DATA_FILE)
    if output_path.exists():
        print(f"{output_path} already exists, skipping combine")
        return

    country_files = sorted(data_path.glob("*_word_counts.parquet"))

    def find_country_code(country_file_name):
        return country_file_name.split("_")[0]

    selected_country_files = [
        country_file
        for country_file in country_files
        if find_country_code(country_file.name) in INTIAL_COUNTRIES_TO_COMPARE
    ]

    dfs = []
    for file in selected_country_files:
        print(f"Loading {file.name}")
        df = prune_and_load(file)
        dfs.append(df)

    print("Combining country DataFrames...")
    combined = pd.concat(dfs, axis=0)

    word_cols = combined.columns.difference(META_COLS)
    # Fill in missing vocabulary words as 0s
    print("Filling in NaN values...")
    combined[word_cols] = combined[word_cols].fillna(0).astype(int)

    combined.to_parquet(output_path)
    print(f"Saved combined data to {output_path}")

    print("Finished combining country data")


if __name__ == "__main__":
    combine_country_data()
