import os
import random
from collections import defaultdict
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords

from program_params import (
    CHARACTERS_TO_REMOVE,
    CORPUS_DOCUMENTS_FOLDER,
    COUNTRY_CODE_COLUMN_TITLE,
    COUNTRY_DOCUMENT_CAP,
    COUNTRY_DOCUMENT_NO_CAP_FLAG,
    COUNTRY_WORD_FREQUENCY_DATA_FOLDER,
    RANDOM_SEED,
    REMOVE_STOP_WORDS,
)


def find_word_counts(doc_text: str):
    # Basic word cleaning + normalization
    doc_text = doc_text.strip().lower()

    for character in CHARACTERS_TO_REMOVE:
        doc_text = doc_text.replace(character, "")

    doc_words = doc_text.split()

    if REMOVE_STOP_WORDS:
        stop_words = set(stopwords.words("english"))
        doc_words = [word for word in doc_words if word not in stop_words]

    freq_dict = defaultdict(int)
    for word in doc_words:
        freq_dict[word] += 1

    return freq_dict


def process_country(country_code: str, country_dir: Path, out_file_path: Path):
    print(f"Beginning to process {country_code}")

    rows = []
    doc_ids = []
    documents_processed = 0

    files = sorted(country_dir.glob("*.txt"))
    if COUNTRY_DOCUMENT_CAP != COUNTRY_DOCUMENT_NO_CAP_FLAG:
        print(
            f"COUNTRY_DOCUMENT_CAP set to {COUNTRY_DOCUMENT_CAP}, randomly selecting {COUNTRY_DOCUMENT_CAP} files with seed {RANDOM_SEED}"
        )
        random.seed(RANDOM_SEED)
        random.shuffle(files)
        files = files[:COUNTRY_DOCUMENT_CAP]

    for documents_processed, file in enumerate(files):
        text = file.read_text(encoding="utf-8", errors="ignore")
        rows.append(find_word_counts(text))
        doc_ids.append(f"{country_code}_{file.stem}")

        if (documents_processed + 1) % 100 == 0:
            print(
                f"Processed {documents_processed + 1} documents for country {country_code}"
            )

    df = pd.DataFrame(rows, index=doc_ids)
    df = df.fillna(0).astype(int)
    df[COUNTRY_CODE_COLUMN_TITLE] = country_code
    print("Exporting data frame..")
    df.to_parquet(
        out_file_path
    )  # export as parquet file becuase of how sparse + large the dfs are
    print(
        f"Finished processing {country_code} documents, word frequency data output to {out_file_path}"
    )


def collect_word_frequencies():
    # Make directory to output frequency data files
    print("Collecting country word frequencies...")

    print(
        f"Outputing country word frequecy data to: {COUNTRY_WORD_FREQUENCY_DATA_FOLDER}"
    )
    os.makedirs(COUNTRY_WORD_FREQUENCY_DATA_FOLDER, exist_ok=True)

    if REMOVE_STOP_WORDS:
        nltk.download("stopwords")
        nltk.download("punkt")

    corpus_dir = Path(CORPUS_DOCUMENTS_FOLDER)
    for country_dir in corpus_dir.iterdir():
        if not country_dir.is_dir():
            continue
        country_code = country_dir.name

        out_file_name = f"{country_code}_word_counts.parquet"
        out_file_path = Path(COUNTRY_WORD_FREQUENCY_DATA_FOLDER, out_file_name)
        if out_file_path.exists():
            print(f"Skipping {country_code}, already processed")
            continue

        process_country(country_code, country_dir, out_file_path)

    print("Finished collecting country word frequencies")


if __name__ == "__main__":
    collect_word_frequencies()
