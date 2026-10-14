from enum import Flag, auto

from a_create_corpus import extract_glowbe_documents
from b_organize_corpus import organize_corpus_documents
from c_collect_frequencies import collect_word_frequencies
from d_create_combined_baseline_data import create_combined_baseline_data
from e_analyze_baseline_data import analyze_baseline_data
from f_analyze_additional_data import analyze_additional_data


# Program Steps Param
class Step(Flag):
    EXTRACT_CORPUS = auto()
    ORGANIZE_CORPUS = auto()
    COLLECT_WORD_FREQUENCIES = auto()
    PRUNE_AND_COMBINE = auto()
    ANALYZE_BASELINE = auto()
    ANALYZE_ADDITIONAL = auto()
    ALL = (
        EXTRACT_CORPUS
        | ORGANIZE_CORPUS
        | COLLECT_WORD_FREQUENCIES
        | PRUNE_AND_COMBINE
        | ANALYZE_BASELINE
        | ANALYZE_ADDITIONAL
    )


# Adjust this to the steps that you want to run
STEPS_TO_RUN = Step.PRUNE_AND_COMBINE | Step.ANALYZE_BASELINE | Step.ANALYZE_ADDITIONAL


def main():
    if Step.EXTRACT_CORPUS in STEPS_TO_RUN:
        extract_glowbe_documents()
    if Step.ORGANIZE_CORPUS in STEPS_TO_RUN:
        organize_corpus_documents()
    if Step.COLLECT_WORD_FREQUENCIES in STEPS_TO_RUN:
        collect_word_frequencies()
    if Step.PRUNE_AND_COMBINE in STEPS_TO_RUN:
        create_combined_baseline_data()
    if Step.ANALYZE_BASELINE in STEPS_TO_RUN:
        analyze_baseline_data()
    if Step.ANALYZE_ADDITIONAL in STEPS_TO_RUN:
        analyze_additional_data()


if __name__ == "__main__":
    main()
