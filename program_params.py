# Random Seed
RANDOM_SEED = 1

# Corpus Extraction Params
CORPUS_DOCUMENTS_FOLDER = "Country_Corpus"
GLOWBE_DATA_FOLDER_PATH = "GlowBe_Data"

# Word Frequncy Extraction Params
COUNTRY_DOCUMENT_NO_CAP_FLAG = -1
COUNTRY_DOCUMENT_CAP = 2500  # Set to COUNTRY_DOCUMENT_NO_CAP_FLAG to remove cap
COUNTRY_CODE_COLUMN_TITLE = "Country Code"
COUNTRY_WORD_FREQUENCY_DATA_FOLDER = "Country_Word_Frequency_Data"

REMOVE_STOP_WORDS = True
CHARACTERS_TO_REMOVE = [",", ";", ".", "?", "!", '"']

# Prune And Combine Country Word Data Params
MIN_DOC_FREQ = 5  # How many docs within a country's corpus a word must appear in for us to not prune it
META_COLS = [COUNTRY_CODE_COLUMN_TITLE]
COUNTRY_DATA_CAP_INCLUDE_ALL_FLAG = -1
