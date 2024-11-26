import os

DATA_PATH = r"path\to\data\root\folder"

LMD_MATCHED_PATH = os.path.join(DATA_PATH, "lmd_matched")
H5_PATH = os.path.join(DATA_PATH, "lmd_matched_h5")
READY_MIDI_PATH = os.path.join(DATA_PATH, "ready_midi")

SCORES_FILE = os.path.join(DATA_PATH, "match_scores.json")
GENRE_MAP_FILE = os.path.join(DATA_PATH, "genre_mapping.json")
METADATA_FILE = os.path.join(DATA_PATH, "metadata.json")
