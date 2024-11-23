import os
import re
from unidecode import unidecode

VALID_NAME_CHARS = ["&", "_", "-", "(", ")", ","]


# Utility functions for retrieving paths
def msd_id_to_dirs(msd_id):
    """Given an MSD ID, generate the path prefix.
    E.g. TRABCD12345678 -> A/B/C/TRABCD12345678"""
    return os.path.join(msd_id[2], msd_id[3], msd_id[4], msd_id)


def msd_id_to_h5(h5_path, msd_id):
    """Given an MSD ID, return the path to the corresponding h5"""
    return os.path.join(h5_path, msd_id_to_dirs(msd_id) + ".h5")


def get_midi_path(lmd_matched_path, msd_id, midi_md5):
    """Given an MSD ID and MIDI MD5, return path to a MIDI file."""
    return os.path.join(lmd_matched_path, msd_id_to_dirs(msd_id), midi_md5 + ".mid")


def retain_best_midi(scores):
    best_midi_files = {}
    for msd_id, midi_scores in scores.items():
        # Find the MIDI file with the highest score
        best_midi_md5 = max(midi_scores, key=midi_scores.get)
        best_midi_files[msd_id] = best_midi_md5
    return best_midi_files


def delete_non_best_midis(scores, best_midi_files, lmd_matched_path):
    for msd_id, midi_scores in scores.items():
        for midi_md5 in midi_scores:
            if midi_md5 != best_midi_files[msd_id]:
                midi_path = get_midi_path(lmd_matched_path, msd_id, midi_md5)
                if os.path.exists(midi_path):
                    os.remove(midi_path)
                else:
                    print("Error - path {} doesn't exist".format(midi_path))


def clean_name(name):
    unidecoded = unidecode(name)
    cleaned_name = "".join(
        c if c.isalnum() or c.isspace() or c in VALID_NAME_CHARS else " "
        for c in unidecoded
    )
    cleaned_name = re.sub(r"\s+", " ", cleaned_name).strip()

    # Truncate if the cleaned name is too long
    max_length = 64  # Limit the cleaned name length to 64 characters
    if len(cleaned_name) > max_length:
        cleaned_name = cleaned_name[:max_length]

    return cleaned_name.strip()
