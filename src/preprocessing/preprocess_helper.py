import os
import json
import hashlib
import numpy as np
import shutil

from .genre import SpotifyHandler
from .utils import clean_name, get_midi_path


class Checkpoint_helper:
    def __init__(self, checkpoint_file: str) -> None:
        self.checkpoint_file = checkpoint_file

    def save_checkpoint(self, i):
        """Save the current index to a checkpoint file."""
        with open(self.checkpoint_file, "w") as f:
            f.write(str(i))

    def load_checkpoint(self):
        """Load the last processed index from the checkpoint file."""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as f:
                return int(f.read().strip())
        return 0

    def remove_checkpoint_file(self):
        os.remove(self.checkpoint_file)


def generate_nice_dataset(
    metadata,
    spotify_handler: SpotifyHandler,
    genre_mapping,
    msd_to_midi,
    data_path: str,
    ready_midi_path: str,
    lmd_matched_path: str,
):
    ch = Checkpoint_helper(os.path.join(data_path, "checkpoint_updater.txt"))
    last_updated = ch.load_checkpoint()
    metadata_len = len(metadata)

    for i, song_info in enumerate(metadata, start=1):
        if i <= last_updated:
            continue

        msd_id = song_info["msd_id"]
        try:
            artist_name = clean_name(song_info["artist_name"])
            genre, subgenre = spotify_handler.estimate_artist_genre(
                artist_name, genre_mapping
            )

            song_info["genre"] = clean_name(genre)
            song_info["subgenre"] = clean_name(subgenre)

            song_dir = os.path.join(
                ready_midi_path,
                song_info["genre"],
                song_info["subgenre"],
                song_info["artist_name"],
                song_info["title"],
            )
            os.makedirs(song_dir, exist_ok=True)

            old_midi_path = get_midi_path(lmd_matched_path, msd_id, msd_to_midi[msd_id])
            song_info_path = os.path.join(song_dir, msd_id + ".json")

            midi_new_path = os.path.join(song_dir, msd_id + ".mid")

            # Save metadata to file after each successful update
            with open(song_info_path, "w", encoding="utf-8") as f:
                json.dump(song_info, f, indent=4)

            shutil.move(old_midi_path, midi_new_path)

            ch.save_checkpoint(i)
            print(
                f"{i}/{metadata_len} | Metadata and MIDI file saved for {song_info["title"]}, artist: {song_info["artist_name"]}"
            )
        except Exception as e:
            print(
                f"Error saving metadata or MIDI file for i:{i}, msd_id: {msd_id}, title: {song_info["title"]}, artist: {song_info["artist_name"]} : {e}"
            )
            ch.save_checkpoint(i - 1)
    ch.remove_checkpoint_file()


def count_duplicate_midis(dataset_path):
    songs = {}  # To store unique MIDI files by their MD5 hash
    duplicates = 0

    for dirpath, _, files in os.walk(dataset_path):
        if not files:
            continue

        midi_files = [f for f in files if f.endswith(".mid")]

        for midi_file in midi_files:
            midi_path = os.path.join(dirpath, midi_file)

            # Compute MD5 hash of the MIDI file
            with open(midi_path, "rb") as file:
                midi_md5 = hashlib.md5(file.read()).hexdigest()

            if midi_md5 not in songs:
                songs[midi_md5] = midi_path
            else:
                # Duplicate found
                duplicates += 1

    print(f"Total duplicates: {duplicates}")


def count_valid_fields(json_data):
    """
    Count the number of valid (non-NaN, non-empty) fields in the JSON data.
    """
    valid_count = 0
    for key, value in json_data.items():
        if value not in [None, "", "NaN"] and not (
            isinstance(value, float) and np.isnan(value)
        ):
            valid_count += 1
    return valid_count


def handle_duplicates(dataset_path):
    """
    Detect and handle duplicate MIDI files in the dataset by checking MD5 hashes.
    Retains the file with more information in the associated JSON file.
    """
    songs = {}
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if not filename.endswith(".mid"):
                continue

            midi_path = os.path.join(dirpath, filename)
            json_path = os.path.splitext(midi_path)[0] + ".json"
            try:
                # Calculate MD5 hash of the MIDI file
                with open(midi_path, "rb") as midi_file:
                    midi_md5 = hashlib.md5(midi_file.read()).hexdigest()

                if midi_md5 not in songs:
                    songs[midi_md5] = [(dirpath, midi_path, json_path)]
                else:
                    songs[midi_md5].append((dirpath, midi_path, json_path))
            except Exception as e:
                print(f"Error processing file {midi_path}: {e}")

    # Process duplicates
    for md5, files in songs.items():
        if len(files) == 1:
            continue  # No duplicates for this MD5

        print(f"Found duplicates for MD5 {md5}:")
        for file_info in files:
            print(f"  - {file_info[1]}")

        # Compare duplicates
        files_info = []
        for dirpath, midi_path, json_path in files:
            try:
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                    valid_fields = count_valid_fields(json_data)
                    genre = json_data.get("genre", "Unknown")
                    subgenre = json_data.get("subgenre", "Unknown")
                    files_info.append(
                        (valid_fields, genre, subgenre, midi_path, json_path)
                    )
            except Exception as e:
                print(f"Error reading JSON file {json_path}: {e}")
                continue

        # Sort by valid fields, then genre/subgenre
        files_info.sort(key=lambda x: (-x[0], x[1] == "Unknown", x[2] == "Unknown"))

        # Keep the best file, delete the rest
        for file_to_delete in files_info[1:]:
            _, _, _, midi_path, json_path = file_to_delete
            try:
                os.remove(midi_path)
                os.remove(json_path)
                print(f"Deleted duplicate: {midi_path} and {json_path}")
            except Exception as e:
                print(f"Error deleting files {midi_path} or {json_path}: {e}")


def delete_empty_directories(src_dir):
    """
    Recursively delete all empty directories in the dataset path.
    """
    for dirpath, _, _ in os.walk(src_dir, topdown=False):
        try:
            os.rmdir(dirpath)
            print(f"Deleted empty directory: {dirpath}")
        except OSError as e:
            print(f"Error deleting directory {dirpath}: {e}")


def verify_song_directories(base_path):
    errors = []
    for dirpath, _, files in os.walk(base_path):
        # Skip directories with no files (not a song directory)
        if len(files) == 0:
            continue

        # Count .mid and .json files
        mid_files = [f for f in files if f.endswith(".mid")]
        json_files = [f for f in files if f.endswith(".json")]

        # Check for exactly one .mid and one .json file
        if len(mid_files) != 1 or len(json_files) != 1:
            errors.append(
                {
                    "path": dirpath,
                    "mid_count": len(mid_files),
                    "json_count": len(json_files),
                    "files": files,
                }
            )

    return errors


def handle_multiple_midis_in_directory(dataset_path):
    """
    Handles situations where a directory contains multiple MIDI files.
    Deletes all but one based on criteria such as file size and metadata completeness.
    """
    for dirpath, _, filenames in os.walk(dataset_path):
        midi_files = [f for f in filenames if f.endswith(".mid")]
        json_files = {
            f.replace(".json", ""): f for f in filenames if f.endswith(".json")
        }

        if len(midi_files) > 1:
            print(f"Multiple MIDI files found in directory: {dirpath}")

            midi_file_info = []

            for midi_file in midi_files:
                midi_path = os.path.join(dirpath, midi_file)
                json_file = json_files.get(midi_file.replace(".mid", ""))

                # Calculate file size
                file_size = os.path.getsize(midi_path)

                # Assess metadata completeness
                metadata_score = 0
                if json_file:
                    json_path = os.path.join(dirpath, json_file)
                    try:
                        with open(json_path, "r") as f:
                            metadata = json.load(f)
                        metadata_score = sum(
                            1
                            for key, value in metadata.items()
                            if value not in [None, "", "NaN"]
                        )
                    except Exception as e:
                        print(f"Error reading JSON file {json_path}: {e}")

                midi_file_info.append((midi_path, json_file, file_size, metadata_score))

            # Sort by metadata score (desc), then by file size (desc)
            midi_file_info.sort(key=lambda x: (x[3], x[2]), reverse=True)

            # Keep only the best file
            best_file = midi_file_info[0]
            print(f"Keeping: {best_file[0]}")

            # Delete the rest
            for midi_path, json_file, _, _ in midi_file_info[1:]:
                try:
                    os.remove(midi_path)
                    print(f"Deleted MIDI file: {midi_path}")
                except Exception as e:
                    print(f"Error deleting MIDI file {midi_path}: {e}")

                if json_file:
                    json_path = os.path.join(dirpath, json_file)
                    try:
                        os.remove(json_path)
                        print(f"Deleted JSON file: {json_path}")
                    except Exception as e:
                        print(f"Error deleting JSON file {json_path}: {e}")
