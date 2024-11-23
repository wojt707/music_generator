import os
import json
from .hdf5_getters import *


# Extract metadata for a single song
def get_song_metadata(h5_file_path):
    """Get relevant fields from an .h5 file."""
    try:
        with open_h5_file_read(h5_file_path) as h5:
            msd_id = os.path.splitext(os.path.basename(h5_file_path))[0]
            artist_location = get_artist_location(h5)
            metadata = {
                "msd_id": msd_id,
                "title": get_title(h5),
                "artist_name": get_artist_name(h5),
                "artist_location": (
                    artist_location.decode("utf-8") if artist_location else ""
                ),
                "tempo": float(get_tempo(h5)),
                "key": int(get_key(h5)),
                "key_confidence": float(get_key_confidence(h5)),
                "time_signature": int(get_time_signature(h5)),
                "time_signature_confidence": float(get_time_signature_confidence(h5)),
                "duration": float(get_duration(h5)),
                "release_year": int(get_year(h5)) if get_year(h5) else 0,
                "danceability": (
                    float(get_danceability(h5)) if get_danceability(h5) else 0.0
                ),
                "song_hotttnesss": (
                    float(get_song_hotttnesss(h5)) if get_song_hotttnesss(h5) else 0.0
                ),
                "artist_hotttnesss": (
                    float(get_artist_hotttnesss(h5))
                    if get_artist_hotttnesss(h5)
                    else 0.0
                ),
            }
            return metadata
    except Exception as e:
        print(f"Error processing file {h5_file_path}: {e}")
        return None


# Process all .h5 files in h5_root_dir directory
def get_all_songs_metadata(h5_root_dir):
    """Process all .h5 files in the h5_root_dir directory."""
    all_metadata = []

    # Walk through the input directory
    for root, _, files in os.walk(h5_root_dir):
        for file in files:
            if file.endswith(".h5"):
                h5_file_path = os.path.join(root, file)
                metadata = get_song_metadata(h5_file_path)
                if metadata:
                    all_metadata.append(metadata)

    print(f"Metadata for {len(all_metadata)} extracted.")
    return all_metadata


def generate_metadata_json(h5_root_dir, output_json_path):
    """Save the metadata to a JSON file."""
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(get_all_songs_metadata(h5_root_dir), f, indent=4)
        print(f"Metadata saved to {output_json_path}")
