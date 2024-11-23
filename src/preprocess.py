import json
import os
import shutil
from preprocessing.million_song.gather_metadata import generate_metadata_json
from preprocessing.paths import (
    H5_PATH,
    METADATA_FILE,
    GENRE_MAP_FILE,
    READY_MIDI_PATH,
    MSD_TO_MIDI_FILE,
    LMD_MATCHED_PATH,
)
from preprocessing.utils import clean_name, get_midi_path
from preprocessing.genre import SpotifyHandler

CHECKPOINT_FILE = "checkpoint.txt"
AUTH_TOKEN = "BQDn9e_K_WiU8FtMNIHYvx0zDtKw9g8DnwCs2WZ6ftt51AUqfnKHSWBOcXGsn-NypdTt_HXtyqLezy5acBds0pKOOxrrqOPC1ahT093QelwQkPEqbRw"


def save_checkpoint(index):
    """Save the current index to a checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))


def load_checkpoint():
    """Load the last processed index from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def add_song(
    midi_path: str,
    song_name: str,
    artist_name: str,
    ready_midi_path,
    spotify_handler: SpotifyHandler,
    genre_mapping=None,
):
    if genre_mapping:
        # Get artist genre from Spotify API
        artist_genre, artist_subgenre = spotify_handler.estimate_artist_genre(
            artist_name, genre_mapping
        )
        artist_genre = clean_name(artist_genre)
        artist_subgenre = clean_name(artist_subgenre)

        # Create genre directory
        genre_dir = os.path.join(ready_midi_path, artist_genre, artist_subgenre)

        try:
            os.makedirs(genre_dir)
        except:
            print("Genre directory already exists.")

    # Create artist directory
    artist_dir = os.path.join(genre_dir, artist_name)

    try:
        os.makedirs(artist_dir)
    except:
        print("Artist directory already exists.")

    midi_new_path = os.path.join(artist_dir, song_name + ".mid")
    shutil.copyfile(midi_path, midi_new_path)


if __name__ == "__main__":
    generate_metadata_json(H5_PATH, METADATA_FILE)

    # Load msd to midi mapping
    msd_to_midi = None
    with open(MSD_TO_MIDI_FILE) as f:
        msd_to_midi = json.load(f)

    # Load genres mapping
    genre_mapping = None
    with open(GENRE_MAP_FILE) as f:
        genre_mapping = json.load(f)

    # Load metadata
    metadata = None
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    last_processed_index = load_checkpoint()

    spotify_handler = SpotifyHandler(AUTH_TOKEN)
    for i, song_info in enumerate(metadata, start=1):
        if i <= last_processed_index:
            continue

        try:
            msd_id = song_info["msd_id"]
            song_name = clean_name(song_info["title"])
            artist_name = clean_name(song_info["artist_name"])
            midi_path = get_midi_path(LMD_MATCHED_PATH, msd_id, msd_to_midi[msd_id])

            print(f"[{i}/{len(metadata)}] Adding song {song_name} by {artist_name}")

            add_song(
                midi_path,
                song_name,
                artist_name,
                READY_MIDI_PATH,
                spotify_handler,
                genre_mapping,
            )
            save_checkpoint(i)
        except Exception as e:
            print(f"Error processing song at index {i}: {e}")
            save_checkpoint(i - 1)
            break

    # print(spotify_handler.estimate_artist_genre_by_song("Teach Us", genre_mapping))
