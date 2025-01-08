import music21
import glob
import os
from collections import Counter


# Map quarter lengths to musical note types
# duplex-maxima == 64.0, maxima == 32.0, longa == 16.0
# breve == 8.0, whole == 4.0, half == 2.0
# quarter == 1.0, eighth == 0.5, 16th == 0.25
# 32nd == 0.125, 64th == 0.0625, 128th == 0.03125
# 256th == 0.015625, 512th == 0.0078125, 1024th == 0.00390625
# 2048th == 0.001953125, zero == 0.0


def quarter_length_to_type(q_len: float) -> str:
    """
    Converts quarter lengths to note types (e.g., "whole", "quarter", "eighth").
    Omits 32nd notes and shorter by default.

    Parameters:
    - q_len (float): Quarter length of the note or rest.

    Returns:
    - str: Closest note type or "zero" for very short durations.
    """
    if q_len < 0.25:  # Omit 32nd notes and shorter
        return "zero"
    elif q_len == 128.0:  # Special case to avoid exception for 128.0
        q_len += 1.0
    return music21.duration.quarterLengthToClosestType(q_len)[0]


def encode_midi_to_sequence(midi_path: str) -> list:
    """
    Encodes a MIDI file into a sequence of words representing notes, chords, and rests.

    Parameters:
    - midi_path (str): Path to the MIDI file to be encoded.

    Returns:
    - list: Encoded word sequence for the MIDI file.
    """
    midi_stream = music21.converter.parse(midi_path)
    sequence = []

    flat_stream = midi_stream.flatten().notes
    last_offset = 0  # Tracks the start of previous note in quarter lengths
    longest_note_end = 0  # Tracks the end of note which lasts the longest before pause in quarter lengths

    for element in flat_stream:
        current_offset = element.offset

        # Handle pauses
        if longest_note_end < current_offset:
            pause_duration = current_offset - longest_note_end
            if pause_duration > 0.0:
                pause_word = f"PAUSE_{quarter_length_to_type(pause_duration)}"
                sequence.append(pause_word)
                last_offset = longest_note_end

        # Relative start time
        rel_start_time = current_offset - last_offset
        rel_start_type = quarter_length_to_type(rel_start_time)

        if isinstance(element, music21.note.Note):
            # Encode a single note
            pitch = element.pitch.nameWithOctave
            duration_type = quarter_length_to_type(element.quarterLength)
            word = f"{pitch}_{duration_type}_{rel_start_type}"
            sequence.append(word)
            longest_note_end = max(
                longest_note_end, current_offset + element.quarterLength
            )

        elif isinstance(element, music21.chord.Chord):
            # Encode each note in the chord
            duration_type = quarter_length_to_type(element.quarterLength)
            longest_note_end = max(
                longest_note_end, current_offset + element.quarterLength
            )
            for i, pitch in enumerate(element.pitches):
                pitch_name = pitch.nameWithOctave
                word = f"{pitch_name}_{duration_type}_{rel_start_type if i == 0 else quarter_length_to_type(0.0)}"
                sequence.append(word)

        last_offset = current_offset

    return sequence


def create_midi_from_sequence(word_sequence: list, bpm: int, output_path: str):
    """
    Converts a sequence of words (representing music events) into a MIDI file.

    Parameters:
    - word_sequence (list): The sequence of generated words (representing music events).
    - bpm (int): The tempo (beats per minute).
    - output_path (str): The path where the generated MIDI file will be saved.
    """
    midi_stream = music21.stream.Stream()

    bpm = music21.tempo.MetronomeMark(number=max(1, min(bpm, 512)))
    midi_stream.append(bpm)

    current_offset = 0  # Tracks the end of current note or pause in quarter lengths
    last_event_duration = 0
    longest_note_offset = 0  # Tracks the end of note which lasts the longest before pause in quarter lengths

    for word in word_sequence:
        if "PAUSE" in word:
            # Handle pauses
            _, duration_str = word.split("_")
            duration_qlen = music21.duration.Duration(duration_str).quarterLength
            current_offset = longest_note_offset + duration_qlen
            last_event_duration = duration_qlen
            longest_note_offset = 0

        else:
            # Handle notes
            pitch_str, dur_str, rel_start_str = word.split("_")
            pitch = music21.pitch.Pitch(pitch_str)
            duration_qlen = music21.duration.Duration(dur_str).quarterLength
            rel_start_time = music21.duration.Duration(rel_start_str).quarterLength

            # Adjust the note's start time
            current_offset += rel_start_time - last_event_duration

            note = music21.note.Note(pitch)
            note.quarterLength = duration_qlen

            midi_stream.insert(current_offset, note)
            current_offset += duration_qlen
            last_event_duration = duration_qlen
            longest_note_offset = max(longest_note_offset, current_offset)

    midi_stream.write("midi", fp=output_path)


def generate_word_files(dirname: str, padding_len: int = 0):
    """
    Encodes MIDI files into word sequences and saves them as text files.

    Parameters:
    - dirname (str): Directory containing MIDI files.
    - padding_len (int): Number of "PAD" tokens to prepend to each sequence.
    """
    for midi in glob.glob(f"{dirname}/**/*.mid", recursive=True):
        try:
            encoded_sequence = encode_midi_to_sequence(midi)
            head, tail = os.path.split(midi)

            words = ["PAD"] * padding_len + encoded_sequence
            txt_name = tail.replace(".mid", ".txt")

            with open(os.path.join(head, txt_name), "w", encoding="utf-8") as f:
                f.write(" ".join(words))
                print(f"Saved: {txt_name}")
        except Exception as e:
            print(f"Error processing {midi}: {e}")


def get_tempos_from_midi(file_path: str) -> list[int] | None:
    """
    Extracts tempo values from a MIDI file.

    Parameters:
    - file_path (str): Path to the MIDI file.

    Returns:
    - list[int] | None: List of tempo values if found, or None if no tempos exist.
    """
    try:
        s = music21.converter.parse(file_path)
        tempos = []
        for el in s.flatten().getElementsByClass("MetronomeMark"):
            tempos.append(round(el.number))

        return tempos if tempos else None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_genre_tempos(data_path: str) -> dict:
    """
    Extracts the most common tempo for each genre in the dataset.

    Parameters:
    - data_path (str): Path to the root data directory containing 'train' and 'test' folders.

    Returns:
    - dict: A dictionary mapping each genre to its most common tempo or a message if no tempo data is found.
    """
    genre_tempo_summary = {}
    for genre in os.listdir(os.path.join(data_path, "train")):
        genre_test_dir = os.path.join(data_path, "test", genre)
        genre_train_dir = os.path.join(data_path, "train", genre)

        print(f"Processing genre: {genre}")
        tempo_list = []

        for mid in glob.glob(f"{genre_test_dir}/**/*.mid", recursive=True):
            tempos = get_tempos_from_midi(mid)
            if tempos is not None:
                tempo_list += tempos
        for mid in glob.glob(f"{genre_train_dir}/**/*.mid", recursive=True):
            tempos = get_tempos_from_midi(mid)
            if tempos is not None:
                tempo_list += tempos

        if tempo_list:
            most_common_tempo = Counter(tempo_list).most_common(1)[0][0]
            print(f"{genre} tempo = {most_common_tempo}")
            genre_tempo_summary[genre] = most_common_tempo
        else:
            genre_tempo_summary[genre] = "No tempo data found"
    return genre_tempo_summary
