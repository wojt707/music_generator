import music21
import glob
import os

# duplex-maxima == 64.0
# maxima == 32.0
# longa == 16.0
# breve == 8.0
# whole == 4.0
# half == 2.0
# quarter == 1.0
# eighth == 0.5
# 16th == 0.25
# 32nd == 0.125
# 64th == 0.0625
# 128th == 0.03125
# 256th == 0.015625
# 512th == 0.0078125
# 1024th == 0.00390625
# 2048th == 0.001953125
# zero == 0.0


def quarter_length_to_type(q_len: float):
    # if q_len < 0.0078125:  # omit 1024th notes and shorter
    if q_len < 0.25:  # omit 32nd notes and shorter
        return "zero"
    elif (
        q_len == 128.0
    ):  # for some reason exception is raised when q_len is exactly 128.0
        q_len += 1.0
    # TODO maybe this -> music21.duration.durationTupleFromQuarterLength
    return music21.duration.quarterLengthToClosestType(q_len)[0]


def encode_midi_to_sequence(midi_path: str) -> list:
    """
    Encodes a MIDI file into a sequence of words, including notes, chords, and rests.

    Parameters:
        midi_path (str): Path to the MIDI file to be encoded.

    Returns:
        list: A list of encoded words representing the MIDI file.
    """

    # Load the MIDI file
    midi_stream = music21.converter.parse(midi_path)
    sequence = []

    # Flatten the stream to access individual notes, chords, and rests
    flat_stream = midi_stream.flatten().notes

    # Variables to track timing
    last_offset = 0  # Tracks the start of previous note in quarter lengths
    longest_note_end = 0  # Tracks the end of note which lasts the longest before pause in quarter lengths

    for element in flat_stream:
        # Current offset of the element
        current_offset = element.offset
        if longest_note_end < current_offset:
            # PAUSEEEEEEEEEE
            duration_qlen = current_offset - longest_note_end
            if duration_qlen != 0.0:

                word = f"PAUSE_{quarter_length_to_type(duration_qlen)}"
                last_offset = longest_note_end
                sequence.append(word)

        # Calculate the relative start time
        rel_start_time = current_offset - last_offset
        rel_start_time_type = quarter_length_to_type(rel_start_time)

        if isinstance(element, music21.note.Note):
            # Encode a single note
            pitch = element.pitch.nameWithOctave
            duration_type = quarter_length_to_type(element.quarterLength)
            word = f"{pitch}_{duration_type}_{rel_start_time_type}"
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
                word = f"{pitch_name}_{duration_type}_{rel_start_time_type if i == 0 else quarter_length_to_type(0.0)}"
                sequence.append(word)

        # Update the last offset
        last_offset = current_offset

    return sequence


def create_midi_from_sequence(word_sequence: list, bpm: int, output_path: str):

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

            # Add the note to the stream
            midi_stream.insert(current_offset, note)
            current_offset += duration_qlen
            last_event_duration = duration_qlen
            longest_note_offset = max(longest_note_offset, current_offset)

    midi_stream.write("midi", fp=output_path)


def generate_word_files(dirname, padding_len: int = 0):
    """
    Processes MIDI files, encodes tracks into word sequences, and saves them to text files.

    :param dirname: Directory containing MIDI files.
    :param padding_len: Number of "PAD" tokens to prepend to each sequence.
    """
    for midi in glob.glob(f"{dirname}/**/*.mid", recursive=True):
        try:
            encoded_sequence = encode_midi_to_sequence(midi)
            head, tail = os.path.split(midi)

            # Create a txt file and write the words with padding before into it
            words = ["PAD"] * padding_len
            words += encoded_sequence

            txt_name = tail.replace(".mid", ".txt")

            with open(os.path.join(head, txt_name), "w", encoding="utf-8") as f:
                f.write(" ".join(words))

                print(f"Saved: {txt_name}")
        except Exception as e:
            print(f"Error processing {midi}: {e}")
