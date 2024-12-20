import music21
import glob
import os

import music21.midi.translate


def encode_midi(midi_path):
    # Open and read the MIDI file
    midi_file = music21.midi.MidiFile()
    midi_file.open(midi_path)
    midi_file.read()
    midi_file.close()

    ticks_per_quarter = midi_file.ticksPerQuarterNote
    if ticks_per_quarter is None:
        raise ValueError("Ticks per quarter not found in MIDI file.")

    time_signature = None
    tempo = None  # Microseconds per quarter note. BPM = 60_000_000 / tempo

    sequences_by_track = []

    for track in midi_file.tracks:
        if is_percussion_track(track):
            print("Percussion detected eeeeeeeeeeeeeeeee")
            continue

        pending_notes = []
        sequence = []
        current_time = 0  # In ticks
        instrument_id = None

        for event in track.events:
            # TODO add tempo change and time signature change as event
            if isinstance(event, music21.midi.DeltaTime):
                current_time += event.time
            elif isinstance(event, music21.midi.MidiEvent):
                # Extract global metadata from track 0
                if track.index == 0:
                    if event.type == music21.midi.MetaEvents.TIME_SIGNATURE:
                        numerator, denominator, _, _ = event.data
                        time_signature = (numerator, 2**denominator)

                    elif event.type == music21.midi.MetaEvents.SET_TEMPO:
                        tempo = int.from_bytes(event.data, byteorder="big")

                # Extract instrument (PROGRAM_CHANGE)
                if event.type == music21.midi.ChannelVoiceMessages.PROGRAM_CHANGE:
                    instrument_id = event.data
                # Process NOTE_ON and NOTE_OFF
                elif (
                    event.type == music21.midi.ChannelVoiceMessages.NOTE_ON
                    and event.velocity > 0
                ):
                    # Start a note
                    pending_notes.append(
                        {
                            "type": "note",
                            "pitch": event.pitch,
                            "start_time_ticks": current_time,
                            "duration_ticks": None,  # To be filled when NOTE_OFF is encountered
                        }
                    )
                elif event.type == music21.midi.ChannelVoiceMessages.NOTE_OFF or (
                    event.type == music21.midi.ChannelVoiceMessages.NOTE_ON
                    and event.velocity == 0
                ):
                    for note in reversed(pending_notes):
                        if (
                            note["type"] == "note"
                            and note["pitch"] == event.pitch
                            and note["duration_ticks"] is None
                        ):
                            note["duration_ticks"] = (
                                current_time - note["start_time_ticks"]
                            )
                            break

        pending_notes.sort(key=lambda x: (x["start_time_ticks"], x["pitch"]))

        max_end_time_ticks = (
            0  # Maximum end time in ticks of notes which were already added to sequence
        )

        previous_start_time_ticks = 0  # Start time in ticks of previous note or pause

        for note in pending_notes:
            # Insert a pause if there is gap between previous notes and current
            if note["start_time_ticks"] > max_end_time_ticks:
                sequence.append(
                    {
                        "type": "pause",
                        # "rel_start_time": (
                        #     max_end_time_ticks - previous_start_time_ticks
                        # )
                        # / ticks_per_quarter,
                        "duration": (note["start_time_ticks"] - max_end_time_ticks)
                        / ticks_per_quarter,
                    }
                )
                previous_start_time_ticks = max_end_time_ticks

            if note["duration_ticks"] is not None:
                # Insert note with duration and relative start time in quarter beats
                sequence.append(
                    {
                        "type": "note",
                        "pitch": note["pitch"],
                        "rel_start_time": (
                            note["start_time_ticks"] - previous_start_time_ticks
                        )
                        / ticks_per_quarter,
                        "duration": (note["duration_ticks"] / ticks_per_quarter),
                    }
                )
            max_end_time_ticks = max(
                max_end_time_ticks,
                note["start_time_ticks"] + (note["duration_ticks"] or 0),
            )
            previous_start_time_ticks = note["start_time_ticks"]

        if sequence:
            instrument_name = "unknown"
            if instrument_id is not None:
                try:
                    instrument_name = midi_program_to_group(instrument_id)
                except Exception as e:
                    print(
                        f"Error: Unable to map instrument_id {instrument_id}. Using 'unknown'."
                    )

            sequences_by_track.append(
                {
                    "track_id": track.index,
                    "instrument_id": instrument_id,
                    "instrument": instrument_name,
                    "sequence": sequence,
                }
            )
    return {
        "ticks_per_quarter": ticks_per_quarter,
        "time_signature": time_signature,
        "tempo": tempo,
        "tracks": sequences_by_track,
    }


def decode_midi(midi_data, output_path):
    # TODO - doesnt work
    midi_file = music21.midi.MidiFile()
    midi_file.ticksPerQuarterNote = midi_data["ticks_per_quarter"]

    # Create the first track
    head_track = music21.midi.MidiTrack(0)
    midi_file.tracks.append(head_track)

    # Add time signature
    if midi_data.get("time_signature"):
        numerator, denominator = midi_data["time_signature"]
        time_sig_event = music21.midi.MidiEvent(head_track)
        time_sig_event.type = music21.midi.MetaEvents.TIME_SIGNATURE
        time_sig_event.data = bytes(
            [numerator, int(denominator).bit_length() - 1, 24, 8]
        )
        head_track.events.append(time_sig_event)

    # Add tempo
    if midi_data.get("tempo"):
        tempo_event = music21.midi.MidiEvent(head_track)
        tempo_event.type = music21.midi.MetaEvents.SET_TEMPO
        tempo_event.data = midi_data["tempo"].to_bytes(3, byteorder="big")
        head_track.events.append(tempo_event)

    # head_track.events.append(
    #     music21.midi.MidiEvent(head_track, type=music21.midi.MetaEvents.END_OF_TRACK)
    # )

    # # Create tracks from the note sequences
    for track_data in midi_data["tracks"]:
        track = music21.midi.MidiTrack(track_data["track_id"])
        midi_file.tracks.append(track)

        # Add program change (instrument)
        if track_data.get("instrument_id") is not None:
            program_change_event = music21.midi.MidiEvent(track)
            program_change_event.type = music21.midi.ChannelVoiceMessages.PROGRAM_CHANGE
            program_change_event.data = track_data["instrument_id"]
            track.events.append(program_change_event)

        current_time = 0
        for event in track_data["sequence"]:
            rel_start_time_ticks = int(
                event["rel_start_time"] * midi_data["ticks_per_quarter"]
            )
            if event["type"] == "pause":
                # Add a pause (delta time update only)
                current_time += rel_start_time_ticks + int(
                    event["duration"] * midi_data["ticks_per_quarter"]
                )
            elif event["type"] == "note":
                # Add NOTE_ON event
                delta_time_event = music21.midi.DeltaTime(
                    track, time=rel_start_time_ticks
                )
                track.events.append(delta_time_event)
                current_time += rel_start_time_ticks

                note_on_event = music21.midi.MidiEvent(track)
                note_on_event.type = music21.midi.ChannelVoiceMessages.NOTE_ON
                note_on_event.pitch = event["pitch"]
                note_on_event.velocity = 64  # Default velocity
                track.events.append(note_on_event)

                # Add NOTE_OFF event
                duration_ticks = int(event["duration"] * midi_data["ticks_per_quarter"])
                delta_time_event = music21.midi.DeltaTime(track, time=duration_ticks)
                track.events.append(delta_time_event)

                note_off_event = music21.midi.MidiEvent(track)
                note_off_event.type = music21.midi.ChannelVoiceMessages.NOTE_OFF
                note_off_event.pitch = event["pitch"]
                note_off_event.velocity = 0
                track.events.append(note_off_event)
        # track.events.append(
        #     music21.midi.MidiEvent(track, type=music21.midi.MetaEvents.END_OF_TRACK)
        # )

    midi_file.open(output_path, attrib="wb")
    midi_file.write()
    midi_file.close()
    print(f"Decoded MIDI saved to {output_path}")


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
    if q_len < 0.0078125:  # omit 1024th notes and shorter
        return "zero"
    elif (
        q_len == 128.0
    ):  # for some reason exception is raised when q_len is exactly 128.0
        q_len += 1.0
    # TODO maybe this -> music21.duration.durationTupleFromQuarterLength
    return music21.duration.quarterLengthToClosestType(q_len)[0]


def midi_program_to_group(midi_program: int) -> str:
    """
    Maps a MIDI program number to a grouped instrument name.
    """
    instrument = ""
    if midi_program < 0 or midi_program > 127:
        instrument = "unknown"
    elif midi_program < 8:
        instrument = "piano"
    elif midi_program < 16:
        instrument = "chromatic percussion"
    elif midi_program < 24:
        instrument = "organ"
    elif midi_program < 32:
        instrument = "guitar"
    elif midi_program < 40:
        instrument = "bass"
    elif midi_program < 48:
        instrument = "strings"
    elif midi_program < 56:
        instrument = "ensemble"
    elif midi_program < 64:
        instrument = "brass"
    elif midi_program < 72:
        instrument = "reed"
    elif midi_program < 80:
        instrument = "pipe"
    elif midi_program < 88:
        instrument = "synth lead"
    elif midi_program < 96:
        instrument = "synth pad"
    elif midi_program < 104:
        instrument = "synth effects"
    elif midi_program < 112:
        instrument = "ethnic"
    elif midi_program < 120:
        instrument = "percussive"
    elif midi_program < 128:
        instrument = "sound effects"

    return instrument


def is_percussion_track(midi_track):
    """
    Determines if a track is a percussion track based on Channel 10.
    """
    for event in midi_track.events:
        if isinstance(event, music21.midi.MidiEvent):
            if event.channel == 9:  # Channel 10 in MIDI
                return True
    return False


def sequence_to_words(sequence):
    words = []
    for event in sequence:
        if event["type"] == "note":
            name_with_octave = music21.pitch.Pitch(event["pitch"]).nameWithOctave

            dur_str = quarter_length_to_type(event["duration"])
            rel_st_t_str = quarter_length_to_type(event["rel_start_time"])

            if dur_str == "zero":
                continue

            words.append(f"{name_with_octave}_{dur_str}_{rel_st_t_str}")

        elif event["type"] == "pause":
            dur_str = quarter_length_to_type(event["duration"])

            if dur_str == "zero":
                continue

            words.append(f"PAUSE_{dur_str}")

    return words


def generate_word_files(dirname):
    for midi in glob.glob(f"{dirname}/**/*.mid", recursive=True):
        try:
            encoded = encode_midi(midi)
            head, tail = os.path.split(midi)

            for track in encoded["tracks"]:
                # For each track, create a txt file and write the words into it.
                words = sequence_to_words(track["sequence"])

                txt_name = f"{tail.replace(".mid", "")}_{track['instrument']}_{track['instrument_id']}_{track['track_id']}.txt"

                with open(os.path.join(head, txt_name), "w", encoding="utf-8") as f:
                    f.write(" ".join(words))

                # print(f"Saved: {txt_name}")
        except Exception as e:
            print(f"Error processing {midi}: {e}")
