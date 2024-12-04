import os


def load_dataset(dataset_path):
    dataset = {}

    # Read all MIDI songs
    for dir, _, files in os.walk(dataset_path):
        if len(files) == 0:
            continue

        path_parts = dir.split("\\")

        song = path_parts[-1]
        artist = path_parts[-2]
        subgenre = path_parts[-3]
        genre = path_parts[-4]

        # Add full path to midi file and json file
        full_path_files = [os.path.join(dir, filename) for filename in files]

        if genre not in dataset:
            dataset[genre] = {subgenre: {artist: {song: full_path_files}}}
            continue

        if subgenre not in dataset[genre]:
            dataset[genre][subgenre] = {artist: {song: full_path_files}}
            continue

        if artist not in dataset[genre][subgenre]:
            dataset[genre][subgenre][artist] = {song: full_path_files}
            continue

        if song not in dataset[genre][subgenre][artist]:
            dataset[genre][subgenre][artist][song] = full_path_files
            continue

        print("Error - duplicated. Song already in dataset.")

    return dataset


def get_midis_by_genre(dataset):
    midis_by_genre = {}

    for genre in dataset:
        for subgenre in dataset[genre]:
            for artist in dataset[genre][subgenre]:
                for song in dataset[genre][subgenre][artist]:
                    for file in dataset[genre][subgenre][artist][song]:
                        _, ext = os.path.splitext(file)
                        if ext != ".mid":
                            continue
                        if genre not in midis_by_genre:
                            midis_by_genre[genre] = []
                        midis_by_genre[genre].append(file)

    return midis_by_genre


def get_stats(dataset):
    stats = {"Artists": 0, "Genres": 0, "Sub-Genres": 0, "Songs": 0}

    for genre in dataset:
        stats["Genres"] += 1
        for subgenre in dataset[genre]:
            stats["Sub-Genres"] += 1
            for artist in dataset[genre][subgenre]:
                stats["Artists"] += 1
                for song in dataset[genre][subgenre][artist]:
                    stats["Songs"] += 1

    return stats
