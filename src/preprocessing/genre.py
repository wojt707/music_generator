import spotipy


class SpotifyHandler:
    def __init__(self, token: str) -> None:
        self.token = token
        self.spotify = spotipy.Spotify(
            auth=token,
            requests_timeout=30,
        )

    def set_token(self, token: str) -> None:
        self.token = token

    def estimate_artist_genre(self, artist_name: str, genre_mapping):

        results = self.spotify.search(q="artist:" + artist_name, type="artist", limit=1)

        genres = []
        for item in results["artists"]["items"]:
            genres += item["genres"]

        if len(genres) == 0:
            return "Unknown", "Unknown"

        # Map spotify genre to our list of main genres
        for main_genre in genre_mapping:
            for sub_genre in genre_mapping[main_genre]:
                for spotify_genre in genres:
                    if sub_genre in spotify_genre.lower():
                        return main_genre.title(), spotify_genre.title()

        return "Unknown", genres[0].title()

    def estimate_artist_genre_by_song(self, song_name: str, genre_mapping):

        results = self.spotify.search(q="track:" + song_name, type="track", limit=1)
        # print(results)
        artist_name = None
        for item in results["tracks"]["items"]:
            for artist in item["album"]["artists"]:
                artist_name = artist["name"]

        return self.estimate_artist_genre(artist_name, genre_mapping)
