import responses
import requests
from utils import fetch_poster, is_uplifting

# This URL MUST exactly match the one in utils.py
TMDB_URL = "https://api.themoviedb.org/3/movie/123?api_key=8265bd1679663a7ea12ac168da84d2e8"


@responses.activate
def test_fetch_poster_success():
    responses.add(
        responses.GET,
        TMDB_URL,
        json={"poster_path": "/testposter.jpg"},
        status=200
    )

    poster = fetch_poster(123)
    assert poster.endswith("/testposter.jpg")


@responses.activate
def test_fetch_poster_no_poster():
    responses.add(
        responses.GET,
        TMDB_URL,
        json={"poster_path": None},
        status=200
    )

    poster = fetch_poster(123)
    assert "No+Poster" in poster


@responses.activate
def test_fetch_poster_api_failure():
    responses.add(
        responses.GET,
        TMDB_URL,
        body=requests.exceptions.Timeout()
    )

    poster = fetch_poster(123)
    assert "Error" in poster


def test_is_uplifting_positive_text():
    assert is_uplifting("This movie was amazing and inspiring!")


def test_is_uplifting_negative_text():
    assert not is_uplifting("This movie was terrible and depressing.")