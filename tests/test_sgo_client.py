"""SGOClient sends its key in a header, never in the URL."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "predictions"))
requests = pytest.importorskip("requests")
import sgo_client  # noqa: E402


class FakeResponse:
    ok = True
    status_code = 200

    def json(self):
        return {"data": []}


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return FakeResponse()


def test_key_goes_in_header_not_params(tmp_path, monkeypatch):
    client = sgo_client.SGOClient(api_key="test-key-not-real")
    fake = FakeSession()
    fake.headers.update(client.session.headers)
    client.session = fake

    params = {"leagueID": "NBA"}
    client._get("events", params)

    url, sent = fake.calls[-1]
    assert "apiKey" not in sent
    assert "test-key-not-real" not in url
    assert fake.headers.get("x-api-key") == "test-key-not-real"
    assert params == {"leagueID": "NBA"}  # caller's dict not modified
