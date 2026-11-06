import httpx

from ceo_rag_chatbot.rag.local_probe import ModelState, probe_model

BASE_URL = "http://inference:11434/v1"
MODEL = "llama3.1:8b-instruct-q4_K_M"


def _client(tags=(), ps=(), up=True):
    """Fake Ollama server: `tags` are downloaded models, `ps` are models loaded in VRAM."""
    def handler(request: httpx.Request) -> httpx.Response:
        if not up:
            raise httpx.ConnectError("connection refused")
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.0.0"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in tags]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": n} for n in ps]})
        return httpx.Response(404)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_unreachable_when_server_is_down():
    assert probe_model(BASE_URL, MODEL, _client(up=False)) == ModelState.UNREACHABLE


def test_server_up_when_model_not_downloaded():
    assert probe_model(BASE_URL, MODEL, _client()) == ModelState.SERVER_UP


def test_downloaded_when_model_not_in_vram():
    assert probe_model(BASE_URL, MODEL, _client(tags=[MODEL])) == ModelState.DOWNLOADED


def test_loaded_when_model_in_vram():
    assert probe_model(BASE_URL, MODEL, _client(tags=[MODEL], ps=[MODEL])) == ModelState.LOADED


def test_model_name_must_match_exactly():
    # A different tag of the same model family does not count as our model.
    client = _client(tags=["llama3.1:8b"], ps=["llama3.1:8b"])
    assert probe_model(BASE_URL, MODEL, client) == ModelState.SERVER_UP
