from unittest.mock import MagicMock, patch

from avlnn.notify import DEFAULT_NTFY_TOPIC, NtfyConfig, resolve_ntfy_config, send_ntfy


def test_from_env_missing_topic_returns_none(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert NtfyConfig.from_env() is None


def test_from_env_reads_topic_and_default_server(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    config = NtfyConfig.from_env()
    assert config == NtfyConfig(topic="my-topic", server="https://ntfy.sh")


def test_from_env_reads_custom_server(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "my-topic")
    monkeypatch.setenv("NTFY_SERVER", "https://ntfy.example.com")
    config = NtfyConfig.from_env()
    assert config == NtfyConfig(topic="my-topic", server="https://ntfy.example.com")


def test_resolve_explicit_arg_beats_env(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "env-topic")
    config = resolve_ntfy_config("cli-topic", None)
    assert config.topic == "cli-topic"


def test_resolve_empty_string_disables(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "env-topic")
    assert resolve_ntfy_config("", None) is None


def test_resolve_env_beats_default(monkeypatch):
    monkeypatch.setenv("NTFY_TOPIC", "env-topic")
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    config = resolve_ntfy_config(None, None)
    assert config.topic == "env-topic"


def test_resolve_falls_back_to_project_default(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    monkeypatch.delenv("NTFY_SERVER", raising=False)
    config = resolve_ntfy_config(None, None)
    assert config.topic == DEFAULT_NTFY_TOPIC


def test_send_ntfy_success():
    config = NtfyConfig(topic="my-topic")
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        ok = send_ntfy(config, "hello", title="Test", tags=["airplane"])

    assert ok is True
    request = mock_urlopen.call_args[0][0]
    assert request.full_url == "https://ntfy.sh/my-topic"
    assert request.data == b"hello"
    assert request.headers["Title"] == "Test"
    assert request.headers["Tags"] == "airplane"


def test_send_ntfy_network_failure_returns_false_not_raises():
    config = NtfyConfig(topic="my-topic")
    with patch("urllib.request.urlopen", side_effect=OSError("network down")):
        ok = send_ntfy(config, "hello")
    assert ok is False
