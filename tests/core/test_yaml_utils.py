import pytest
from ceo_chatbot_core.yaml_utils import load_yaml


def test_load_yaml_returns_dict(tmp_path):
    """Parses a valid YAML file and returns the correct nested dict."""
    f = tmp_path / "config.yml"
    f.write_text("key: value\nnested:\n  a: 1\n")
    assert load_yaml(f) == {"key": "value", "nested": {"a": 1}}


def test_load_yaml_empty_file_returns_empty_dict(tmp_path):
    """An empty file returns {} rather than None or raising."""
    f = tmp_path / "empty.yml"
    f.write_text("")
    assert load_yaml(f) == {}


def test_load_yaml_missing_file_raises(tmp_path):
    """Raises FileNotFoundError for a path that does not exist."""
    with pytest.raises(FileNotFoundError):
        load_yaml(tmp_path / "nonexistent.yml")


def test_load_yaml_accepts_string_path(tmp_path):
    """Works correctly when called with a str instead of a Path."""
    f = tmp_path / "config.yml"
    f.write_text("x: 1\n")
    assert load_yaml(str(f))["x"] == 1
