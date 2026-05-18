"""Unit tests for core.config module."""

from __future__ import annotations

from core.config import config_to_dict, load_config


class TestLoadConfig:
    def test_load_nonexistent_returns_empty(self):
        cfg = load_config("nonexistent_file.yaml")
        assert cfg is not None

    def test_load_existing_yaml(self, tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("project:\n  name: test\n  seed: 99\n")
        cfg = load_config(str(yaml_file))
        d = config_to_dict(cfg)
        assert d["project"]["name"] == "test"
        assert d["project"]["seed"] == 99

    def test_merge_two_configs(self, tmp_path):
        f1 = tmp_path / "base.yaml"
        f1.write_text("a: 1\nb: 2\n")
        f2 = tmp_path / "override.yaml"
        f2.write_text("b: 99\nc: 3\n")
        cfg = load_config(str(f1), str(f2))
        d = config_to_dict(cfg)
        assert d["a"] == 1
        assert d["b"] == 99
        assert d["c"] == 3

    def test_dotlist_overrides(self, tmp_path):
        f = tmp_path / "cfg.yaml"
        f.write_text("model:\n  lr: 0.001\n")
        cfg = load_config(str(f), overrides=["model.lr=0.01"])
        d = config_to_dict(cfg)
        assert d["model"]["lr"] == 0.01
