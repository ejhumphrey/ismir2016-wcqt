import os
import pytest
import wcqtlib.config as config

DEFAULT_PATH = os.path.join(os.path.dirname(__file__),
                            os.pardir, "data", "master_config.yaml")


def __test_config(obj, key, expected_value):
    assert obj[key] == expected_value


@pytest.mark.skipif(not all([os.path.exists(DEFAULT_PATH)]),
                    reason="Config doesn't exist")
def test_from_yaml():
    primary_config = config.Config.from_yaml(DEFAULT_PATH)
    assert bool(primary_config)


def test_load_config_simple():
    data = {
        "key1": "value1",
        "foo": "bar",
        "key2": 10
    }

    c = config.Config(data)
    yield __test_config, c, "key1", "value1"
    yield __test_config, c, "foo", "bar"
    yield __test_config, c, "key2", 10


def test_load_config_hierarchical():
    data = {
        "key1a": {
            "key1b": {
                "key1c": "value1"
            },
            "key2b": 42
        },
        "foo": "bar",
        "key2": {
            "key2b": "ornot2b"
        }
    }

    c = config.Config(data)
    yield __test_config, c, "key1a/key1b/key1c", "value1"
    yield __test_config, c, "key1a/key2b", 42
    yield __test_config, c, "foo", "bar"
    yield __test_config, c, "key2", {"key2b": "ornot2b"}
    yield __test_config, c, "key2/key2b", "ornot2b"
