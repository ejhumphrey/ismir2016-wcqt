import logging
import os
import yaml

logger = logging.getLogger(__name__)


class Config(object):
    """Simple class for loading config files, with
    hierarchical indexing.

    config = {
        "fred": {
            "bob": 10
        }
    }
    config["fred/bob"] will get you the value 10.
    """
    def __init__(self, data):
        self.data = data

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, 'r') as fh:
            return cls(yaml.load(fh))

    def get(self, key, default=None):
        return self._recursive_get(self.data, key, default)

    def _recursive_get(self, search_dict, key, default=None):
        # Split the key on '/'
        key_segments = key.split('/')
        if len(key_segments) == 1:
            return search_dict[key_segments[0]]
        else:
            new_dict = search_dict[key_segments[0]]
            new_key = "/".join(key_segments[1:])
            if isinstance(new_dict, dict):
                return self._recursive_get(new_dict, new_key, default)
            else:
                logger.error("Intermediate key is not referring to "
                             "a dict; returing the value as-is.")
                return search_dict[new_dict]

    def __getitem__(self, key):
        return self.get(key)

    def __bool__(self):
        return bool(self.data)

    def save(self, path):
        with open(path, 'w') as fh:
            yaml.dump(self.data, fh, default_flow_style=False)
