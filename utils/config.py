import collections
from gym.core import Env


###################################################
### Implements recursive walk over a dictionary ###
###################################################
class Configurable(object):
    def __init__(self, config=None):
        self.config = self.default_config()
        if config:
            Configurable.custom_config(self.config, config)
            Configurable.custom_config(config, self.config)

    def update_config(self, config):
        Configurable.custom_config(self.config, config)

    @classmethod
    def default_config(cls):
        return {}

    @staticmethod
    def custom_config(d, u):
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = Configurable.custom_config(d.get(k, {}), v)
            else:
                d[k] = v
        return d


###############################################
### Implements recursive JSON serialization ###
###############################################
class Serializable(dict):
    def to_dict(self):
        d = dict()
        for (key, value) in self.__dict__.items():
            if isinstance(value, Serializable):
                d[key] = value.to_dict()
            else:
                d[key] = repr(value)
        return d

    def from_dict(self, dictionary):
        for (key, value) in dictionary.items():
            if key in self.__dict__:
                if isinstance(value, Serializable):
                    self.__dict__[key].from_config(dictionary)
                else:
                    self.__dict__[key] = value


def serialize(obj):
    if hasattr(obj, "config"):
        d = obj.config
    elif isinstance(obj, Serializable):
        d = obj.to_dict()
    else:
        d = {key: repr(value) for (key, value) in obj.__dict__.items()}
    d['__class__'] = repr(obj.__class__)
    if isinstance(obj, Env):
        d['id'] = obj.spec.id
        d['import_module'] = getattr(obj, "import_module", None)
    return d
