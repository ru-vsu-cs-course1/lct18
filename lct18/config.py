from pathlib import Path
from typing import Any, Callable
import yaml

from attrdictionary import AttrDict


pwd = Path(__file__).parents[1].absolute()
filename = pwd / 'config.yml'

with filename.open() as stream:
    config = yaml.safe_load(stream)


# Не берем из utils, т.к. возникают циклические ссылки
def deep_update(v: Any, updater: Callable[[Any], Any]) -> Any:
    if isinstance(v, dict):
        return {k: deep_update(v, updater) for k, v in v.items()}
    if isinstance(v, list):
        return [deep_update(v, updater) for v in v]
    return updater(v)


# Не берем из utils, т.к. возникают циклические ссылки
def str_key_replace(s: str, key: str, value: Any):
    value = str(value)
    s = s.replace('${' + key + '}', value)
    s = s.replace('$' + key, value)
    s = s.replace('{' + key + '}', value)
    return s


def config_value_updater(v: Any) -> Any:
    if isinstance(v, str):
        return str_key_replace(v, 'pwd', pwd)
    return v


config = deep_update(config, config_value_updater)
config = AttrDict(config)
