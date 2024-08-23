__author__ = "SHIJIE_LI"
__email__ = "lsj72123@gmail.com"

import logging
import os
from collections import defaultdict, namedtuple
from itertools import product
from pathlib import Path
from typing import List, Optional, Set, Union

from overrides import overrides

logger = logging.getLogger(__name__)

# FLAGS
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

DEBUG_MODE = os.environ.get("AWESOMENLP_DEBUG", "").upper() in ENV_VARS_TRUE_VALUES
WANDB_DISABLED = os.environ.get("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES

# PATHS
PROJECT_DIR = Path(__file__).parents[1]

EXPERIMENT_DIR: str = Path(PROJECT_DIR, "experiments").as_posix()
DATASETS_DIR: str = Path(Path.home(), "Data", ".cache", "datasets").as_posix()
TRANSFORMERS_DIR: str = Path(Path.home(), "Data", ".cache", "transformers").as_posix()


# ---------------------------------------- REGISTRIES ----------------------------------------#
def register_command(name, config: str = "config"):
    COMMAND = namedtuple("COMMAND", ["main", "config"])

    def wrapper(func):
        if name in COMMAND_REGISTRY:
            raise ValueError(f"Cannot register duplicate command ({name})")
        COMMAND_REGISTRY[name] = COMMAND(main=func, config=config)
        return func

    return wrapper


def register_models_and_tasks(*tasks, models: Union[str, List[str]], commands: Union[str, List[str]]):
    if isinstance(models, str):
        models = [models]

    if isinstance(commands, str):
        commands = [commands]

    commanded_models = list(".".join(cm) for cm in product(commands, models))

    def wrapper(cls):
        for m in commanded_models:
            for t in tasks:
                if t in MODEL_TASK_REGISTRY[m]:
                    raise ValueError(f"Cannot register duplicate task ({t}) for model ({m})")
                MODEL_TASK_REGISTRY[m][t] = cls
        return cls

    return wrapper


class ConfigRegistry(dict):
    @overrides
    def update(self, configs, _verbose=True, _level="info"):
        for name, value in configs.items():
            if name in self:
                if value == self[name]:
                    message = f"unchanged: {name} {value}"
                else:
                    message = f"modified: {name} {self[name]} -> {value}"
                    self[name] = value
            else:
                message = f"add: {name} -> {value}"
                self[name] = value

            if _verbose:
                getattr(logger, _level)(message)

    def quiet_update(self, configs):
        self.update(configs, _verbose=False)

    def from_args_list(self, args_list, _verbose=True):
        if len(args_list) == 0:
            return

        for key, val in zip(args_list, args_list[1:]):
            if not key.startswith("--"):
                continue

            # if key.startswith("--") and val.startswith("--"), it is a bool argument
            if val.startswith("--"):
                self.update({key: True}, _verbose=_verbose)

            # use false to cancel some default True bool settings
            elif val.lower() == "_false_":
                self.update({key: False}, _verbose=_verbose)

            # normal argument
            else:
                self.update({key: val}, _verbose=_verbose)

        # take care of the last argument, which is not a key-value pair
        if args_list[-1].startswith("--"):
            self.update({args_list[-1]: True}, _verbose=_verbose)

    def to_args_list(self):
        args_list, positional = [], []
        for key, val in self.items():
            if val in [False, None]:
                continue

            if val in [True]:
                args_list += [key]
                continue

            if key == "_positional_args_":
                positional = val
                continue

            args_list += [key, val]

        return positional + args_list

    def parse_group_configs(self, required_fields=None, included=None, excluded=None):
        """
        we use symbol `::` to define configs that belong to a specific group and `@` to define the config type.
        For example, {`--eval-bleu-args::int@beam`: 1} registers a config beam=int(1) to group `--eval-bleu-args`
        required_fields: must include configs
        included: optional include configs
        """

        convert_to_set = lambda x: set() if x is None else {x} if isinstance(x, str) else set(x)

        required_fields = convert_to_set(required_fields)

        if included is not None and excluded is not None:
            raise ValueError("included and excluded cannot set together.")

        included = tuple(convert_to_set(included))
        excluded = tuple(convert_to_set(excluded))

        if included:
            filtered_keys = list(k for k in self.keys() if k.startswith(included))
        elif excluded:
            filtered_keys = list(k for k in self.keys() if not k.startswith(excluded))
        else:
            filtered_keys = list(self.keys())

        group_configs = defaultdict(dict)
        for key in filter(lambda k: "::" in k, filtered_keys):
            g_name, g_key = key.split("::")
            if "@" in g_key:
                g_key, g_type = g_key.split("@")
            else:
                g_type = "str"
            g_val = self.pop(key)  # drop it as it is not a standard config format

            # convert to real value before dumping
            g_val = eval(g_type)(g_val)

            group_configs[g_name][g_key] = g_val

        for field in group_configs.keys():
            if field in required_fields:
                required_fields.discard(field)

        if len(required_fields) > 0:
            raise ValueError(f"Missing required config group(s) ({required_fields}).")

        return group_configs

    def safe_hasattr(self, attr):
        """Returns True if the given key exists and is not None."""
        return attr in self and self[attr] is not None

    def safe_eq(self, attr, value):
        return attr in self and type(value)(self[attr]) == value

    def safe_ne(self, attr, value):
        return attr in self and type(value)(self[attr]) != value

    def safe_gt(self, attr, value):
        return attr in self and type(value)(self[attr]) > value

    def safe_lt(self, attr, value):
        return attr in self and type(value)(self[attr]) < value


COMMAND_REGISTRY = {}
MODEL_TASK_REGISTRY = defaultdict(dict)
CONFIG_REGISTRY = ConfigRegistry()


# ---------------------------------------- BASE CLASS ----------------------------------------#
class BaseTask:
    """Base Config for all Tasks"""

    def __init__(self, command, model, task):
        self.command = command
        self.model = model
        self.task = task

    def post_process_configs(self):
        pass

    def run_task(self, run_main):
        Headline.log("POST-PROCESSING CONFIGS", level=3)
        kwargs = self.post_process_configs()

        # after post process, there should be no more config group
        group_configs = CONFIG_REGISTRY.parse_group_configs()
        assert len(group_configs) == 0, f"Found unparsed group configs {group_configs.keys()} after post-processing."

        kwargs = {} if kwargs is None else kwargs
        return run_main(CONFIG_REGISTRY.to_args_list(), **kwargs)


# ---------------------------------------- TOOLS ----------------------------------------#
def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        if len(x) == 0:
            return []
        x = eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


class Headline:
    default_width = 20

    def __init__(self, msg, level=1, log_level="info"):
        self.msg = msg
        self.level = level
        self.log_level = log_level

    @staticmethod
    def _log(msg, level, log_func):
        total_width = max(Headline.default_width * level, len(msg) + 4)  # the longest case is - msg -
        side_width = (total_width - len(msg) - 2) // 2

        msg_formatted = "-" * side_width + f" {msg} " + "-" * side_width
        if len(msg_formatted) < total_width:
            msg_formatted += "-"

        log_func(msg_formatted)

        return msg_formatted

    @staticmethod
    def log(msg, level=1, log_level="info"):
        # make sure level is valid
        log_func = getattr(logger, log_level, logger.info)
        return Headline._log(msg, level, log_func)

    def __enter__(self):
        self.log_func = getattr(logger, self.log_level, logger.info)
        self.msg_formatted = Headline._log(self.msg, self.level, self.log_func)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log_func("*" * len(self.msg_formatted))
