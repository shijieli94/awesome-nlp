import argparse
import importlib
import logging
import os
import sys
import time
from pathlib import Path

import torch
from colorama import Fore, Style


class ErrorFilter(logging.Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


class ColorFormatter(logging.Formatter):
    colors = {
        "DEBUG": Fore.LIGHTBLACK_EX,
        "INFO": Style.RESET_ALL,  # key the info style
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_fmt = self.colors.get(record.levelname) + self._fmt
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def prepare_global_logging(level=logging.INFO) -> None:
    root_logger = logging.getLogger()

    # create handlers
    formatter = ColorFormatter("%(levelname)s | %(asctime)s | %(name)s:%(lineno)d | %(message)s")
    stdout_handler: logging.Handler = logging.StreamHandler(sys.stdout)
    stderr_handler: logging.Handler = logging.StreamHandler(sys.stderr)

    handler: logging.Handler
    for handler in [stdout_handler, stderr_handler]:
        handler.setFormatter(formatter)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    stdout_handler.setLevel(level)
    stdout_handler.addFilter(ErrorFilter())  # Make sure errors only go to stderr
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(level)

    # put all the handlers on the root logger
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


def import_module_and_submodules(package_name: str) -> None:
    """Import all submodules under the given package."""
    module = importlib.import_module(package_name)

    path = getattr(module, "__path__", [])
    path_string = "" if not path else path[0]

    if path_string:
        exclude = set()
        all_files = os.listdir(path_string)
        if ".gitignore" in all_files:
            with open(os.path.join(path_string, ".gitignore")) as ignore:
                for f in ignore.readlines():
                    assert f.startswith("/"), "ignored folders/files in awesome_nlp should start with `/`"
                    exclude.add(f[1:])

        for file in filter(lambda x: not (x in exclude or x.startswith(("_", "."))), all_files):
            name = file[: file.find(".py")] if file.endswith(".py") else file
            filepath = os.path.join(path_string, name)
            if os.path.isdir(filepath) or file.endswith(".py"):
                subpackage = f"{package_name}.{name}"
                import_module_and_submodules(subpackage)


def main() -> None:
    base_path = Path(".").resolve()

    # ensure `base_path` is the first in `sys.path`
    if str(base_path) not in sys.path:
        sys.path.insert(0, str(base_path))
    else:
        sys.path.insert(0, sys.path.pop(sys.path.index(str(base_path))))

    tik = time.time()
    importlib.invalidate_caches()
    import_module_and_submodules("awesome_nlp")

    import awesome_nlp

    prepare_global_logging(level=logging.DEBUG if awesome_nlp.DEBUG_MODE else logging.INFO)

    logger = logging.getLogger("awesome_nlp")
    logger.info(f"awesome_nlp imported in {time.time() - tik:.2f} (s)")

    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info("-" * 50)
        total_memory = torch.cuda.get_device_properties(i).total_memory

        logger.info(f"CUDA device:{i} name: {torch.cuda.get_device_name(i)}")
        logger.info(f"CUDA device:{i} total memory: {total_memory / (1024 ** 3):.2f} GB")

    # register command and add model to command
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers()

    for name, command in awesome_nlp.COMMAND_REGISTRY.items():
        # register commands
        command_parser = subparsers.add_parser(name, allow_abbrev=False)
        command_subparsers = None

        for model, tasks in awesome_nlp.MODEL_TASK_REGISTRY.items():
            assert "." in model  # must belong to a specific command
            parent, model = model.rsplit(".", maxsplit=1)
            if parent != name:
                continue

            if model != "dummy":
                if command_subparsers is None:
                    command_subparsers = command_parser.add_subparsers()
                model_parser = command_subparsers.add_parser(model, allow_abbrev=False)
            else:
                model_parser = command_parser

            if len(tasks) == 1 and "dummy" in tasks:
                model_parser.set_defaults(
                    command=command, task_cls=tasks["dummy"], command_name=name, model_name="dummy", task_name="dummy"
                )
            else:
                model_subparsers = model_parser.add_subparsers()
                for task, cls in tasks.items():
                    task_parser = model_subparsers.add_parser(task, allow_abbrev=False)
                    task_parser.set_defaults(
                        command=command, task_cls=cls, command_name=name, model_name=model, task_name=task
                    )

    parsed, remaining_args = parser.parse_known_args(sys.argv[1:])

    parsed_task = parsed.task_cls(parsed.command_name, parsed.model_name, parsed.task_name)
    parsed_configs = getattr(parsed_task, parsed.command.config)

    # first register all default configs, therefore we can override them later
    awesome_nlp.CONFIG_REGISTRY.update(parsed_configs, _verbose=False)

    # register system arguments
    with awesome_nlp.Headline("SYSTEM CONFIGS", level=3):
        awesome_nlp.CONFIG_REGISTRY.from_args_list(remaining_args)

    parsed_task.run_task(parsed.command.main)


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv(dotenv_path=".env", override=True)

    main()
