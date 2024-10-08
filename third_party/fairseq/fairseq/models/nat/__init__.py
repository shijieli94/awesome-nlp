import importlib
import os

# automatically import any Python files in the models/ema/ directory
for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("fairseq.models.nat." + file_name)
