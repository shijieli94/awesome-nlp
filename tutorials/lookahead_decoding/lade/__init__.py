from typing import Optional

import dotenv


# load env before load packages
def load_envs(env_file: Optional[str] = None) -> None:
    """Load all the environment variables defined in the `env_file`.
    This is equivalent to `. env_file` in bash.
    It is possible to define all the system specific variables in the `env_file`.
    :param env_file: the file that defines the environment variables to use. If None
                     it searches for a `.env` file in the project.
    """
    dotenv.load_dotenv(dotenv_path=env_file, override=True)


# Load environment variables
load_envs()

from .lade_distributed import distributed, get_device
from .utils import (
    augment_all,
    augment_generate,
    augment_llama,
    config_lade,
    log_history,
    save_log,
)
