from .decoding import CONFIG_MAP


def get_device():
    return CONFIG_MAP.get("LOCAL_RANK", 0)


def distributed():
    return CONFIG_MAP.get("DIST_WORKERS", 1) > 1
