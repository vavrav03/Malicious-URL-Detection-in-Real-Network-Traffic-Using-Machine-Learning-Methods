import sys

MACOS_PLATFORM = "darwin"


def is_local_development():
    return sys.platform == MACOS_PLATFORM
