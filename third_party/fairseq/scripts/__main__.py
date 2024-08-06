import os
import subprocess
import sys


def cli_main():
    root_dir = os.path.dirname(os.path.abspath(__file__))

    command_str = f"{sys.executable} {os.path.join(root_dir, sys.argv[1])} {' '.join(sys.argv[2:])}"

    try:
        result = subprocess.run(
            command_str, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)


if __name__ == "__main__":
    cli_main()
