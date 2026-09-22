#!/usr/bin/env python3
"""
Single entry point for laya-snake.

Run it as:
    python3 play.py         (Linux / macOS)
    python play.py          (Windows)
or via the platform launcher: ./run.sh  /  run.bat

It detects your OS, creates an isolated virtual environment next to this
file (so it never touches your system Python), installs the few packages
needed (torch + laya, plus windows-curses on Windows), lets `laya.load()`
download the model checkpoint from Hugging Face on first run (cached after
that), and then starts the live Snake game. Nothing here needs to be
installed by hand first — just a working Python 3.9+.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, ".venv")
IS_WINDOWS = platform.system() == "Windows"
BOOTSTRAP_MARKER = os.path.join(VENV_DIR, ".deps_ok")
REENTRY_FLAG = "_LAYA_SNAKE_IN_VENV"

REQUIRED_PACKAGES = ["laya", "huggingface_hub"]
if IS_WINDOWS:
    # stdlib `curses` doesn't exist on Windows; this package provides it.
    REQUIRED_PACKAGES.append("windows-curses")


def venv_python():
    scripts_dir = "Scripts" if IS_WINDOWS else "bin"
    candidates = ["python.exe"] if IS_WINDOWS else ["python3", "python"]
    for name in candidates:
        path = os.path.join(VENV_DIR, scripts_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(VENV_DIR, scripts_dir, candidates[0])


def create_venv():
    print(f"[setup] creating a virtual environment at {VENV_DIR} (one-time)...")
    try:
        subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])
    except subprocess.CalledProcessError as exc:
        print("[setup] failed to create a virtual environment.")
        if not IS_WINDOWS:
            print("        On Debian/Ubuntu you may need: sudo apt install python3-venv")
        raise SystemExit(1) from exc


def install_deps():
    print("[setup] installing dependencies (torch + laya) — first run only, "
          "this can take a few minutes...")
    py = venv_python()
    # pip's build/download temp files default to the system temp dir, which
    # can be a small quota-limited tmpfs (containers, some sandboxes). Route
    # them next to the venv instead so a small /tmp can't fail the install.
    pip_tmp = os.path.join(VENV_DIR, ".pip-tmp")
    os.makedirs(pip_tmp, exist_ok=True)
    env = dict(os.environ)
    env["TMPDIR"] = pip_tmp
    env["TEMP"] = pip_tmp
    env["TMP"] = pip_tmp
    try:
        subprocess.check_call([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], env=env)
        subprocess.check_call([py, "-m", "pip", "install", "--quiet", *REQUIRED_PACKAGES], env=env)
    except subprocess.CalledProcessError as exc:
        print("[setup] dependency installation failed. Re-run this command to retry.")
        raise SystemExit(1) from exc
    finally:
        shutil.rmtree(pip_tmp, ignore_errors=True)
    with open(BOOTSTRAP_MARKER, "w") as f:
        f.write("ok")


def bootstrap_and_reexec(argv):
    if sys.version_info < (3, 9):
        print(f"Python 3.9+ is required (found {platform.python_version()}).")
        raise SystemExit(1)

    if not os.path.exists(venv_python()):
        create_venv()
    if not os.path.exists(BOOTSTRAP_MARKER):
        install_deps()

    env = dict(os.environ)
    env[REENTRY_FLAG] = "1"
    ret = subprocess.call([venv_python(), os.path.abspath(__file__), *argv], env=env)
    raise SystemExit(ret)


def run_game(argv):
    sys.path.insert(0, HERE)
    from laya_snake.app import main as game_main

    sys.argv = [sys.argv[0], *argv]
    game_main()


def main():
    argv = sys.argv[1:]
    if os.environ.get(REENTRY_FLAG) != "1":
        bootstrap_and_reexec(argv)
        return  # unreachable; bootstrap_and_reexec always exits
    run_game(argv)


if __name__ == "__main__":
    main()
