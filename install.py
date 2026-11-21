"""Run by ComfyUI-Manager after it clones this node (or by hand: python install.py).

Checks the machine and installs the VigenFlow engine into engine/<version>/. Model weights are
downloaded later, the first time each model is used. Never fails the ComfyUI-Manager install:
problems are printed, and the node reports them again when it is used.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import installer  # noqa: E402


def main():
    problems = installer.check_system()
    if problems:
        print("[VigenFlow] This machine cannot run the VigenFlow engine yet:", *problems, sep="\n  ")
        return
    try:
        installer.ensure_engine()
    except Exception as exc:  # noqa: BLE001 - report, do not fail the install
        print(f"[VigenFlow] Engine not installed: {exc}\n[VigenFlow] It is retried when the node is first used.")
        return
    settings = installer.npu_worker.load_settings()
    exe = os.path.join(settings["engine"]["dir"], "exe_models", "flux2_klein_4B", "run.exe")
    missing = installer.missing_libraries(exe)
    if missing:
        print("[VigenFlow] Install these libraries before generating:", *missing, sep="\n  ")
    print(f"[VigenFlow] Engine ready. Model weights download into {settings['weights_dir']} on first use.")


if __name__ == "__main__":
    main()
