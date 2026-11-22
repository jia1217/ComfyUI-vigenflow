"""Gets what the node needs on first use, so users only clone this repo:

- the VigenFlow engine: the pinned release zip from GitHub (models.json "engine"), checked
  against its SHA256, of which only the text-to-image workers (exe_models/<id>/), their NPU
  files and the weights downloader (prepare/) are unpacked into engine/<version>/;
- a model's weights: the release's prepare_base.exe downloads the pre-packed NPU weights from
  Hugging Face into ComfyUI/models/vigenflow/<id>/ the first time that model is used. Krea 2 and
  Qwen-Image weights are under their own licenses and download only after the user accepts them
  in vigenflow_settings.json.

No ComfyUI imports here. Set VIGENFLOW_ENGINE_ZIP to a local copy of the release zip to install
without downloading it (the SHA256 is still checked).
"""
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from contextlib import contextmanager

try:
    from . import worker as npu_worker
except ImportError:   # run as a plain module (install.py)
    import worker as npu_worker

LORA_PREFIX = "../../../../all_model_weights/"   # LoRA paths in the release's loras.jsonc
COMPLETE = ".complete"                            # written last: the engine folder is whole


class InstallError(npu_worker.WorkerError):
    pass


def _say(status, text):
    print(f"[VigenFlow] {text}", file=sys.stderr, flush=True)
    if status:
        status(text)


@contextmanager
def _lock(folder):
    """One installer at a time (ComfyUI, install.py and a second ComfyUI may overlap)."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------- system check

def check_system():
    """Problems that stop the engine from running on this machine (empty when it can run)."""
    problems = []
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        problems.append(f"VigenFlow for ComfyUI needs Linux on x86-64 (this is {platform.system()} "
                        f"{platform.machine()})")
    if not os.path.exists("/dev/accel/accel0"):
        problems.append("No AMD NPU found (/dev/accel/accel0). It needs a Ryzen AI processor with an "
                        "XDNA 2 NPU and the amdxdna driver")
    return problems


def missing_libraries(exe):
    """Shared libraries run.exe needs but the system lacks, with install hints."""
    try:
        out = subprocess.run(["ldd", exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    missing = sorted({m.group(1) for m in re.finditer(r"^\s*(\S+) => not found", out, re.M)})
    hints = []
    for lib in missing:
        if "boost_program_options" in lib:
            hints.append(f"{lib} (sudo apt install libboost-program-options1.83.0)")
        elif "xrt" in lib:
            hints.append(f"{lib} (install AMD XRT and the NPU driver; then source /opt/xilinx/xrt/setup.sh)")
        else:
            hints.append(lib)
    return hints


# ---------------------------------------------------------------------------- engine

def _engine_parts(settings):
    """Zip member prefixes to unpack: each engine model's worker and NPU files, and prepare/."""
    root = settings["engine"]["archive_root"]
    parts = [f"{root}/prepare/"]
    for spec in settings["models"].values():
        if spec["engine"]:
            parts.append(f"{root}/server_running/exe_models/{spec['id']}/")
            if spec["npu_files_path"]:
                rel = os.path.relpath(spec["npu_files_path"], settings["engine"]["dir"])
                parts.append(f"{root}/{rel}/")
    return parts


def engine_ready(settings=None):
    """The engine folder is whole and has every engine model of models.json (a newer node can
    add models: then the engine is unpacked again)."""
    settings = settings or npu_worker.load_settings()
    if not os.path.isfile(os.path.join(settings["engine"]["dir"], COMPLETE)):
        return False
    return all(os.path.isfile(os.path.join(spec["host_dir"], "run.exe")) and
               (not spec["npu_files_path"] or os.path.isdir(spec["npu_files_path"]))
               for spec in settings["models"].values() if spec["engine"])


def _download(url, target, status):
    request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-vigenflow"})
    with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                share = f" ({100 * done // total} %)" if total else ""
                _say(status, f"Downloading the VigenFlow engine: {done >> 20} MB{share}")


def _sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_engine(status=None):
    """Install the engine release if it is not there yet; fix the LoRA paths either way."""
    settings = npu_worker.load_settings()
    engine = settings["engine"]
    folder = engine["dir"]
    if not engine_ready(settings):
        with _lock(os.path.dirname(folder)):
            if not engine_ready(settings):
                _install_engine(settings, status)
    write_lora_configs(settings)
    return folder


def _install_engine(settings, status):
    engine = settings["engine"]
    folder = engine["dir"]
    local = os.environ.get("VIGENFLOW_ENGINE_ZIP")
    with tempfile.TemporaryDirectory(prefix="vigenflow_engine_", dir=os.path.dirname(folder)) as tmp:
        archive = os.path.join(tmp, "engine.zip")
        if local:
            _say(status, f"Using the VigenFlow engine zip {local}")
            shutil.copyfile(local, archive)
        else:
            _say(status, f"Downloading the VigenFlow engine {engine['version']} (about 35 MB)")
            try:
                _download(engine["url"], archive, status)
            except OSError as exc:
                raise InstallError(f"Cannot download the VigenFlow engine from {engine['url']}: {exc}. "
                                   "Download it yourself and set VIGENFLOW_ENGINE_ZIP to the file") from exc
        digest = _sha256(archive)
        if digest != engine["sha256"]:
            raise InstallError(f"The VigenFlow engine zip has SHA256 {digest}, expected {engine['sha256']}; "
                               "not installed")
        staging = os.path.join(tmp, "engine")
        parts = _engine_parts(settings)
        root = engine["archive_root"] + "/"
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                name = info.filename
                if not any(name.startswith(p) for p in parts):
                    continue
                rel = name[len(root):]
                rel = rel[len("server_running/"):] if rel.startswith("server_running/") else rel
                target = os.path.normpath(os.path.join(staging, rel))
                if not target.startswith(staging + os.sep):
                    raise InstallError(f"Unexpected path in the engine zip: {name}")
                if info.is_dir():
                    os.makedirs(target, exist_ok=True)
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                mode = (info.external_attr >> 16) & 0o777
                if mode:
                    os.chmod(target, mode)
        for spec in settings["models"].values():   # keep the release's LoRA list as shipped
            if spec["engine"]:
                config = os.path.join(staging, "exe_models", spec["id"], "loras.jsonc")
                if os.path.isfile(config):
                    os.replace(config, config.replace("loras.jsonc", "loras.release.jsonc"))
        open(os.path.join(staging, COMPLETE), "w").write(engine["version"] + "\n")
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(os.path.dirname(folder), exist_ok=True)
        os.replace(staging, folder)
    _say(status, f"VigenFlow engine {engine['version']} installed in {folder}")


def read_user_loras(spec):
    """(hide, entries) from ComfyUI/models/loras/<id>/loras.jsonc, the user's own LoRA list:
    "hide" names engine LoRAs to leave out, "loras" adds entries (an entry with an engine
    LoRA's name replaces it). "file" is relative to that folder. ((), []) without the file;
    a broken file is reported in the console and ignored."""
    filename = npu_worker.user_lora_file(spec["weights_folder"])
    if not os.path.isfile(filename):
        return set(), []
    try:
        data = npu_worker._read_jsonc(filename)
        hide = data.get("hide", [])
        entries = data.get("loras", [])
        if not isinstance(hide, list) or not all(isinstance(h, str) for h in hide):
            raise ValueError('"hide" must be a list of names')
        if not isinstance(entries, list):
            raise ValueError('"loras" must be a list')
        folder = os.path.dirname(filename)
        result = []
        for e in entries:
            if not isinstance(e, dict) or not e.get("name") or not isinstance(e.get("file"), str):
                raise ValueError('each entry in "loras" needs a "name" and a "file"')
            if e["name"] == "none":
                raise ValueError('"none" cannot be a LoRA name')
            result.append(dict(e, file=os.path.normpath(os.path.join(folder, os.path.expanduser(e["file"])))))
        return set(hide), result
    except (OSError, ValueError, AttributeError) as exc:
        print(f"[VigenFlow] Ignoring {filename}: {exc}", file=sys.stderr, flush=True)
        return set(), []


def write_lora_configs(settings=None):
    """loras.jsonc for each engine worker: the release list (its LoRA paths pointing to
    <weights_dir>/<id>/loras/; the release expects all_model_weights/ beside its package),
    minus the user's hidden names, plus the user's own entries. Paths are absolute."""
    settings = settings or npu_worker.load_settings()
    for spec in settings["models"].values():
        if not spec["engine"]:
            continue
        release = os.path.join(spec["host_dir"], "loras.release.jsonc")
        if not os.path.isfile(release):
            continue
        data = npu_worker._read_jsonc(release)
        engine_entries = []
        for e in data.get("loras", []):
            file = e.get("file", "")
            if file.startswith(LORA_PREFIX):
                file = os.path.join(settings["weights_dir"], file[len(LORA_PREFIX):])
            engine_entries.append(dict(e, file=os.path.normpath(os.path.join(spec["host_dir"], file))))
        hide, own = read_user_loras(spec)
        replaced = {e["name"] for e in own}
        data["loras"] = [e for e in engine_entries if e.get("name") not in hide | replaced] + own
        text = "// Written by ComfyUI-vigenflow from loras.release.jsonc and\n" \
               f"// {npu_worker.user_lora_file(spec['weights_folder'])} (edit that file, not this one).\n" + \
               json.dumps(data, indent=2) + "\n"
        config = os.path.join(spec["host_dir"], "loras.jsonc")
        if not os.path.isfile(config) or open(config, encoding="utf-8").read() != text:
            with open(config, "w", encoding="utf-8") as file:
                file.write(text)


# ---------------------------------------------------------------------------- weights

def weights_ready(spec):
    folder = spec["weights_path"]
    if not folder or not os.path.isdir(folder):
        return False
    if spec["weights_given"] or not spec["engine"]:
        return any(os.scandir(folder))   # the user's own folder: trust it
    return os.path.isfile(os.path.join(folder, "prepare_manifest.json"))


def ensure_weights(spec, status=None, should_cancel=None, accept_license=False):
    """Download a model's weights with the release's prepare_base.exe on first use. A model under its
    own license downloads only once it is accepted: accept_license (the node's switch) or its id in
    accept_licenses of vigenflow_settings.json."""
    if weights_ready(spec):
        return spec["weights_path"]
    if spec["weights_given"] or not spec["engine"]:
        raise InstallError(f"{spec['name']}: no weights in {spec['weights_path']}")
    settings = npu_worker.load_settings()
    license_ = spec.get("license")
    if license_ and not accept_license and spec["weights_folder"] not in settings["accept_licenses"]:
        raise InstallError(
            f"{spec['name']} weights are under the {license_['name']} ({license_['url']}). "
            "To accept it and download them, turn on accept_license on the node (it means \"I agree\"), "
            f"or add \"{spec['weights_folder']}\" to \"accept_licenses\" in "
            f"{os.path.join(npu_worker.NODE_DIR, npu_worker.SETTINGS_FILE)} and restart ComfyUI.")
    prepare = os.path.join(settings["engine"]["dir"], "prepare")
    weights_dir = os.path.dirname(spec["weights_path"])
    os.makedirs(weights_dir, exist_ok=True)
    cmd = [os.path.join(prepare, "prepare_base.exe"), "--config", os.path.join(prepare, "prepare_models.jsonc"),
           "--weights_root", weights_dir, "--only", spec["weights_folder"]]
    if license_:
        cmd.append("--accept_license")   # accepted in vigenflow_settings.json
    _say(status, f"Downloading the {spec['name']} weights into {spec['weights_path']} "
                 "(first use; several GB, this takes a while)")
    with _lock(weights_dir):
        if weights_ready(spec):
            return spec["weights_path"]
        proc = subprocess.Popen(cmd, cwd=prepare, stdin=subprocess.DEVNULL)
        while proc.poll() is None:
            if should_cancel is not None and should_cancel():
                proc.kill()
                proc.wait()
                raise npu_worker.Cancelled()
            time.sleep(0.5)
    if proc.returncode != 0 or not weights_ready(spec):
        raise InstallError(f"Downloading the {spec['name']} weights failed (prepare_base.exe exit "
                           f"{proc.returncode}); see the console above")
    _say(status, f"{spec['name']} weights ready")
    return spec["weights_path"]


def ensure_model(spec, status=None, should_cancel=None, accept_license=False):
    """Everything one model needs before its worker starts."""
    problems = check_system()
    if problems:
        raise InstallError("; ".join(problems))
    if spec["engine"]:
        ensure_engine(status)
        missing = missing_libraries(os.path.join(spec["host_dir"], "run.exe"))
        if missing:
            raise InstallError("The VigenFlow engine needs these libraries: " + ", ".join(missing))
    ensure_weights(spec, status, should_cancel, accept_license)
