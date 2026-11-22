"""Client for `run.exe --mode performance --worker` (one JSON line per image on stdin/stdout).

Every VigenFlow text-to-image host (FLUX.2 Klein, FLUX.1 Schnell, Krea 2, Qwen-Image,
Z-Image, ...) speaks the same protocol, so one client drives them all. The models the
node offers are in models.json (and the optional vigenflow_settings.json).

No ComfyUI or torch imports here, so it can be tested alone:
    python3 worker.py --model "FLUX.2 Klein 4B" --prompt "a red fox" --output /tmp/fox.png
"""
import atexit
import json
import os
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache


class WorkerError(RuntimeError):
    pass


class Cancelled(Exception):
    pass


NODE_DIR = os.path.dirname(os.path.realpath(__file__))
SETTINGS_FILE = "vigenflow_settings.json"   # optional, next to this file; not in git


def default_host_dir():
    """This node's folder: models.json and the optional vigenflow_settings.json are there."""
    return NODE_DIR


def comfy_models_dir():
    """ComfyUI's models folder (from ComfyUI when it is running, else ../../models)."""
    try:
        import folder_paths
        return folder_paths.models_dir
    except ImportError:
        return os.path.abspath(os.path.join(NODE_DIR, "..", "..", "models"))


def _path(base, value, field, filename):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkerError(f"{field} in {filename} must be a path string or null")
    return os.path.abspath(os.path.join(base, os.path.expanduser(value)))


def _steps(name, entry, filename):
    """(default, allowed list or None, min, max) from a model entry."""
    default = entry.get("steps", 4)
    allowed = entry.get("allowed_steps")
    lo, hi = entry.get("min_steps", 1), entry.get("max_steps", default)
    ints = [default, lo, hi] + (allowed if isinstance(allowed, list) else [])
    if allowed is not None and (not isinstance(allowed, list) or not allowed):
        raise WorkerError(f"models.{name}.allowed_steps in {filename} must be a non-empty list")
    if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in ints):
        raise WorkerError(f"steps fields of model '{name}' in {filename} must be positive integers")
    if allowed is not None:
        if default not in allowed:
            raise WorkerError(f"models.{name}.steps in {filename} must be one of allowed_steps")
        return default, sorted(allowed), min(allowed), max(allowed)
    if not lo <= default <= hi:
        raise WorkerError(f"models.{name}: need min_steps <= steps <= max_steps in {filename}")
    return default, None, lo, hi


def _read_json(filename, required):
    try:
        with open(filename, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        if required:
            raise WorkerError(f"{filename} is missing; reinstall ComfyUI-vigenflow") from None
        return {}
    except (OSError, ValueError) as exc:
        raise WorkerError(f"Cannot read {filename}: {exc}") from exc
    if not isinstance(data, dict):
        raise WorkerError(f"{filename} must contain a JSON object")
    return data


@lru_cache(maxsize=1)
def load_settings(host_dir=NODE_DIR):
    """models.json (the engine release and the built-in models) merged with the optional
    vigenflow_settings.json. Returns output_dir, engine (version, url, sha256, archive_root,
    dir), weights_dir, accept_licenses and models (name -> spec with absolute paths)."""
    builtin_file = os.path.join(host_dir, "models.json")
    user_file = os.path.join(host_dir, SETTINGS_FILE)
    builtin, user = _read_json(builtin_file, True), _read_json(user_file, False)

    engine = dict(builtin["engine"])
    engine["dir"] = (_path(host_dir, user.get("engine_dir"), "engine_dir", user_file)
                     or os.path.join(host_dir, "engine", engine["version"]))
    weights_dir = (_path(host_dir, user.get("weights_dir"), "weights_dir", user_file)
                   or os.path.join(comfy_models_dir(), "vigenflow"))
    accepted = user.get("accept_licenses", [])
    if not isinstance(accepted, list):
        raise WorkerError(f"accept_licenses in {user_file} must be a list of model ids")

    entries = {name: dict(entry) for name, entry in builtin["models"].items()}
    user_models = user.get("models", {})
    if not isinstance(user_models, dict):
        raise WorkerError(f"models in {user_file} must be an object of model name -> settings")
    for name, override in user_models.items():   # change a built-in model or add one
        if not isinstance(override, dict):
            raise WorkerError(f"models.{name} in {user_file} must be an object")
        entries.setdefault(name, {}).update(override)

    specs = {}
    for name, entry in entries.items():
        if entry.get("enabled", True) is False:
            continue
        filename = user_file if name in user_models else builtin_file
        model_id = entry.get("id")
        if entry.get("host_dir"):     # a model from elsewhere: paths relative to this folder
            host = _path(host_dir, entry["host_dir"], f"models.{name}.host_dir", filename)
            npu = _path(host_dir, entry.get("npu_files_path"), f"models.{name}.npu_files_path", filename)
        elif model_id:                # a model of the engine release
            host = os.path.join(engine["dir"], "exe_models", model_id)
            npu = os.path.join(engine["dir"], entry["npu_files"]) if entry.get("npu_files") else None
        else:
            raise WorkerError(f"models.{name} in {filename} needs an id (engine model) or a host_dir")
        weights = _path(host_dir, entry.get("weights_path"), f"models.{name}.weights_path", filename)
        weights_id = entry.get("weights_id") or model_id   # an edit model uses its base model's weights
        kind = entry.get("kind", "text_to_image")
        if kind not in ("text_to_image", "edit"):
            raise WorkerError(f"models.{name}.kind in {filename} must be text_to_image or edit")
        default, allowed, lo, hi = _steps(name, entry, filename)
        specs[name] = {
            "name": name, "id": model_id, "engine": not entry.get("host_dir"), "kind": kind,
            "max_images": int(entry.get("max_images", 1)) if kind == "edit" else 0,
            "reference_token_limit": int(entry.get("reference_token_limit", 0)),   # 0: no budget
            "worker": entry.get("worker", True),   # false: run.exe has no worker mode (one run per image)
            "host_dir": host, "npu_files_path": npu,
            "weights_path": weights or (os.path.join(weights_dir, weights_id) if weights_id else None),
            "weights_given": weights is not None,
            "weights_folder": weights_id, "license": entry.get("license"),
            "steps": default, "allowed_steps": allowed, "min_steps": lo, "max_steps": hi,
        }
    return {"output_dir": _path(host_dir, user.get("output_dir"), "output_dir", user_file),
            "engine": engine, "weights_dir": weights_dir, "accept_licenses": accepted, "models": specs}


def available_models(host_dir=NODE_DIR, kind=None):
    """Configured model names in order, optionally only text_to_image or edit models
    (the engine and weights download on first use)."""
    return [name for name, spec in load_settings(host_dir)["models"].items() if kind in (None, spec["kind"])]


def model_spec(host_dir, model):
    spec = load_settings(host_dir)["models"].get(model)
    if spec is None:
        raise WorkerError(f"Unknown model '{model}'; see models.json and vigenflow_settings.json")
    return spec


def resolve_steps(spec, steps):
    """0 = the model's default; otherwise check against what its host accepts."""
    if not steps:
        return spec["steps"]
    allowed = spec["allowed_steps"]
    if allowed is not None and steps not in allowed:
        raise WorkerError(f"{spec['name']} accepts steps {', '.join(map(str, allowed))} (0 = default {spec['steps']})")
    if not spec["min_steps"] <= steps <= spec["max_steps"]:
        raise WorkerError(f"{spec['name']} accepts steps {spec['min_steps']}..{spec['max_steps']} "
                          f"(0 = default {spec['steps']})")
    return steps


def strip_jsonc(text):
    """JSON with // and /* */ comments and trailing commas -> plain JSON."""
    out, i, n, in_string = [], 0, len(text), False
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif c == '"':
                in_string = False
        elif c == '"':
            in_string = True
            out.append(c)
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(c)
        i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))


def _read_jsonc(filename):
    with open(filename, encoding="utf-8") as f:
        return json.loads(strip_jsonc(f.read()))


def comfy_loras_dir():
    """ComfyUI's LoRA folder (models/loras)."""
    try:
        import folder_paths
        return folder_paths.get_folder_paths("loras")[0]
    except (ImportError, IndexError, KeyError):
        return os.path.join(comfy_models_dir(), "loras")


def user_lora_file(model_id):
    """The user's own LoRA list for a model: ComfyUI/models/loras/<id>/loras.jsonc."""
    return os.path.join(comfy_loras_dir(), model_id, "loras.jsonc")


def lora_entries(model_dir):
    """Entries of a host's loras.jsonc (comments allowed), with "file" made
    absolute the way the host resolves it: from the folder of the .jsonc file."""
    try:
        entries = _read_jsonc(os.path.join(model_dir, "loras.jsonc")).get("loras", [])
    except (OSError, ValueError, TypeError, AttributeError):
        return []
    result = []
    for e in entries:
        if isinstance(e, dict) and e.get("name") and isinstance(e.get("file"), str):
            result.append(dict(e, file=os.path.normpath(os.path.join(model_dir, os.path.expanduser(e["file"])))))
    return result


def lora_names(model_dir):
    """Names in a host's loras.jsonc."""
    return [e["name"] for e in lora_entries(model_dir)]


def write_lora_stack(stack):
    """A LoRA list for --lora_config: one entry per (file, strength, extra) in the stack, named
    stack_1, stack_2, ... so each LoRA keeps its own strength. Returns (path, names)."""
    entries, names = [], []
    for i, (file, strength, extra) in enumerate(stack, 1):
        name = f"stack_{i}"
        entries.append(dict(extra, name=name, file=file, scale=float(strength), enabled=False))
        names.append(name)
    fd, path = tempfile.mkstemp(prefix="vigenflow_loras_", suffix=".jsonc")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"loras": entries}, f, indent=2)
    return path, names


def all_lora_names(host_dir):
    """LoRA names of every available model (each run checks the chosen model's own list)."""
    names = []
    for model in available_models(host_dir):
        for name in lora_names(model_spec(host_dir, model)["host_dir"]):
            if name not in names:
                names.append(name)
    return names


class NPUWorker:
    """One resident run.exe. Model, height, width and LoRA are fixed for its lifetime. A model
    whose run.exe has no worker mode ("worker": false in models.json) runs one run.exe per image
    instead (one_shot), with the same interface."""

    def __init__(self, spec, height, width, lora="none", lora_scale=None, comfy_models_dir=None,
                 lora_stack=None, input_images=None):
        self.spec = spec
        self.name = spec["name"]
        self.key = (spec["name"], spec["host_dir"], height, width, lora, lora_scale, comfy_models_dir,
                    lora_stack)
        self._lora_config = None
        self.one_shot = spec.get("worker") is False
        host_dir = spec["host_dir"]
        exe = os.path.join(host_dir, "run.exe")
        if not os.path.isfile(exe):
            raise WorkerError(f"{self.name}: run.exe not found in {host_dir}; the VigenFlow engine is not installed")
        cmd = [exe] + ([] if self.one_shot else ["--mode", "performance", "--worker"]) + \
              ["--height", str(height), "--width", str(width)]
        if lora_stack:   # Load LoRA nodes: own list, one strength per LoRA, no --lora_scale
            self._lora_config, names = write_lora_stack(lora_stack)
            cmd += ["--lora_config", self._lora_config]
            for name in names:
                cmd += ["--lora", name]
        else:
            if lora is not None:   # None: the host merges the entries enabled in loras.jsonc
                cmd += ["--lora", lora]
            if lora != "none" and lora_scale is not None:
                cmd += ["--lora_scale", str(lora_scale)]
        for image in () if self.one_shot else input_images or ():   # edit references (requests send them too)
            cmd += ["--input_image", image]
        if spec["weights_path"]:
            cmd += ["--weights_path", spec["weights_path"]]
        if spec["npu_files_path"]:
            cmd += ["--npu_files_path", spec["npu_files_path"]]
        # The host resolves its default weights, NPU files and LoRA paths relative to its own folder.
        # stderr (progress bars, errors) is copied to this process's stderr, so it still shows
        # in the ComfyUI console, and its tail is kept for progress() and error messages.
        self._buf = b""
        self._next_id = 0
        self._tail = b""
        self._pending = b""
        self._log = b""
        self._peak = 0.0
        self._cmd, self._cwd = cmd, host_dir
        self.proc = None
        if self.one_shot:
            return
        self.proc = subprocess.Popen(cmd, cwd=host_dir, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._copy_output, args=(self.proc.stderr,), daemon=True).start()

    # Lines only worker mode prints: its memory mode, and H/W/Prompt once at start-up (with the
    # built-in demo prompt). Left out so the console matches a normal run; the node prints
    # H, W and the real prompt for each image instead.
    _WORKER_ONLY = (b"Memory mode:", b"H = ", b"W = ", b"Prompt = ")

    def _console(self, chunk):
        """The part of `chunk` to show in the console: whole lines minus _WORKER_ONLY ones.
        A partial line is shown at once (progress bars redraw with \\r) unless it may still
        become a left-out line."""
        self._pending += chunk
        shown = b""
        while b"\n" in self._pending:
            line, self._pending = self._pending.split(b"\n", 1)
            if not line.lstrip(b"\r").startswith(self._WORKER_ONLY):
                shown += line + b"\n"
        rest = self._pending.rsplit(b"\r", 1)[-1]
        if not rest or not any(p.startswith(rest) or rest.startswith(p) for p in self._WORKER_ONLY):
            shown += self._pending
            self._pending = b""
        return shown

    def _copy_output(self, stream):
        fd = stream.fileno()
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            if self.one_shot:   # a normal run: shown as is, kept whole for the timings
                self._log += chunk
            shown = chunk if self.one_shot else self._console(chunk)
            if shown:
                try:
                    os.write(2, shown)
                except OSError:
                    pass
            self._tail = (self._tail + chunk)[-2000:]

    _BAR = re.compile(rb"(Text Encoder|Denoising|VAE Decoder): *\d+%\|[^|]*\| *(\d+)/(\d+)")

    def progress(self):
        """(stage, done, total) of the newest progress bar of this request, or None before
        the first one (model still loading)."""
        bars = self._BAR.findall(self._tail)
        if not bars:
            return None
        stage, done, total = bars[-1]
        return stage.decode(), int(done), int(total)

    def _exit_error(self):
        code = self.proc.wait()
        errors = [l.strip() for l in self._tail.decode("utf-8", "replace").splitlines()
                  if l.startswith("Error:")]
        detail = errors[-1] if errors else "see the console output above"
        return WorkerError(f"{self.name} NPU worker exited (code {code}): {detail}")

    @property
    def requests_sent(self):
        return self._next_id

    def alive(self):
        return self.one_shot or self.proc.poll() is None

    def memory_gb(self):
        """(resident now, peak since this worker started) in GB, from /proc. NPU buffers are
        host RAM mapped into the worker, so they are included. (0, 0) if it has exited."""
        now, peak = 0.0, self._peak
        if self.proc is None:
            return now, peak
        try:
            with open(f"/proc/{self.proc.pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        now = int(line.split()[1]) / 1e6
                    elif line.startswith("VmHWM:"):
                        peak = int(line.split()[1]) / 1e6
        except (OSError, ValueError, IndexError):
            pass
        return now, peak

    def _read_line(self, should_cancel):
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            if should_cancel is not None and should_cancel():
                self.close(kill=True)
                raise Cancelled()
            if not select.select([fd], [], [], 0.2)[0]:
                continue
            chunk = os.read(fd, 65536)
            if not chunk:
                raise self._exit_error()
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line

    def generate(self, prompt, seed, output, steps=None, should_cancel=None, input_images=None):
        """Runs one image, writes the PNG to `output`, returns the worker's reply (timings).
        `should_cancel` is called about every 0.2 s while waiting; True kills the worker."""
        if self.one_shot:
            return self._generate_once(prompt, seed, output, steps, should_cancel, input_images)
        self._next_id += 1
        request = {"id": self._next_id, "prompt": prompt, "seed": int(seed),
                   "steps": int(steps or self.spec["steps"]), "output": os.path.abspath(output)}
        if input_images:
            request["input_image"] = [os.path.abspath(p) for p in input_images]
        self._tail = b""
        try:
            self.proc.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise self._exit_error()
        while True:
            line = self._read_line(should_cancel)
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            if isinstance(reply, dict) and reply.get("id") == request["id"]:
                break
        if not reply.get("ok"):
            raise WorkerError(f"{self.name} NPU worker: {reply.get('error', 'request failed')}")
        return reply

    _STAGE = re.compile(rb"(Vision Encoder|Text Encoder|VAE Encoder|Denoising|VAE Decoder): *100%\|[^|]*\| *"
                        rb"(\d+)/\d+ \[[^,\]]*, *([\d.]+)(s/it|it/s)")

    def _generate_once(self, prompt, seed, output, steps, should_cancel, input_images):
        """One run.exe for one image (no worker mode); timings from its progress bars and E2E line."""
        self._next_id += 1
        cmd = self._cmd + [f"--prompt={prompt}", "--seed", str(int(seed)), "--steps", str(int(steps or self.spec["steps"])),
                           "--output", os.path.abspath(output)]
        for image in input_images or ():
            cmd += ["--input_image", os.path.abspath(image)]
        self._tail = self._log = b""
        started = time.time()
        self.proc = subprocess.Popen(cmd, cwd=self._cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, bufsize=0)
        reader = threading.Thread(target=self._copy_output, args=(self.proc.stdout,), daemon=True)
        reader.start()
        while self.proc.poll() is None:
            self._peak = max(self._peak, self.memory_gb()[1])
            if should_cancel is not None and should_cancel():
                self.proc.kill()
                self.proc.wait()
                raise Cancelled()
            time.sleep(0.2)
        reader.join(timeout=5)
        if self.proc.returncode != 0 or not os.path.isfile(output):
            raise self._exit_error()
        seconds = {}
        for stage, count, value, unit in self._STAGE.findall(self._log.replace(b"\r", b"\n")):
            per_item = float(value) if unit == b"s/it" else 1.0 / max(float(value), 1e-9)
            seconds[stage] = seconds.get(stage, 0.0) + int(count) * per_item
        e2e = re.search(rb"E2E runtime: *([\d.]+) *s", self._log)
        reference = seconds.get(b"Vision Encoder", 0.0) + seconds.get(b"VAE Encoder", 0.0)
        return {"ok": True, "output": output, "conditioning_seconds": reference,
                "text_encoder_seconds": seconds.get(b"Text Encoder", 0.0),
                "denoising_seconds": seconds.get(b"Denoising", 0.0),
                "vae_decoder_seconds": seconds.get(b"VAE Decoder", 0.0),
                "e2e_seconds": float(e2e.group(1)) if e2e else time.time() - started,
                "request_seconds": time.time() - started, "reused": False}

    def close(self, kill=False):
        self._close_process(kill)
        if self._lora_config:
            try:
                os.remove(self._lora_config)
            except OSError:
                pass
            self._lora_config = None

    def _close_process(self, kill):
        if self.proc is not None and self.proc.poll() is None:
            try:
                if kill:
                    self.proc.kill()
                else:
                    self.proc.stdin.close()  # EOF: the worker releases everything and exits
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()
                self.proc.wait()


Flux2Worker = NPUWorker   # older name

_lock = threading.Lock()
_worker = None


def get_worker(spec, height, width, lora="none", lora_scale=None, comfy_models_dir=None, lora_stack=None,
               input_images=None):
    """Reuse the resident worker while its launch inputs match. Only one model is resident:
    the NPU runs one job at a time, so switching models stops the previous worker first.
    lora_stack: tuple of (absolute file, strength, extra loras.jsonc fields) from Load LoRA nodes;
    it replaces lora and lora_scale."""
    global _worker
    key = (spec["name"], spec["host_dir"], height, width, lora, lora_scale, comfy_models_dir, lora_stack)
    with _lock:
        if _worker is not None and (_worker.key != key or not _worker.alive()):
            _worker.close()
            _worker = None
        if _worker is None:
            _worker = NPUWorker(spec, height, width, lora, lora_scale, comfy_models_dir, lora_stack, input_images)
        return _worker


def release_worker():
    global _worker
    with _lock:
        if _worker is not None:
            _worker.close()
            _worker = None


atexit.register(release_worker)


if __name__ == "__main__":
    import argparse
    import time
    host = default_host_dir()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="A model name from models.json")
    ap.add_argument("--list", action="store_true", help="List the available models and exit")
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--output", help="PNG path; _<n> is added from the second prompt on")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=0, help="0 = the model's default")
    ap.add_argument("--lora", default="none")
    args = ap.parse_args()
    models = available_models(host)
    if args.list:
        for name in models:
            s = model_spec(host, name)
            print(f"{name}: {s['host_dir']} (steps {s['steps']})")
        sys.exit(0)
    if not args.prompt or not args.output:
        ap.error("--prompt and --output are required")
    spec = model_spec(host, args.model or models[0])
    steps = resolve_steps(spec, args.steps)
    for i, prompt in enumerate(args.prompt):
        out = args.output if i == 0 else args.output.replace(".png", f"_{i}.png")
        t = time.time()
        w = get_worker(spec, args.height, args.width, args.lora)
        print(json.dumps(w.generate(prompt, args.seed, out, steps)), f"wall {time.time() - t:.2f} s",
              "RAM now %.2f GB, peak %.2f GB" % w.memory_gb())
