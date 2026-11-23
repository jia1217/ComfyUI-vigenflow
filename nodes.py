"""ComfyUI nodes: VigenFlow text-to-image models on the Ryzen AI NPU, through the hosts' worker mode.

One node, "VigenFlow: Text to Image (NPU)", drives every model listed under "models" in
models.json (and the optional vigenflow_settings.json); see README.md.
"""
import hashlib
import os
import sys
import tempfile
import time
import uuid

import numpy as np
import torch
from PIL import Image

from . import installer
from . import worker as npu_worker

try:  # present inside ComfyUI only
    import comfy.model_management as model_management
    import comfy.utils
    import folder_paths
    from server import PromptServer
except ImportError:
    model_management = None
    folder_paths = None
    PromptServer = None


def _interrupted():
    return model_management is not None and model_management.processing_interrupted()


def _image_size_inputs():
    options = {"default": 1024, "min": 16, "max": 2048, "step": 16}
    return {name: ("INT", options.copy()) for name in ("width", "height")}


# aspect_ratio -> the sizes offered for it: exact ratio, multiples of 16, sides up to 2048.
# Portrait ratios are the landscape sizes turned around. The first size chosen for a ratio
# is the one closest to 1 megapixel. The web UI's 1:1 sizes are all here.
_LANDSCAPE_SIZES = {
    "1:1": [(512, 512), (768, 768), (1024, 1024), (1536, 1536), (2048, 2048)],
    "4:3": [(768, 576), (1024, 768), (1152, 864), (1280, 960), (1536, 1152), (2048, 1536)],
    "3:2": [(768, 512), (1152, 768), (1248, 832), (1536, 1024), (1920, 1280)],
    "16:9": [(1024, 576), (1280, 720), (1536, 864), (1792, 1008), (2048, 1152)],
}
RATIO_SIZES = {}
for _ratio, _sizes in _LANDSCAPE_SIZES.items():
    RATIO_SIZES[_ratio] = _sizes
    if _ratio != "1:1":
        _w, _h = _ratio.split(":")
        RATIO_SIZES[f"{_h}:{_w}"] = [(h, w) for w, h in _sizes]
RATIO_ORDER = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
SIZE_CUSTOM = "Custom"


def _size_label(width, height):
    return f"{width} × {height}"


ALL_SIZES = [_size_label(*s) for r in RATIO_ORDER for s in RATIO_SIZES[r]]


def _custom_size_inputs():
    """width and height typed in for Custom only: no sockets, sizes are connected through image_size."""
    options = {"default": 1024, "min": 16, "max": 2048, "step": 16, "socketless": True}
    return {name: ("INT", options.copy()) for name in ("width", "height")}


def _size_preset_inputs():
    """aspect_ratio, size (filtered to the ratio in the browser), then width and height,
    which are used when aspect_ratio is Custom."""
    return {
        "aspect_ratio": (RATIO_ORDER + [SIZE_CUSTOM], {
            "default": "1:1",
            "tooltip": "Width:height. Custom: type width and height (16-2048, multiples of 16)."}),
        "size": (ALL_SIZES, {"default": _size_label(1024, 1024),
                             "tooltip": "Image size (width × height) for the chosen aspect ratio"}),
        **_custom_size_inputs(),
    }


def _edit_toggle_inputs():
    """The VigenFlow Studio's two edit size options."""
    return {
        "keep_ratio": ("BOOLEAN", {"default": True,
                                   "tooltip": "Keep the input image's aspect ratio: fit it inside the chosen size"}),
        "match_input": ("BOOLEAN", {"default": False,
                                    "tooltip": "Output at the input image's own size (up to 2048, multiples of 16)"}),
    }


def _edit_output_size(image_w, image_h, box_w, box_h, keep_ratio, match_input):
    """The edit output size, as the VigenFlow Studio computes it: match_input uses the input's size
    (scaled down to fit 2048), keep_ratio fits the input's shape inside box_w x box_h, else the box.
    Sides are rounded to multiples of 16 within 16..2048."""
    if match_input:
        scale = min(1.0, 2048 / max(image_w, image_h))
    elif keep_ratio:
        scale = min(box_w / image_w, box_h / image_h)
    else:
        return box_w, box_h
    return tuple(max(16, min(2048, int(round(side * scale / 16)) * 16)) for side in (image_w, image_h))


SIZE_FROM_IMAGE = ("INT", {"default": 1, "min": 1, "max": 4,
                           "tooltip": "Which image keep_ratio and match_input use (1 = the image being edited)"})


def _size_box(image_size):
    """VIGENFLOW_SIZE -> (width, height, keep_ratio, match_input, size_from_image). VigenFlow: Image
    Size sends (width, height); VigenFlow: Edit Output Size also sends the edit options."""
    if isinstance(image_size, dict):
        return (*image_size["size"], image_size["keep_ratio"], image_size["match_input"],
                image_size.get("size_from_image", 1))
    width, height = image_size
    return width, height, False, False, 1


INPUT_LIMITS = ["Auto", "512", "1024", "2048"]   # the VigenFlow Studio's "Input limit"


def _aligned(width, height, factor):
    w, h = max(1, round(width * factor)), max(1, round(height * factor))
    return (w, h), (max(16, w // 16 * 16), max(16, h // 16 * 16))


def _reference_plan(sizes, limit, budget):
    """Each reference's (scaled, sent) size, as the VigenFlow Studio plans them: its own shape,
    scaled down only when the long edge is over the limit (Auto = 2048), sides rounded down to
    multiples of 16. With Auto and a model token budget (16x16 blocks), larger references are
    reduced first until the total fits."""
    long_edge = 2048 if limit == "Auto" else int(limit)
    full = [_aligned(w, h, min(1.0, long_edge / max(w, h))) for w, h in sizes]
    tokens = lambda plan: sum((tw // 16) * (th // 16) for _, (tw, th) in plan)
    if limit != "Auto" or not budget or tokens(full) <= budget:
        return full
    plan_at = lambda area: [_aligned(w, h, min(1.0, (area / (w * h)) ** 0.5)) for (w, h), _ in full]
    lower, upper = 0.0, float(max(w * h for (w, h), _ in full))
    for _ in range(24):
        middle = (lower + upper) / 2
        if tokens(plan_at(middle)) <= budget:
            lower = middle
        else:
            upper = middle
    return plan_at(lower)


def _flux3_prepared(width, height, multi):
    """The size the FLUX.3 Edit worker itself resizes a reference to before encoding (the official
    FLUX.3 rule with sides capped at 2048; same as the VigenFlow Studio's flux3Prepared): several
    references each fitted to 1024 x 1024 pixels and cropped to the 16-pixel grid, one reference
    fitted to 4 megapixels on a 16-pixel canvas."""
    max_side, grid = 2048, 16
    if multi:
        rw, rh = width, height
        if width * height > 1024 * 1024:
            scale = (1024 * 1024 / (width * height)) ** 0.5
            rw, rh = int(width * scale), int(height * scale)
        if max(rw, rh) > max_side:
            scale = max_side / max(rw, rh)
            rw, rh = int(rw * scale), int(rh * scale)
        return rw // grid * grid, rh // grid * grid
    ceil16 = lambda v: -(-v // grid) * grid
    limit = 4 * 1024 * 1024
    scale = min(1.0, (limit / (width * height)) ** 0.5, max_side / max(width, height))
    while True:
        fw, fh = max(1, round(width * scale)), max(1, round(height * scale))   # round: half to even, as JS
        cw, ch = min(max_side, max(64, ceil16(fw))), min(max_side, max(64, ceil16(fh)))
        if fw <= cw and fh <= ch and cw * ch <= limit:
            return cw, ch
        scale *= 0.995


def _prepare_references(images, folder, limit="Auto", budget=0, preparation=""):
    """ComfyUI IMAGE tensors -> PNG files for the edit worker, sized by _reference_plan (the part
    beyond the multiple of 16 is cropped from the center). Returns (paths, tokens, one line per
    reference: its size and, when changed, the size sent)."""
    pictures = []
    for tensor in images:
        array = (tensor[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        pictures.append(Image.fromarray(array[..., :3]).convert("RGB"))
    plan = _reference_plan([im.size for im in pictures], limit, budget)
    paths, tokens, lines = [], 0, []
    for i, (im, ((w, h), (tw, th))) in enumerate(zip(pictures, plan), 1):
        sent = (tw // 16) * (th // 16)
        size = f"{im.size[0]} × {im.size[1]}"
        used = ""
        if preparation == "flux3":   # the worker resizes it again: show that size and its tokens
            uw, uh = _flux3_prepared(tw, th, len(plan) > 1)
            sent = (uw // 16) * (uh // 16)
            used = f" → FLUX.3 uses {uw} × {uh}" if (uw, uh) != (tw, th) else ""
        lines.append(f"· image {i}: {size}" + (f" → resized {tw} × {th}" if (tw, th) != im.size else "") +
                     f"{used} ({sent} tokens)")
        if (w, h) != im.size:
            im = im.resize((w, h), Image.LANCZOS)
        left, top = (w - tw) // 2, (h - th) // 2
        im = im.crop((left, top, left + tw, top + th)) if (tw, th) != (w, h) else im
        path = os.path.join(folder, f"vigenflow_ref_{uuid.uuid4().hex}.png")
        im.save(path)
        paths.append(path)
        tokens += sent
    return paths, tokens, lines


def _chosen_size(aspect_ratio, size, width, height):
    """(width, height) from the ratio's size, or the width and height inputs for Custom."""
    if aspect_ratio == SIZE_CUSTOM:
        return width, height
    sizes = RATIO_SIZES.get(aspect_ratio)
    if sizes is None:
        raise ValueError(f"Unknown aspect ratio '{aspect_ratio}'")
    for w, h in sizes:
        if size == _size_label(w, h):
            return w, h
    raise ValueError(f"Size {size} is not a {aspect_ratio} size; pick one of "
                     f"{', '.join(_size_label(w, h) for w, h in sizes)}")


# Latent channels -> pixels per latent cell: Empty Flux 2 Latent (128 ch, /16),
# Empty SD3 Latent (16 ch, /8) and Empty Latent Image (4 ch, /8).
_LATENT_SCALE = {128: 16, 16: 8, 4: 8}


def _size_from_latent(latent):
    samples = latent.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim != 4 or samples.shape[1] not in _LATENT_SCALE:
        raise ValueError("Connect Empty Flux 2 Latent, Empty SD3 Latent or Empty Latent Image to latent_size")
    scale = _LATENT_SCALE[samples.shape[1]]
    height, width = (dimension * scale for dimension in samples.shape[-2:])
    return width, height


def _accept_license_input(models):
    """The "I agree" switch for models whose weights have their own license (Krea 2, Qwen-Image)."""
    host = npu_worker.default_host_dir()
    licensed = [npu_worker.model_spec(host, m) for m in models if npu_worker.model_spec(host, m).get("license")]
    terms = "; ".join(f"{s['name']}: {s['license']['name']}, {s['license']['url']}" for s in licensed)
    return ("BOOLEAN", {"default": False,
                        "tooltip": "On = you accept the model's license (like typing I agree), so its weights "
                                   f"can download. Only for: {terms or 'none'}"})


LORA_NONE = "none"


def _lora_files(spec):
    """{name: absolute path} of the .safetensors files in ComfyUI/models/loras/<weights_folder>/:
    each model has its own folder, so files of different models are never offered together."""
    folder = spec.get("weights_folder")
    if folder_paths is None or not folder:
        return {}
    files = {}
    for rel in folder_paths.get_filename_list("loras"):
        rel_posix = rel.replace(os.sep, "/")
        if rel_posix.startswith(folder + "/"):
            path = folder_paths.get_full_path("loras", rel)
            if path:
                files[rel_posix[len(folder) + 1:]] = os.path.abspath(path)
    return files


def _lora_models(host):
    """Models that take LoRAs (not the upscaler)."""
    return [m for m in npu_worker.available_models(host) if npu_worker.model_spec(host, m)["kind"] != "upscale" and npu_worker.model_spec(host, m)["lora"]]


def model_loras():
    """{model: ["none", the model's LoRA list..., other files in ComfyUI/models/loras/<id>/...]}.
    The list is the engine's LoRAs merged with the user's ComfyUI/models/loras/<id>/loras.jsonc
    (rebuilt here, so a page reload shows changes); files that an entry uses or that "hide"
    names are not listed again."""
    host = npu_worker.default_host_dir()
    try:
        if installer.engine_ready():
            installer.write_lora_configs()
    except Exception as exc:   # noqa: BLE001 - keep the node usable
        print(f"[VigenFlow] {exc}", file=sys.stderr)
    result = {}
    for model in _lora_models(host):
        spec = npu_worker.model_spec(host, model)
        entries = npu_worker.lora_entries(spec["host_dir"])
        used = {os.path.normpath(e["file"]) for e in entries}
        hide = installer.read_user_loras(spec)[0] if spec["engine"] else set()
        names = [LORA_NONE] + [e["name"] for e in entries]
        names += [name for name, path in _lora_files(spec).items()
                  if name not in names and name not in hide and os.path.normpath(path) not in used]
        result[model] = names
    return result


def _resolve_lora_stack(spec, model, stack):
    """Load LoRA items -> hashable ((absolute file, strength, extra jsonc fields), ...) plus
    the trigger-word notes. Every item must belong to the generator's model."""
    entries = {e["name"]: e for e in npu_worker.lora_entries(spec["host_dir"])}
    resolved, notes = [], []
    for item in stack:
        if item.get("model") not in (None, model):
            raise ValueError(f"The Load LoRA nodes are set to {item['model']}, but this node runs {model}. "
                             f"Set them to {model}; LoRAs of different models cannot be mixed.")
        if item.get("config_name"):
            entry = entries.get(item["config_name"])
            if entry is None:
                raise ValueError(f"LoRA '{item['config_name']}' is not in {model}'s loras.jsonc "
                                 f"({spec['host_dir']}); choose a LoRA made for {model}")
            extra = tuple(sorted((k, v) for k, v in entry.items()
                                 if k in ("trigger", "source", "license") and isinstance(v, str)))
            if entry.get("trigger"):
                notes.append(f"{entry['name']}: {entry['trigger']}")
            resolved.append((entry["file"], float(item["strength"]), extra))
        else:
            if not os.path.isfile(item["file"]):
                raise ValueError(f"LoRA file not found: {item['file']}")
            resolved.append((item["file"], float(item["strength"]), ()))
    return tuple(resolved), notes


def run_text_to_image(model, prompt, seed, steps, width, height, lora, lora_scale, keep_loaded,
                      lora_file="", unique_id=None, latent_size=None, lora_stack=None, input_images=None,
                      input_limit="Auto", accept_license=False):
    keep_loaded = False   # always release the model after the image (the widget is hidden)
    if not prompt.strip():
        raise ValueError("prompt is empty")
    if latent_size is not None:
        width, height = _size_from_latent(latent_size)
    if not (16 <= width <= 2048 and 16 <= height <= 2048):
        raise ValueError("width and height must be between 16 and 2048 pixels")
    if width % 16 or height % 16:
        raise ValueError("width and height must be multiples of 16")
    host = npu_worker.default_host_dir()
    spec = npu_worker.model_spec(host, model)
    if min(width, height) < spec["min_size"]:
        raise ValueError(f"{model} makes images from {spec['min_size']} to 2048 pixels per side")
    if not spec["lora"] and (lora_stack or lora_file.strip() or lora not in ("none", "config default")):
        raise ValueError(f"{model} takes no LoRA yet; set lora to none and disconnect Load LoRA nodes")
    images = list(input_images or [])
    if spec["kind"] == "edit" and not 1 <= len(images) <= spec["max_images"]:
        raise ValueError(f"{model} takes 1 to {spec['max_images']} images")
    if spec["kind"] != "edit" and images:
        raise ValueError(f"{model} is a text-to-image model; use it on VigenFlow: Text to Image (NPU)")
    steps = npu_worker.resolve_steps(spec, steps)
    stack, trigger_notes = _resolve_lora_stack(spec, model, lora_stack) if lora_stack else ((), [])
    lora_file = lora_file.strip()
    if stack:
        lora, lora_file = "none", ""   # the connected Load LoRA chain replaces the lora widgets
    elif lora_file:
        lora_file = os.path.abspath(os.path.expanduser(lora_file))
        if not os.path.isfile(lora_file):
            raise ValueError(f"LoRA file not found: {lora_file}")
        lora = lora_file
    elif lora not in ("none", "config default") and lora not in npu_worker.lora_names(spec["host_dir"]):
        raise ValueError(f"LoRA '{lora}' is not in {model}'s loras.jsonc ({spec['host_dir']})")

    tmp_dir = folder_paths.get_temp_directory() if folder_paths else tempfile.gettempdir()
    os.makedirs(tmp_dir, exist_ok=True)
    png = os.path.join(tmp_dir, f"vigenflow_npu_{uuid.uuid4().hex}.png")
    refs, ref_tokens, ref_lines = [], 0, []
    try:
        refs, ref_tokens, ref_lines = _prepare_references(images, tmp_dir, input_limit, spec["reference_token_limit"],
                                                          spec["reference_preparation"])
        if lora == "config default":   # no --lora: the host merges the enabled entries with their own scales
            lora, scale = None, None
        else:
            scale = None if lora == "none" else float(lora_scale)
        comfy_models_dir = folder_paths.models_dir if folder_paths is not None else None
        if stack:
            lora, scale = None, None
        def install_status(text):   # engine or weights download, shown on the node
            if PromptServer is not None and unique_id is not None:
                PromptServer.instance.send_progress_text(f"{model}: {text}", unique_id)

        try:
            installer.ensure_model(spec, install_status, _interrupted, accept_license)
        except npu_worker.Cancelled:
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()
            raise
        w = npu_worker.get_worker(spec, height, width, lora, scale, comfy_models_dir, stack or None, refs)

        # Live status on the node itself: stage, elapsed time and worker RAM as text, plus
        # the node's progress bar (text encoder 1 + denoising steps + VAE 1 units).
        total = steps + 2
        show = PromptServer is not None and unique_id is not None
        bar = comfy.utils.ProgressBar(total, node_id=unique_id) if show else None
        last = [0.0]

        def status(text):
            if show:
                now, peak = w.memory_gb()
                PromptServer.instance.send_progress_text(
                    f"{model}: {text}\nNPU worker RAM: {now:.1f} GB (peak {peak:.1f} GB)", unique_id)

        first = w.requests_sent == 0   # the model loads during the first request

        def tick():
            if show and time.time() - last[0] >= 0.5:
                last[0] = time.time()
                stage, done, count = w.progress() or ("", 0, 0)   # newest finished step
                if stage == "":
                    status("Loading the model on the NPU..." if first else "Text encoder...")
                elif stage == "Text Encoder":
                    bar.update_absolute(1, total)
                    status(f"Denoising 0/{steps}")
                elif stage == "Denoising":
                    bar.update_absolute(1 + min(done, steps), total)
                    status(f"Denoising {done}/{count}" if done < count else "VAE decoder...")
                else:
                    bar.update_absolute(total, total)
                    status("VAE decoder done")
            return _interrupted()

        try:
            if not w.one_shot:   # a worker prints them once at start-up; a normal run prints them itself
                print(f"H = {height}\nW = {width}\nPrompt = {prompt}", file=sys.stderr, flush=True)
            reply = w.generate(prompt, seed, png, steps, should_cancel=tick, input_images=refs)
            ram_now, ram_peak = w.memory_gb()   # before keep_loaded=False releases the worker
        except npu_worker.Cancelled:
            npu_worker.release_worker()
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()
            raise
        with Image.open(png) as im:
            pixels = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    finally:
        for path in [png] + refs:
            if os.path.exists(path):
                os.remove(path)
        if not keep_loaded:
            npu_worker.release_worker()

    text_s, denoise_s, vae_s = (reply.get(k, 0.0) for k in
                                ("text_encoder_seconds", "denoising_seconds", "vae_decoder_seconds"))
    e2e_s = reply.get("e2e_seconds", reply.get("request_seconds", 0.0))
    timings = (f"E2E {e2e_s:.2f} s (text {text_s:.2f}, denoise {denoise_s:.2f}, VAE {vae_s:.2f}); "
               f"request {reply.get('request_seconds', 0.0):.2f} s; "
               f"{'model was resident' if reply.get('reused') else 'model loaded for this image'}")
    # RAM of run.exe (model weights + NPU buffers); ComfyUI's own process is not included.
    memory = (f"NPU worker RAM: {ram_now:.2f} GB now, {ram_peak:.2f} GB peak since load"
              f"{'' if keep_loaded else '; released after this image'}")
    loras = (" LoRAs " + ", ".join(f"{os.path.basename(f)} x{s:g}" for f, s, _ in stack) + ";") if stack else ""
    print(f"[VigenFlow {model}] {width}x{height} seed {seed} steps {steps};{loras} {timings}; {memory}", flush=True)
    for line in ref_lines:
        print(f"[VigenFlow {model}] reference {line[2:]}", flush=True)
    for note in trigger_notes:
        print(f"[VigenFlow {model}] LoRA trigger words, {note}", flush=True)
    reference_s = reply.get("conditioning_seconds", 0.0)
    runtime_summary = (
        (f"References {len(refs)} image{'s' * (len(refs) > 1)}, {ref_tokens} tokens\n" +
         "".join(line + "\n" for line in ref_lines) +
         f"Reference encoder runtime {reference_s:.2f} s\n"
         f"Output size {width} × {height}\n" if refs else "") +
        f"Text encoder runtime {text_s:.2f} s\n"
        f"Denoising runtime {denoise_s:.2f} s\n"
        f"VAE decoder runtime {vae_s:.2f} s\n"
        f"E2E runtime {e2e_s:.2f} s"
    )
    return {
        "ui": {"text": [runtime_summary], **({"vigenflow_refs": [f"Sent to {model}:"] + ref_lines} if refs else {})},
        "result": (torch.from_numpy(pixels).unsqueeze(0),),
    }


def _lora_inputs(names):
    return {
        "lora": (["none", "config default"] + names, {
            "default": "none",
            "tooltip": "A name from the model's loras.jsonc. 'config default' merges the entries enabled there."}),
        "lora_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
        "keep_loaded": ("BOOLEAN", {"default": False,
                                    "tooltip": "Keep the model resident between images. Changing model, size or LoRA reloads it."}),
    }


def _optional_inputs():
    return {
        "prompt_text": ("STRING", {"forceInput": True,
                                   "tooltip": "Connect a VigenFlow: Prompt node; replaces the prompt box."}),
        "lora_file": ("STRING", {"default": "",
                                 "tooltip": "Path to a .safetensors LoRA; replaces the 'lora' choice when set."}),
        "latent_size": ("LATENT", {"tooltip": "Use the width and height from an empty latent node "
                                              "(Empty Flux 2 Latent, Empty SD3 Latent or Empty Latent Image)."}),
        "lora_stack": ("VIGENFLOW_LORA", {"tooltip": "LoRAs from VigenFlow: Load LoRA nodes; "
                                                     "replaces lora, lora_scale and lora_file."}),
    }


class VigenFlowTextToImage:
    """Prompt -> image with any VigenFlow text-to-image model in models.json.
    Text encoder, denoising and VAE decode all run in that model's run.exe."""

    @classmethod
    def INPUT_TYPES(cls):
        host = npu_worker.default_host_dir()
        models = npu_worker.available_models(host, "text_to_image")
        step_help = "; ".join(
            f"{m} {npu_worker.model_spec(host, m)['steps']}" for m in models
            if m in npu_worker.load_settings(host)["models"])
        lora = _lora_inputs(npu_worker.all_lora_names(host))
        optional = _optional_inputs()
        # Widget order = layout on the canvas (web/denoising_runtime.js converts older saved graphs
        # by widget name): model, prompt, size, sampling, LoRA. The model is always released after
        # an image, so there is no keep_loaded widget (older graphs' value is ignored).
        return {
            "required": {
                "model": (models, {"default": models[0],
                                   "tooltip": "VigenFlow models (models.json); the engine and weights download on first use"}),
                "prompt": ("STRING", {"multiline": True, "default": "A mountain lake at sunrise",
                                      "tooltip": "Text prompt; a connected prompt_text replaces it"}),
                **_size_preset_inputs(),
                "steps": ("INT", {"default": 0, "min": 0, "max": 50,
                                  "tooltip": f"0 = the model's default ({step_help})"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                "lora": lora["lora"],
                "lora_scale": lora["lora_scale"],
                "lora_file": ("STRING", {"default": "",
                                         "tooltip": "Path to a .safetensors LoRA; replaces the 'lora' choice when set."}),
                "accept_license": _accept_license_input(models),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "prompt_text": optional["prompt_text"],
                "image_size": ("VIGENFLOW_SIZE", {"tooltip": "Connect VigenFlow: Image Size; replaces aspect_ratio, "
                                                             "size, width and height."}),
                "lora_stack": optional["lora_stack"],
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "NPU"

    def generate(self, model, prompt, aspect_ratio, size, width, height, steps, seed, lora, lora_scale,
                 lora_file, keep_loaded=False, unique_id=None, lora_stack=None, prompt_text=None, image_size=None,
                 accept_license=False):
        if image_size is not None:
            width, height = _size_box(image_size)[:2]
        else:
            width, height = _chosen_size(aspect_ratio, size, width, height)
        if prompt_text is not None:
            prompt = prompt_text
        return run_text_to_image(model, prompt, seed, steps, width, height, lora, lora_scale,
                                 keep_loaded, lora_file, unique_id, None, lora_stack, accept_license=accept_license)


class VigenFlowImageEdit:
    """Prompt + reference images -> image with a VigenFlow edit model (FLUX.2 Klein 4B Edit: 1-4
    references; Qwen-Image 2.1 Turbo Edit and FLUX.3 image klein Edit: 1-10). Text encoder, reference encoder, denoising and VAE
    decode run in its run.exe."""

    @classmethod
    def INPUT_TYPES(cls):
        host = npu_worker.default_host_dir()
        models = npu_worker.available_models(host, "edit")
        specs = [npu_worker.model_spec(host, m) for m in models]
        step_help = "; ".join(f"{s['name']} {s['steps']}" for s in specs)
        image_help = "; ".join(f"{s['name']} 1-{s['max_images']}" for s in specs)
        return {
            "required": {
                "model": (models, {"default": models[0] if models else None,
                                   "tooltip": "VigenFlow edit models (models.json); the engine and weights "
                                              "download on first use"}),
                "prompt": ("STRING", {"multiline": True, "default": "Make it a snowy winter scene",
                                      "tooltip": "What to change; a connected prompt_text replaces it"}),
                **_size_preset_inputs(),
                **_edit_toggle_inputs(),
                "steps": ("INT", {"default": 0, "min": 0, "max": 50,
                                  "tooltip": f"0 = the model's default ({step_help})"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2**31 - 1}),
                **{k: v for k, v in _lora_inputs(npu_worker.all_lora_names(host)).items() if k != "keep_loaded"},
                "lora_file": ("STRING", {"default": "",
                                         "tooltip": "Path to a .safetensors LoRA; replaces the 'lora' choice when set."}),
                "input_limit": (INPUT_LIMITS, {
                    "default": "Auto",
                    "tooltip": "References keep their own size; this only scales down those whose long edge is "
                               "larger. Auto = 2048 and within the model's reference token budget (larger "
                               "references first). Smaller is faster: each 16x16 block is one token."}),
                "size_from_image": SIZE_FROM_IMAGE,
                "accept_license": _accept_license_input(models),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
            "optional": {
                "edit_input": ("VIGENFLOW_EDIT", {"tooltip": "From VigenFlow: Edit Prompt: the prompt and its images "
                                                             "in one wire (replaces the prompt box)"}),
                "references": ("VIGENFLOW_IMAGES", {"tooltip": f"From VigenFlow: Reference Images ({image_help})"}),
                "image": ("IMAGE", {"tooltip": "One image to edit (instead of, or before, the references)"}),
                "prompt_text": _optional_inputs()["prompt_text"],
                "image_size": ("VIGENFLOW_SIZE", {"tooltip": "Connect VigenFlow: Edit Output Size (or Image Size); "
                                                             "replaces the size controls."}),
                "lora_stack": _optional_inputs()["lora_stack"],
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "edit"
    CATEGORY = "NPU"

    def edit(self, model, prompt, aspect_ratio, size, width, height, keep_ratio, match_input, steps, seed,
             lora, lora_scale, lora_file, input_limit="Auto", size_from_image=1, accept_license=False,
             unique_id=None, lora_stack=None,
             prompt_text=None, image_size=None, references=None, image=None, edit_input=None):
        images = (list(edit_input["images"]) if edit_input is not None else []) + \
                 ([image[i:i + 1] for i in range(image.shape[0])] if image is not None else []) + list(references or [])
        if not images:
            raise ValueError("Connect VigenFlow: Edit Prompt to edit_input, Reference Images to references, "
                             "or an image to image")
        if image_size is not None:
            box_w, box_h, keep_ratio, match_input, size_from_image = _size_box(image_size)
        else:
            box_w, box_h = _chosen_size(aspect_ratio, size, width, height)
        if (keep_ratio or match_input) and size_from_image > len(images):
            raise ValueError(f"size_from_image is {size_from_image}, but there are only {len(images)} image(s)")
        image = images[size_from_image - 1]
        width, height = _edit_output_size(int(image.shape[2]), int(image.shape[1]), box_w, box_h,
                                          keep_ratio, match_input)
        if prompt_text is not None:
            prompt = prompt_text
        if edit_input is not None:
            prompt = edit_input["prompt"]
        return run_text_to_image(model, prompt, seed, steps, width, height, lora, lora_scale, False,
                                 lora_file, unique_id, None, lora_stack, images, input_limit, accept_license)


class VigenFlowUpscale:
    """Image -> square upscaled image with a VigenFlow upscaler (SeedVR2 3B), on the NPU. The input is
    center-cropped to a square and resized to upscaling_size first, as the VigenFlow server does."""

    @classmethod
    def INPUT_TYPES(cls):
        host = npu_worker.default_host_dir()
        models = npu_worker.available_models(host, "upscale")
        return {
            "required": {
                "model": (models, {"default": models[0] if models else None,
                                   "tooltip": "VigenFlow upscalers (models.json); the engine and weights download on first use"}),
                "image": ("IMAGE", {"tooltip": "The image to upscale (center-cropped to a square)"}),
                "upscaling_size": ("INT", {"default": 2048, "min": 16, "max": 2048, "step": 16,
                                           "tooltip": "Side of the square output, a multiple of 16"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "NPU"

    def upscale(self, model, image, upscaling_size, seed, unique_id=None):
        if upscaling_size % 16 or not 16 <= upscaling_size <= 2048:
            raise ValueError("upscaling_size must be 16-2048 and a multiple of 16")
        spec = npu_worker.model_spec(npu_worker.default_host_dir(), model)

        def status(text):
            if PromptServer is not None and unique_id is not None:
                PromptServer.instance.send_progress_text(f"{model}: {text}", unique_id)

        try:
            installer.ensure_model(spec, status, _interrupted)
        except npu_worker.Cancelled:
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()
            raise
        tmp_dir = folder_paths.get_temp_directory() if folder_paths else tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)
        array = (image[0].clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
        picture = Image.fromarray(array[..., :3]).convert("RGB")
        src_w, src_h = picture.size
        side = min(src_w, src_h)
        left, top = (src_w - side) // 2, (src_h - side) // 2
        source = os.path.join(tmp_dir, f"vigenflow_upscale_in_{uuid.uuid4().hex}.png")
        png = os.path.join(tmp_dir, f"vigenflow_upscale_{uuid.uuid4().hex}.png")
        picture.crop((left, top, left + side, top + side)).resize((upscaling_size, upscaling_size), Image.LANCZOS).save(source)
        status("Upscaling on the NPU...")
        try:
            w = npu_worker.get_worker(spec, upscaling_size, upscaling_size)
            try:
                print(f"[VigenFlow {model}] input {src_w} × {src_h}" + (" (center-cropped to a square)" if src_w != src_h else ""),
                      file=sys.stderr, flush=True)
                reply = w.generate("", seed, png, 1, should_cancel=_interrupted, input_images=[source])
            except npu_worker.Cancelled:
                npu_worker.release_worker()
                if model_management is not None:
                    model_management.throw_exception_if_processing_interrupted()
                raise
            ram_peak = w.memory_gb()[1]
            with Image.open(png) as im:
                pixels = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        finally:
            for path in (source, png):
                if os.path.exists(path):
                    os.remove(path)
            npu_worker.release_worker()
        summary = (f"Input {src_w} × {src_h}" + (" → center-cropped to a square" if src_w != src_h else "") + "\n"
                   f"Output size {upscaling_size} × {upscaling_size}\n"
                   f"VAE encoder runtime {reply.get('conditioning_seconds', 0.0):.2f} s\n"
                   f"Denoising runtime {reply.get('denoising_seconds', 0.0):.2f} s\n"
                   f"VAE decoder runtime {reply.get('vae_decoder_seconds', 0.0):.2f} s\n"
                   f"E2E runtime {reply.get('e2e_seconds', 0.0):.2f} s")
        print(f"[VigenFlow {model}] {upscaling_size}x{upscaling_size} seed {seed}; E2E {reply.get('e2e_seconds', 0.0):.2f} s; "
              f"request {reply.get('request_seconds', 0.0):.2f} s; NPU worker RAM peak {ram_peak:.2f} GB", flush=True)
        return {"ui": {"text": [summary]}, "result": (torch.from_numpy(pixels).unsqueeze(0),)}


class VigenFlowPrompt:
    """A prompt in its own node. Connect it to the generator's prompt_text input; one Prompt node
    can feed several generators."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "default": "A mountain lake at sunrise",
                                                 "tooltip": "Text prompt"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "NPU"

    def get_prompt(self, text):
        return (text,)


class VigenFlowLoadLora:
    """Add one LoRA to a VigenFlow LoRA stack. Chain several to merge several LoRAs, each with its
    own strength, and connect the last one to the generator's lora_stack input. All LoRAs in a
    stack belong to one model; lora_name lists only the LoRAs of the chosen model."""

    @classmethod
    def INPUT_TYPES(cls):
        host = npu_worker.default_host_dir()
        models = _lora_models(host)
        names = []
        for model_names in model_loras().values():   # the browser shows the chosen model's part
            names += [name for name in model_names if name not in names]
        return {
            "required": {
                "model": (models, {"default": models[0],
                                   "tooltip": "The model these LoRAs are for; must match the generator's model"}),
                "lora_name": (names or [LORA_NONE], {
                    "default": LORA_NONE,
                    "tooltip": "This model's LoRAs: entries of its loras.jsonc and the files in "
                               "ComfyUI/models/loras/<model folder>/. none adds nothing."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {
                "lora_stack": ("VIGENFLOW_LORA", {"tooltip": "LoRAs from earlier Load LoRA nodes (same model)"}),
            },
        }

    RETURN_TYPES = ("VIGENFLOW_LORA",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "load"
    CATEGORY = "NPU"

    def load(self, model, lora_name, strength, lora_stack=None):
        stack = list(lora_stack or [])
        for item in stack:
            if item.get("model") not in (None, model):
                raise ValueError(f"This Load LoRA is set to {model}, but an earlier one in the chain is set to "
                                 f"{item['model']}. Use LoRAs of one model in a stack.")
        if lora_name == LORA_NONE:
            return (stack,)
        host = npu_worker.default_host_dir()
        spec = npu_worker.model_spec(host, model)
        if lora_name in npu_worker.lora_names(spec["host_dir"]):
            stack.append({"model": model, "config_name": lora_name, "strength": strength})
            return (stack,)
        path = _lora_files(spec).get(lora_name)
        if not path:
            folder = spec.get("weights_folder") or "<model folder>"
            raise ValueError(f"'{lora_name}' is not a {model} LoRA: not in its loras.jsonc and not in "
                             f"ComfyUI/models/loras/{folder}/")
        stack.append({"model": model, "file": path, "strength": strength})
        return (stack,)


class VigenFlowImageSize:
    """Pick an image size. Connect image_size to the generator's image_size socket (one wire)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _size_preset_inputs()}

    RETURN_TYPES = ("VIGENFLOW_SIZE",)
    RETURN_NAMES = ("image_size",)
    FUNCTION = "get_size"
    CATEGORY = "NPU"

    def get_size(self, aspect_ratio, size, width, height):
        width, height = _chosen_size(aspect_ratio, size, width, height)
        if not (16 <= width <= 2048 and 16 <= height <= 2048) or width % 16 or height % 16:
            raise ValueError("width and height must be 16-2048 pixels and multiples of 16")
        return ((width, height),)


class VigenFlowEditPrompt:
    """The edit instruction and its images in one node: connect a Load Image node (each reference
    its own node, with its own upload and preview) to image_1, image_2, ... One wire to the edit node."""
    SLOTS = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"prompt": ("STRING", {"multiline": True, "default": "Make it a snowy winter scene",
                                               "tooltip": "What to change"})},
            "optional": {
                **{f"image_{i}": ("IMAGE", {"tooltip": "The image to edit" if i == 1 else f"Reference {i}"})
                   for i in range(1, cls.SLOTS + 1)},
                "references": ("VIGENFLOW_IMAGES", {"tooltip": "More references after image_1-image_4, for example "
                                                               "from VigenFlow: Reference Images (Qwen-Image Edit takes up to 10)"}),
            },
        }

    RETURN_TYPES = ("VIGENFLOW_EDIT",) + ("IMAGE",) * SLOTS   # image_1 is where the single image output was
    RETURN_NAMES = ("edit_input",) + tuple(f"image_{i}" for i in range(1, SLOTS + 1))
    OUTPUT_TOOLTIPS = ("The prompt and images, to VigenFlow: Image Edit",) + tuple(
        f"Image {i} as connected, for example for Compare Images" for i in range(1, SLOTS + 1))
    FUNCTION = "build"
    CATEGORY = "NPU"

    def build(self, prompt, references=None, **images):
        loaded = []
        for i in range(1, self.SLOTS + 1):
            batch = images.get(f"image_{i}")
            if batch is not None:
                loaded += [batch[j:j + 1] for j in range(batch.shape[0])]
        loaded += list(references or [])
        if not loaded:
            raise ValueError("Connect at least one image (for example a Load Image node) to VigenFlow: Edit Prompt")
        if not prompt.strip():
            raise ValueError("The edit prompt is empty")
        return ({"prompt": prompt, "images": loaded},) + tuple(images.get(f"image_{i}")
                                                              for i in range(1, self.SLOTS + 1))


class VigenFlowReferenceImages:
    """The images for VigenFlow: Image Edit in one node: pick up to four files from ComfyUI's input
    folder (click or drop a file on a slot's cell, or upload several at once in order),
    and/or connect images from other nodes. One wire to the edit node."""
    SLOTS = 4

    @classmethod
    def INPUT_TYPES(cls):
        folder = folder_paths.get_input_directory() if folder_paths is not None else ""
        files = sorted(f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))) if folder else []
        if folder_paths is not None:
            files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {"image_1": (files, {"tooltip": "The image to edit"})},
            "optional": {
                **{f"image_{i}": (["none"] + files, {"default": "none",
                                                      "tooltip": f"Reference {i} (none = not used)"})
                   for i in range(2, cls.SLOTS + 1)},
                "images": ("IMAGE", {"tooltip": "More references from other nodes; each image of the batch is one"}),
            },
        }

    RETURN_TYPES = ("VIGENFLOW_IMAGES",) + ("IMAGE",) * SLOTS   # image_1 is where the single image output was
    RETURN_NAMES = ("references",) + tuple(f"image_{i}" for i in range(1, SLOTS + 1))
    OUTPUT_TOOLTIPS = ("All references, to VigenFlow: Image Edit",) + tuple(
        f"Reference {i}, for example for Compare Images" for i in range(1, SLOTS + 1))
    FUNCTION = "load"
    CATEGORY = "NPU"

    @staticmethod
    def _names(kwargs):
        return [kwargs[f"image_{i}"] for i in range(1, VigenFlowReferenceImages.SLOTS + 1)
                if kwargs.get(f"image_{i}") not in (None, "", "none")]

    def load(self, images=None, **kwargs):
        from nodes import LoadImage

        loaded = [LoadImage().load_image(name)[0][:1] for name in self._names(kwargs)]
        if images is not None:
            loaded += [images[i:i + 1] for i in range(images.shape[0])]
        return (loaded,) + tuple(loaded[i] if i < len(loaded) else None for i in range(self.SLOTS))

    @classmethod
    def IS_CHANGED(cls, images=None, **kwargs):
        digest = hashlib.sha256()
        for name in cls._names(kwargs):
            with open(folder_paths.get_annotated_filepath(name), "rb") as file:
                digest.update(file.read())
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        for name in cls._names(kwargs):
            if not folder_paths.exists_annotated_filepath(name):
                return f"Invalid image file: {name}"
        return True


class VigenFlowEditSize:
    """The output size for VigenFlow: Image Edit, with the VigenFlow Studio's two options: keep the
    input's aspect ratio inside the chosen size, or match the input's own size."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {**_size_preset_inputs(), **_edit_toggle_inputs(), "size_from_image": SIZE_FROM_IMAGE}}

    RETURN_TYPES = ("VIGENFLOW_SIZE",)
    RETURN_NAMES = ("image_size",)
    FUNCTION = "get_size"
    CATEGORY = "NPU"

    def get_size(self, aspect_ratio, size, width, height, keep_ratio, match_input, size_from_image=1):
        width, height = _chosen_size(aspect_ratio, size, width, height)
        if not (16 <= width <= 2048 and 16 <= height <= 2048) or width % 16 or height % 16:
            raise ValueError("width and height must be 16-2048 pixels and multiples of 16")
        return ({"size": (width, height), "keep_ratio": keep_ratio, "match_input": match_input,
                 "size_from_image": size_from_image},)


class VigenFlowCompare:
    """Before/after view of two images with a slider (move the mouse over it); works in the
    classic node display too. Saves nothing: both images go to ComfyUI's temp folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"before": ("IMAGE", {"tooltip": "For example the image you edited"}),
                             "after": ("IMAGE", {"tooltip": "For example VigenFlow: Image Edit's result"})}}

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "NPU"

    def compare(self, before, after):
        from nodes import PreviewImage

        for name, image in (("before", before), ("after", after)):
            if image is None:
                raise ValueError(f"Compare Images: '{name}' is connected to an empty image slot "
                                 "(for example image_2 of Edit Prompt with no image connected)")

        preview = PreviewImage()
        files = {"before": preview.save_images(before[:1], "vigenflow.compare.before")["ui"]["images"][0],
                 "after": preview.save_images(after[:1], "vigenflow.compare.after")["ui"]["images"][0]}
        return {"ui": {"vigenflow_compare": [files]}}   # nested, so ComfyUI does not add its own image preview


class VigenFlowSaveImage:
    """Save images using output_dir from vigenflow_settings.json."""

    @classmethod
    def INPUT_TYPES(cls):
        from nodes import SaveImage

        return SaveImage.INPUT_TYPES()

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "NPU"

    def save_images(self, images, filename_prefix="vigenflow_npu", prompt=None, extra_pnginfo=None):
        from nodes import PreviewImage, SaveImage

        output_dir = npu_worker.load_settings(npu_worker.default_host_dir()).get("output_dir")
        saver = SaveImage()
        if output_dir and os.path.realpath(output_dir) != os.path.realpath(saver.output_dir):
            saver.output_dir = output_dir
            saver.save_images(images, filename_prefix, prompt, extra_pnginfo)
            preview = PreviewImage().save_images(images)
            return {"ui": preview["ui"], "result": (images,)}

        saved = saver.save_images(images, filename_prefix, prompt, extra_pnginfo)
        return {"ui": saved["ui"], "result": (images,)}


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    from aiohttp import web

    @PromptServer.instance.routes.get("/vigenflow/licenses")
    async def _vigenflow_licenses(request):
        """{model: license name or null}, for greying out accept_license in the browser."""
        host = npu_worker.default_host_dir()
        return web.json_response({m: (npu_worker.model_spec(host, m).get("license") or {}).get("name")
                                  for m in npu_worker.available_models(host)})

    @PromptServer.instance.routes.get("/vigenflow/loras")
    async def _vigenflow_loras(request):
        """The LoRA names per model, for filtering Load LoRA's lora_name in the browser."""
        return web.json_response(model_loras())


try:   # LoRA lists of an installed engine point at this ComfyUI's weights folder
    if installer.engine_ready():
        installer.write_lora_configs()
except Exception as exc:   # noqa: BLE001 - never stop ComfyUI from starting
    print(f"[VigenFlow] {exc}", file=sys.stderr)


NODE_CLASS_MAPPINGS = {
    "VigenFlowTextToImage": VigenFlowTextToImage,
    "VigenFlowImageEdit": VigenFlowImageEdit,
    "VigenFlowUpscale": VigenFlowUpscale,
    "VigenFlowPrompt": VigenFlowPrompt,
    "VigenFlowLoadLora": VigenFlowLoadLora,
    "VigenFlowImageSize": VigenFlowImageSize,
    "VigenFlowEditSize": VigenFlowEditSize,
    "VigenFlowReferenceImages": VigenFlowReferenceImages,
    "VigenFlowEditPrompt": VigenFlowEditPrompt,
    "VigenFlowCompare": VigenFlowCompare,
    "VigenFlowSaveImage": VigenFlowSaveImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VigenFlowTextToImage": "VigenFlow: Text to Image (NPU)",
    "VigenFlowImageEdit": "VigenFlow: Image Edit (NPU)",
    "VigenFlowUpscale": "VigenFlow: Upscale (NPU)",
    "VigenFlowPrompt": "VigenFlow: Prompt",
    "VigenFlowLoadLora": "VigenFlow: Load LoRA",
    "VigenFlowImageSize": "VigenFlow: Image Size",
    "VigenFlowEditSize": "VigenFlow: Edit Output Size",
    "VigenFlowReferenceImages": "VigenFlow: Reference Images",
    "VigenFlowEditPrompt": "VigenFlow: Edit Prompt",
    "VigenFlowCompare": "VigenFlow: Compare Images",
    "VigenFlowSaveImage": "VigenFlow: Save Image",
}
