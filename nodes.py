"""ComfyUI nodes: VigenFlow text-to-image models on the Ryzen AI NPU, through the hosts' worker mode.

One node, "VigenFlow: Text to Image (NPU)", drives every model listed under "models" in
models.json (and the optional vigenflow_settings.json); see README.md.
"""
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
    for model in npu_worker.available_models(host):
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
                      lora_file="", unique_id=None, latent_size=None, lora_stack=None):
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
    try:
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
            installer.ensure_model(spec, install_status, _interrupted)
        except npu_worker.Cancelled:
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()
            raise
        w = npu_worker.get_worker(spec, height, width, lora, scale, comfy_models_dir, stack or None)

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
            print(f"H = {height}\nW = {width}\nPrompt = {prompt}", file=sys.stderr, flush=True)   # as a normal run
            reply = w.generate(prompt, seed, png, steps, should_cancel=tick)
            ram_now, ram_peak = w.memory_gb()   # before keep_loaded=False releases the worker
        except npu_worker.Cancelled:
            npu_worker.release_worker()
            if model_management is not None:
                model_management.throw_exception_if_processing_interrupted()
            raise
        with Image.open(png) as im:
            pixels = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    finally:
        if os.path.exists(png):
            os.remove(png)
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
    for note in trigger_notes:
        print(f"[VigenFlow {model}] LoRA trigger words, {note}", flush=True)
    runtime_summary = (
        f"Text encoder runtime {text_s:.2f} s\n"
        f"Denoising runtime {denoise_s:.2f} s\n"
        f"VAE decoder runtime {vae_s:.2f} s\n"
        f"E2E runtime {e2e_s:.2f} s"
    )
    return {
        "ui": {"text": [runtime_summary]},
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
        models = npu_worker.available_models(host)
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
                 lora_file, keep_loaded=False, unique_id=None, lora_stack=None, prompt_text=None, image_size=None):
        if image_size is not None:
            width, height = image_size
        else:
            width, height = _chosen_size(aspect_ratio, size, width, height)
        if prompt_text is not None:
            prompt = prompt_text
        return run_text_to_image(model, prompt, seed, steps, width, height, lora, lora_scale,
                                 keep_loaded, lora_file, unique_id, None, lora_stack)


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
        models = npu_worker.available_models(host)
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
    "VigenFlowPrompt": VigenFlowPrompt,
    "VigenFlowLoadLora": VigenFlowLoadLora,
    "VigenFlowImageSize": VigenFlowImageSize,
    "VigenFlowSaveImage": VigenFlowSaveImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "VigenFlowTextToImage": "VigenFlow: Text to Image (NPU)",
    "VigenFlowPrompt": "VigenFlow: Prompt",
    "VigenFlowLoadLora": "VigenFlow: Load LoRA",
    "VigenFlowImageSize": "VigenFlow: Image Size",
    "VigenFlowSaveImage": "VigenFlow: Save Image",
}
