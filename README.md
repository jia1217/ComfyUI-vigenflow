# ComfyUI-vigenflow

Text-to-image and image editing on the **AMD Ryzen AI NPU** in ComfyUI, with the
[VigenFlow](https://github.com/jia1217/vigenflow) engine: FLUX.2 Klein 4B, FLUX.1 Schnell, Krea 2 Turbo,
Qwen-Image 2.1 Turbo and Z-Image Turbo, and the edit models FLUX.2 Klein 4B Edit (1–4 reference images)
and Qwen-Image 2.1 Turbo Edit (1–10), and the SeedVR2 3B upscaler, on **Ubuntu and Windows**.
Everything runs on the NPU: about 8 s per 1024×1024 FLUX.2 image, 17 s per FLUX.2 edit, 55 s per
two-image Qwen-Image edit, 23 s per 2048 × 2048 upscale (NPU time; loading the model comes on top,
see [Behaviour](#behaviour)).

## Requirements

| | Ubuntu | Windows |
|---|---|---|
| System | Ubuntu 24.04 (x86-64) | Windows 11 (x86-64) |
| Processor | Ryzen AI with an XDNA 2 NPU | Ryzen AI with an XDNA 2 NPU |
| NPU driver | AMD XRT with the amdxdna driver (`/dev/accel/accel0`) | AMD NPU driver ("NPU Compute Accelerator Device" in Device Manager) |
| Libraries | Boost.ProgramOptions 1.83: `sudo apt install libboost-program-options1.83.0` | none: the release ships its DLLs |
| VigenFlow engine | `vgf_0.2.0-beta.5_ubuntu_amd64.zip` (about 35 MB) | `vgf_0.2.0-beta.5_windows_amd64.zip` (about 39 MB) |

The node picks the engine for your system, downloads it from the
[VigenFlow release](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.5) on first use and
checks its SHA256. It also checks the NPU and libraries and says what is missing. Any ComfyUI
install works: a git clone with a virtual environment, the Windows portable build or ComfyUI
Desktop. ComfyUI's own PyTorch nodes can run on the CPU (`--cpu`); the VigenFlow nodes always use the NPU.

## Quick Start

1. Clone into your ComfyUI custom nodes folder.

   Ubuntu:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/jia1217/ComfyUI-vigenflow.git
   ```

   Windows (PowerShell or Command Prompt; for ComfyUI Desktop the folder is
   `Documents\ComfyUI\custom_nodes`):

   ```bat
   cd ComfyUI\custom_nodes
   git clone https://github.com/jia1217/ComfyUI-vigenflow.git
   ```

2. Start (or restart) ComfyUI, with the start script below or as you usually do. On first use the
   node downloads the VigenFlow engine into `ComfyUI-vigenflow/engine/`. ComfyUI-Manager installs it
   right away; to do that by hand, run `python install.py` in this folder.

3. Open http://127.0.0.1:8188, then **Templates → Extensions → ComfyUI-vigenflow** and choose
   **vigenflow_text_to_image** or **vigenflow_image_edit** (it edits ComfyUI's `input/example.png`;
   add a Load Image node to Edit Prompt's `image_2` for a second reference) or **vigenflow_upscale**
   (or drag [`example_workflows/vigenflow_text_to_image.json`](example_workflows/vigenflow_text_to_image.json)
   onto the canvas), pick a model and click **Run**. The first image of each model downloads its
   weights into `ComfyUI/models/vigenflow/<model>/` (8–27 GB; progress in the ComfyUI console).
   Already have the weights (for example the VigenFlow package's `all_model_weights`)? Set
   `weights_dir` in [`vigenflow_settings.json`](#settings-vigenflow_settingsjson-optional) first.

## Start scripts

| Ubuntu | Windows |
|---|---|
| `./start.sh` | `start_windows.bat` (double-click, or run it in a terminal) |
| `./start.sh --port 8190` | `start_windows.bat --port 8190` |

Both start ComfyUI with these nodes and `--cpu`; extra arguments go to ComfyUI's `main.py`.
Cloned into `ComfyUI/custom_nodes/`, the script starts that ComfyUI; cloned anywhere else, it
first links this folder into `COMFYUI_DIR/custom_nodes/` (default `~/ComfyUI` on Ubuntu,
`%USERPROFILE%\ComfyUI` on Windows; a directory junction there). It stops if another copy of the
VigenFlow nodes is installed in the same ComfyUI. It uses ComfyUI's `.venv` or `venv` (on Windows
also the portable build's `python_embeded`), else `python3` / `python`; set `COMFYUI_PYTHON` to
choose another. `start_windows.bat` also switches the console to UTF-8 so the NPU progress bars
show correctly. With ComfyUI Desktop, restart the app instead.

## Model licenses

Krea 2 Turbo (Krea 2 Community License) and Qwen-Image 2.1 Turbo (Qwen Research License,
research and evaluation only) download only after you accept their terms. Read them on
[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) and
[Qwen-Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), then turn on
**accept_license** on the Text to Image or Image Edit node (it means "I agree"; it is greyed out for
models that do not need it). To accept once for every workflow instead, create
`vigenflow_settings.json` in this folder and restart ComfyUI:

```json
{ "accept_licenses": ["krea2_turbo", "qwen_image_2_1_turbo"] }
```

## Nodes (category NPU)

| Node | Use |
|---|---|
| **VigenFlow: Text to Image (NPU)** | Model, prompt, size, steps, seed and LoRA; returns the image and shows the runtimes |
| **VigenFlow: Upscale (NPU)** | SeedVR2 3B: an image to a square `upscaling_size` (16–2048); the input is center-cropped to a square first |
| **VigenFlow: Image Edit (NPU)** | `edit_input` (or `references` / one `image` plus its own prompt box); the same controls |
| **VigenFlow: Edit Prompt** | The edit prompt and its images in one node: connect a Load Image node (each reference its own node) to `image_1`–`image_4`; one wire to `edit_input`, and `image` (the first) for Compare Images |
| **VigenFlow: Reference Images** | The edit images in one node: four load-image cells (click or drop an image on a cell; **upload images** takes several files in order), or pick from the `image_1`–`image_4` lists, and/or `images` from other nodes; one wire to `references`, and `image` (the first) for Compare Images |
| **VigenFlow: Prompt** | A prompt box; connect to `prompt_text` (one prompt can feed several models) |
| **VigenFlow: Image Size** | Aspect ratio and size; connect `image_size` |
| **VigenFlow: Edit Output Size** | The edit output size with the VigenFlow Studio's options `keep_ratio` and `match_input`; connect `image_size` |
| **VigenFlow: Compare Images** | `before` and `after` with a slider: move the mouse over it (works in the classic node display) |
| **VigenFlow: Load LoRA** | One LoRA of one model with its strength; chain several into `lora_stack` |
| **VigenFlow: Save Image** | Saves to `output_dir` from `vigenflow_settings.json`, else ComfyUI's output folder |

**Size:** `aspect_ratio` (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16) with sizes of that ratio, or
`Custom` width and height (16–2048, multiples of 16).
**Steps:** 0 uses the model's default (FLUX.2 4, FLUX.1 4, Krea 2 8, Qwen-Image 8, Z-Image 8,
FLUX.2 Edit 4, Qwen-Image Edit 8).

**Qwen-Image 2.1 Turbo Edit** uses the Qwen-Image 2.1 Turbo weights (and the same license,
see Model licenses) and takes up to 10 references within 16,384 reference tokens: connect
image_1–image_4 on Edit Prompt and more through its `references` input (VigenFlow: Reference Images).

**Editing output size** (on Image Edit, or from Edit Output Size), as in the VigenFlow Studio:
`keep_ratio` (default on) fits the first image's shape inside the chosen size (1024 × 1024 →
long side 1024); `match_input` uses the image's own size (up to 2048); with both off the
chosen size is used as is. `size_from_image` (default 1, the image being edited) picks which
image keep_ratio and match_input use. Each reference keeps its own size (as in the VigenFlow Studio):
`input_limit` (Auto = 2048, or 512 / 1024 / 2048) only scales down references whose long edge is
larger, and sides are rounded down to multiples of 16. Auto also keeps the total within the model's
reference token budget (`reference_token_limit` in `models.json`; FLUX.2 Edit 65,536, which four
2048 × 2048 references fill exactly), reducing larger references first. Each 16 × 16 block is one token, so smaller
references edit faster. FLUX.2 Klein 4B Edit uses the FLUX.2 Klein 4B weights (no extra download) and
its LoRAs (folder `flux2_klein_4B`).

## LoRAs

Set **Load LoRA**'s `model` to the generator's model; `lora_name` lists only that model's LoRAs:
the ones the engine ships with (downloaded on first use) and your own `.safetensors` files in
`ComfyUI/models/loras/<id>/`:

| Model | id (folder name) |
|---|---|
| FLUX.2 Klein 4B | `flux2_klein_4B` |
| FLUX.1 Schnell | `flux1_schnell` |
| Krea 2 Turbo | `krea2_turbo` |
| Qwen-Image 2.1 Turbo | `qwen_image_2_1_turbo` |
| Z-Image Turbo | `z_image_turbo` |

To delete one of your files, delete it from that folder and reload the page.

**Your own LoRA list.** For names, default strengths, trigger words or downloads, or to remove
one of the engine's LoRAs from the list, create `ComfyUI/models/loras/<id>/loras.jsonc`:

```jsonc
{
  "hide": ["Wall Street Journal Style"],          // engine LoRAs to leave out of the list
  "loras": [                                      // your own entries
    { "name": "My Style", "file": "my_style.safetensors", "scale": 0.8,
      "trigger": "my style", "source": "https://huggingface.co/<user>/<repo>" }
  ]
}
```

`file` is relative to that folder (or absolute); with a `source`, a missing file is downloaded
on first use. An entry with an engine LoRA's name replaces it. Reload the page after editing.
Do not edit `engine/.../loras.jsonc`: the node rebuilds it from the release list and this file.

Each LoRA keeps its own strength. LoRAs of different models cannot be mixed in one stack.
Include a LoRA's trigger words in the prompt (they are printed in the console).

## Behaviour

- One model runs at a time; the NPU takes one job. Each image starts the model's run.exe,
  loads the model, generates and releases it, so the NPU and RAM are free between images.
  Several generator nodes in a graph run one after another.
- Loading adds time to every image: 15–30 s at 1024 for FLUX.2 on Ubuntu, 15–55 s depending on
  the model on Windows. On Ubuntu the node uses run.exe's worker mode; the Windows run.exe has
  none, so there each image is one ordinary run (the same result).
- Windows: the first time a model's weights are opened, Microsoft Defender may scan them, which
  can make that first load take minutes. Later loads are normal.
- Another NPU job (for example the VigenFlow server or Studio) can make the run fail
  (`CREATE_HWCTX ... err=-22` on Ubuntu, `0xc01e0009` on Windows); queue the image again when it
  has finished.
- Cancel in ComfyUI stops the run or the download.
- Peak RAM of the model process is printed for each image: about 8 GB for FLUX.2, up to 18 GB
  for Krea 2.

## Settings (`vigenflow_settings.json`, optional)

| Key | Default |
|---|---|
| `accept_licenses` | `[]`: ids of models whose license you accept |
| `output_dir` | ComfyUI's output folder (used by VigenFlow: Save Image) |
| `weights_dir` | `ComfyUI/models/vigenflow` |
| `engine_dir` | `ComfyUI-vigenflow/engine/<version>` |
| `models` | Per-model changes, for example `{"Z-Image Turbo": {"weights_path": "/data/z_image_turbo"}}` |

Paths are absolute or relative to this folder. On Windows write them with `/` or doubled `\\`
(JSON), for example:

```json
{ "weights_dir": "C:/VigenFlow/all_model_weights" }
```

Restart ComfyUI after changing this file.

**Without internet access**, download the engine zip for your system from the
[release](https://github.com/jia1217/vigenflow/releases/tag/v0.2.0-beta.5) on another computer and
install it from that copy (the SHA256 is still checked):

```bash
VIGENFLOW_ENGINE_ZIP=/path/to/vgf_0.2.0-beta.5_ubuntu_amd64.zip python install.py
```

```powershell
$env:VIGENFLOW_ENGINE_ZIP = "C:\Downloads\vgf_0.2.0-beta.5_windows_amd64.zip"; python install.py
```

## Test without ComfyUI

In this folder, with ComfyUI's Python (it needs no other packages):

```bash
python install.py
python worker.py --list
python worker.py --model "FLUX.2 Klein 4B" --prompt "a red fox in the snow" --output fox.png
```

`--list` shows each model and its run.exe; the last line generates one image and prints its
timings and peak RAM.

## License

The ComfyUI-vigenflow source code is licensed under the [Apache License 2.0](LICENSE).

Model weights are **not** covered by this license. Each model and LoRA keeps its original
license; Krea 2 Turbo (Krea 2 Community License, limited commercial use) and Qwen-Image 2.1 Turbo
(Qwen Research License, research and evaluation only) are not Apache 2.0. See VigenFlow's
[MODEL_LICENSES.md](https://github.com/jia1217/vigenflow/blob/main/MODEL_LICENSES.md) for each model.
