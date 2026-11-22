# ComfyUI-vigenflow

Text-to-image and image editing on the **AMD Ryzen AI NPU** in ComfyUI, with the
[VigenFlow](https://github.com/jia1217/vigenflow) engine: FLUX.2 Klein 4B, FLUX.1 Schnell, Krea 2 Turbo,
Qwen-Image 2.1 Turbo and Z-Image Turbo, and FLUX.2 Klein 4B Edit (1–4 reference images).
Everything runs on the NPU: about 8 s per 1024×1024 FLUX.2 image, 17 s per FLUX.2 edit.

## Quick Start

1. Clone into your ComfyUI custom nodes folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/jia1217/ComfyUI-vigenflow.git
   ```

2. Restart ComfyUI. On first use the node downloads the VigenFlow engine (about 35 MB, checked
   against its published SHA256) into `ComfyUI-vigenflow/engine/`. ComfyUI-Manager installs it
   right away; to do that by hand, run `python install.py` in this folder.

3. Open **Templates → Extensions → ComfyUI-vigenflow** and choose **vigenflow_text_to_image**
   or **vigenflow_image_edit** (it edits ComfyUI's `input/example.png`; add a Load Image node to
   Edit Prompt's `image_2` for a second reference)
   (or drag [`example_workflows/vigenflow_text_to_image.json`](example_workflows/vigenflow_text_to_image.json)
   onto the canvas), pick a model and click **Run**. The first image of each model downloads its
   weights into `ComfyUI/models/vigenflow/<model>/` (8–27 GB; progress in the ComfyUI console).

**Start script:** `./start.sh` starts ComfyUI with these nodes (`--cpu` for ComfyUI's own
PyTorch nodes; VigenFlow always runs on the NPU). Extra arguments go to ComfyUI, for example
`./start.sh --port 8190`. Cloned into `ComfyUI/custom_nodes/`, it starts that ComfyUI; cloned
anywhere else, it links this folder into `$COMFYUI_DIR/custom_nodes/` (default `~/ComfyUI`) first.
It stops if another copy of the VigenFlow nodes is installed in the same ComfyUI.

**Requirements:** Linux on x86-64, a Ryzen AI processor with an XDNA 2 NPU (`/dev/accel/accel0`),
AMD XRT with the NPU driver, and Boost.ProgramOptions 1.83
(`sudo apt install libboost-program-options1.83.0`). The node checks these and says what is missing.

## Model licenses

Krea 2 Turbo (Krea 2 Community License) and Qwen-Image 2.1 Turbo (Qwen Research License,
research and evaluation only) download only after you accept their terms. Read them on
[Krea 2 Turbo](https://huggingface.co/krea/Krea-2-Turbo) and
[Qwen-Image 2.1 Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo), then create
`vigenflow_settings.json` in this folder:

```json
{ "accept_licenses": ["krea2_turbo", "qwen_image_2_1_turbo"] }
```

## Nodes (category NPU)

| Node | Use |
|---|---|
| **VigenFlow: Text to Image (NPU)** | Model, prompt, size, steps, seed and LoRA; returns the image and shows the runtimes |
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
FLUX.2 Edit 4).

**Editing output size** (on Image Edit, or from Edit Output Size), as in the VigenFlow Studio:
`keep_ratio` (default on) fits the first image's shape inside the chosen size (1024 × 1024 →
long side 1024); `match_input` uses the first image's own size (up to 2048); with both off the
chosen size is used as is. Each reference keeps its own size (as in the VigenFlow Studio):
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

- One model runs at a time; the NPU takes one job. Each image starts the model's worker,
  loads it (15–30 s at 1024 for FLUX.2), generates and releases it, so the NPU and RAM are free
  between images. Several generator nodes in a graph run one after another.
- Another NPU job (for example a VigenFlow server) can make the worker fail with
  `CREATE_HWCTX ... err=-22`; queue the image again when it has finished.
- Cancel in ComfyUI stops the worker or the download.

## Settings (`vigenflow_settings.json`, optional)

| Key | Default |
|---|---|
| `accept_licenses` | `[]`: ids of models whose license you accept |
| `output_dir` | ComfyUI's output folder (used by VigenFlow: Save Image) |
| `weights_dir` | `ComfyUI/models/vigenflow` |
| `engine_dir` | `ComfyUI-vigenflow/engine/<version>` |
| `models` | Per-model changes, for example `{"Z-Image Turbo": {"weights_path": "/data/z_image_turbo"}}` |

Paths are absolute or relative to this folder. `VIGENFLOW_ENGINE_ZIP=/path/to/vgf_…_ubuntu_amd64.zip`
installs the engine from a downloaded copy (for computers without internet access).

## Test without ComfyUI

```bash
python install.py
python worker.py --list
```

## License

The ComfyUI-vigenflow source code is licensed under the [Apache License 2.0](LICENSE).

Model weights are **not** covered by this license. Each model and LoRA keeps its original
license; Krea 2 Turbo (Krea 2 Community License, limited commercial use) and Qwen-Image 2.1 Turbo
(Qwen Research License, research and evaluation only) are not Apache 2.0. See VigenFlow's
[MODEL_LICENSES.md](https://github.com/jia1217/vigenflow/blob/main/MODEL_LICENSES.md) for each model.
