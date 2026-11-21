# ComfyUI-vigenflow

Text-to-image on the **AMD Ryzen AI NPU** in ComfyUI, with the [VigenFlow](https://github.com/jia1217/vigenflow)
engine: FLUX.2 Klein 4B, FLUX.1 Schnell, Krea 2 Turbo, Qwen-Image 2.1 Turbo and Z-Image Turbo.
Text encoder, denoising and VAE decoder all run on the NPU (about 8 s per 1024×1024 FLUX.2 image).

## Quick Start

1. Clone into your ComfyUI custom nodes folder:

   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/jia1217/ComfyUI-vigenflow.git
   ```

2. Restart ComfyUI. On first use the node downloads the VigenFlow engine (about 35 MB, checked
   against its published SHA256) into `ComfyUI-vigenflow/engine/`. ComfyUI-Manager installs it
   right away; to do that by hand, run `python install.py` in this folder.

3. Drag [`workflows/vigenflow_text_to_image.json`](workflows/vigenflow_text_to_image.json) onto
   the canvas, pick a model and click **Run**. The first image of each model downloads its
   weights into `ComfyUI/models/vigenflow/<model>/` (8–27 GB; progress in the ComfyUI console).

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
| **VigenFlow: Prompt** | A prompt box; connect to `prompt_text` (one prompt can feed several models) |
| **VigenFlow: Image Size** | Aspect ratio and size; connect `image_size` |
| **VigenFlow: Load LoRA** | One LoRA of one model with its strength; chain several into `lora_stack` |
| **VigenFlow: Save Image** | Saves to `output_dir` from `vigenflow_settings.json`, else ComfyUI's output folder |

**Size:** `aspect_ratio` (1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16) with sizes of that ratio, or
`Custom` width and height (16–2048, multiples of 16).
**Steps:** 0 uses the model's default (FLUX.2 4, FLUX.1 4, Krea 2 8, Qwen-Image 8, Z-Image 8).

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
