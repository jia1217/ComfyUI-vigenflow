#!/usr/bin/env bash
# Start ComfyUI with the VigenFlow nodes.
#
#   ./start.sh                  # then open http://127.0.0.1:8188
#   ./start.sh --port 8190      # extra arguments go to ComfyUI's main.py
#
# Cloned into ComfyUI/custom_nodes/: starts that ComfyUI. Cloned anywhere else: links this folder
# into $COMFYUI_DIR/custom_nodes/ (default ~/ComfyUI) and starts that ComfyUI. ComfyUI runs with
# --cpu (its own PyTorch nodes on the CPU); the VigenFlow nodes always run on the NPU.
# COMFYUI_PYTHON chooses the Python; by default ComfyUI's .venv or venv, else python3.
set -euo pipefail

node_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
node_name="$(basename -- "$node_dir")"
parent="$(dirname -- "$node_dir")"

if [[ "$(basename -- "$parent")" == custom_nodes && -f "$parent/../main.py" ]]; then
    comfyui_dir="$(cd -- "$parent/.." && pwd -P)"
else
    comfyui_dir="${COMFYUI_DIR:-$HOME/ComfyUI}"
    if [[ ! -f "$comfyui_dir/main.py" ]]; then
        printf 'ComfyUI was not found at %s. Set COMFYUI_DIR to your ComfyUI folder,\n' "$comfyui_dir" >&2
        printf 'or clone this repository into ComfyUI/custom_nodes/.\n' >&2
        exit 1
    fi
    comfyui_dir="$(cd -- "$comfyui_dir" && pwd -P)"
fi

# Another copy of these nodes in the same ComfyUI would replace them (same node names).
for other in "$comfyui_dir"/custom_nodes/*; do
    [[ -d "$other" && -f "$other/nodes.py" ]] || continue
    [[ "$(readlink -f -- "$other")" == "$node_dir" ]] && continue
    if grep -q 'class VigenFlowTextToImage' "$other/nodes.py"; then
        printf 'Another copy of the VigenFlow nodes is installed: %s\n' "$other" >&2
        printf 'Remove it (or move it out of custom_nodes) and start again.\n' >&2
        exit 1
    fi
done

# Cloned elsewhere: link this folder into that ComfyUI (after the check above).
link="$comfyui_dir/custom_nodes/$node_name"
if [[ "$(dirname -- "$node_dir")" != "$comfyui_dir/custom_nodes" ]]; then
    if [[ -L "$link" && "$(readlink -f -- "$link")" == "$node_dir" ]]; then
        :
    elif [[ -e "$link" || -L "$link" ]]; then
        printf '%s already exists and is not this folder (%s).\n' "$link" "$node_dir" >&2
        exit 1
    else
        mkdir -p -- "$comfyui_dir/custom_nodes"
        ln -s -- "$node_dir" "$link"
        printf 'Linked %s into %s\n' "$node_dir" "$comfyui_dir/custom_nodes/"
    fi
fi

cd -- "$comfyui_dir"
python=python3
if [[ -x .venv/bin/python ]]; then
    python=.venv/bin/python
elif [[ -x venv/bin/python ]]; then
    python=venv/bin/python
fi
python="${COMFYUI_PYTHON:-$python}"

printf 'Starting ComfyUI in %s\n' "$comfyui_dir"
if [[ -n "${VIGENFLOW_DRY_RUN:-}" ]]; then   # for testing: show the command only
    printf '%s main.py --cpu %s\n' "$python" "$*"
    exit 0
fi
exec "$python" main.py --cpu "$@"
