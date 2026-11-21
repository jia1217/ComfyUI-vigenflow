import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const widgetName = "vigenflow_denoising_runtime";
const widgetHeight = 108;
const generatorNodes = new Set(["VigenFlowTextToImage", "Flux2KleinNPU"]);
const sizeNodes = new Set([...generatorNodes, "VigenFlowImageSize"]);
const CUSTOM = "Custom";

const findWidget = (node, name) => node.widgets?.find((widget) => widget.name === name);

function inputLinked(node, name) {
    const slot = node.findInputSlot(name);
    return slot >= 0 && node.isInputConnected(slot);
}

const DEFAULT_SIZE = "1024 × 1024";

function gcd(a, b) { return b ? gcd(b, a % b) : a; }

// "1024 × 768" -> [1024, 768]
function parseSize(label) {
    const sides = String(label).split("×").map((side) => Number(side.trim()));
    return sides.length === 2 && sides.every(Number.isInteger) ? sides : null;
}

function ratioOf(label) {
    const sides = parseSize(label);
    if (!sides) return null;
    const g = gcd(sides[0], sides[1]);
    return `${sides[0] / g}:${sides[1] / g}`;
}

// The server sends every size; the browser offers the ones of the chosen ratio.
function allSizes(size) {
    size.vigenflowAllSizes ??= [...(size.options?.values ?? [])];
    return size.vigenflowAllSizes;
}

function sizesFor(size, ratio) {
    return allSizes(size).filter((label) => ratioOf(label) === ratio);
}

function closestToOneMegapixel(labels) {
    const area = (label) => { const [w, h] = parseSize(label); return Math.abs(w * h - 1024 * 1024); };
    return labels.reduce((best, label) => (area(label) < area(best) ? label : best), labels[0]);
}

// aspect_ratio -> size list of that ratio; Custom -> width and height. A connected latent_size
// replaces all of them; connected width/height inputs (e.g. from VigenFlow: Image Size) need Custom.
function updateSizeWidgets(node) {
    const ratio = findWidget(node, "aspect_ratio");
    const size = findWidget(node, "size");
    const fromLatent = inputLinked(node, "latent_size");
    const fromInputs = inputLinked(node, "width") || inputLinked(node, "height");
    const fromSizeNode = inputLinked(node, "image_size");   // VigenFlow: Image Size, one wire
    if (ratio && fromInputs && ratio.value !== CUSTOM) ratio.value = CUSTOM;
    const custom = !ratio || ratio.value === CUSTOM;
    if (ratio) ratio.hidden = fromLatent || fromInputs || fromSizeNode;
    if (size) {
        if (!custom) {
            const labels = sizesFor(size, ratio.value);
            size.options.values = labels;
            if (!labels.includes(size.value)) size.value = closestToOneMegapixel(labels);
        }
        size.hidden = fromLatent || fromSizeNode || custom;
    }
    for (const name of ["width", "height"]) {
        const widget = findWidget(node, name);
        if (widget) widget.hidden = fromLatent || fromSizeNode || !custom;
    }
    const [width] = node.size;
    node.setSize([width, node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

// A connected lora_stack (VigenFlow: Load LoRA) replaces the lora widgets, and a connected
// prompt_text (VigenFlow: Prompt) the generator's own prompt box.
function updateLoraWidgets(node) {
    // keep_loaded is always off: the model is released after every image.
    const keep = findWidget(node, "keep_loaded");
    if (keep) {
        keep.value = false;
        keep.hidden = true;
    }
    const fromStack = inputLinked(node, "lora_stack");
    for (const name of ["lora", "lora_scale", "lora_file"]) {
        const widget = findWidget(node, name);
        if (widget) widget.hidden = fromStack;
    }
    const prompt = findWidget(node, "prompt");
    if (prompt) prompt.hidden = inputLinked(node, "prompt_text");
    const [width] = node.size;
    node.setSize([width, node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

// Saved graphs keep widget values by position. Generator nodes are tagged with LAYOUT; an
// untagged node is matched to an older layout by the kinds of its values (the front end may drop
// a trailing empty value, so counts alone are not enough), its values are taken by name, and
// sizes without a ratio become the matching ratio and size, or Custom with the same sides.
const LAYOUT = 3;
const SEED = ["seed", "control_after_generate"];
const OLD_LAYOUTS = {
    VigenFlowTextToImage: [
        ["model", "prompt", ...SEED, "steps", "aspect_ratio", "size", "width", "height", "lora", "lora_scale",
         "keep_loaded", "lora_file"],
        ["model", "prompt", ...SEED, "steps", "size", "width", "height", "lora", "lora_scale", "keep_loaded", "lora_file"],
        ["model", "prompt", ...SEED, "steps", "width", "height", "lora", "lora_scale", "keep_loaded", "lora_file"],
    ],
    VigenFlowImageSize: [["size", "width", "height"], ["width", "height"]],
};
const DEFAULTS = { lora: "none", lora_scale: 1, lora_file: "", keep_loaded: false };

const isNumber = (v) => typeof v === "number";
const KIND = {
    seed: isNumber, steps: isNumber, width: isNumber, height: isNumber, lora_scale: isNumber,
    keep_loaded: (v) => typeof v === "boolean",
    aspect_ratio: (v) => v === CUSTOM || /^\d+:\d+$/.test(v),
    size: (v) => v === CUSTOM || parseSize(v) !== null,
};

function fits(layout, values) {
    if (values.length > layout.length || values.length < layout.length - 1) return false;
    return values.every((value, i) => {
        const check = KIND[layout[i]];
        return check ? check(value) : typeof value === "string";
    });
}

function migrateOldValues(node, info) {
    const layouts = OLD_LAYOUTS[node.type];
    const values = info?.widgets_values;
    if (!layouts || !Array.isArray(values) || info.properties?.vigenflow_layout === LAYOUT) return;
    const current = node.widgets.filter((w) => w.serialize !== false).map((w) => w.name);
    if (fits(current, values)) return;                       // already in today's order
    const names = layouts.find((layout) => fits(layout, values));
    if (!names) return;
    const byName = Object.fromEntries(names.map((name, i) => [name, i < values.length ? values[i] : DEFAULTS[name]]));
    if (!("aspect_ratio" in byName)) {
        const old = byName.size;
        const sides = parseSize(old);
        const size = findWidget(node, "size");
        if (old !== undefined && old !== CUSTOM && size && allSizes(size).includes(old)) {
            Object.assign(byName, { aspect_ratio: ratioOf(old), size: old });
        } else {
            Object.assign(byName, { aspect_ratio: CUSTOM, size: DEFAULT_SIZE });
            if (sides) [byName.width, byName.height] = sides;
        }
    }
    for (const widget of node.widgets) {
        if (widget.serialize === false) continue;
        if (widget.name in byName && byName[widget.name] !== undefined) widget.value = byName[widget.name];
        else if (widget.name in DEFAULTS) widget.value = DEFAULTS[widget.name];
    }
}

// VigenFlow: Load LoRA lists only the LoRAs of its model: {model: ["none", names...]} from
// the server (loras.jsonc entries and ComfyUI/models/loras/<model folder>/ files).
let loraNames = null;

async function fetchLoraNames() {
    try {
        const response = await api.fetchApi("/vigenflow/loras");
        loraNames = response.ok ? await response.json() : null;
    } catch {
        loraNames = null;
    }
}

function updateLoraNames(node) {
    const model = findWidget(node, "model");
    const lora = findWidget(node, "lora_name");
    if (!model || !lora || !loraNames) return;
    const names = loraNames[model.value] ?? ["none"];
    lora.options.values = names;
    if (!names.includes(lora.value)) lora.value = "none";
    node.setDirtyCanvas(true, true);
}

// Load LoRA nodes saved before the model widget: [lora_name, strength], with
// "loras.jsonc: <name>" for configured LoRAs. The model is the one that lists the name.
function migrateLoadLora(node, info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || values.length !== 2 || typeof values[0] !== "string" ||
        typeof values[1] !== "number") return;
    const name = values[0].replace(/^loras\.jsonc: /, "");
    const model = Object.keys(loraNames ?? {}).find((m) => loraNames[m].includes(name));
    if (model) findWidget(node, "model").value = model;
    findWidget(node, "lora_name").value = model ? name : "none";
    findWidget(node, "strength").value = values[1];
}

function hookLoadLora(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
        const result = onNodeCreated?.apply(this, args);
        const model = findWidget(this, "model");
        if (model) {
            const callback = model.callback;
            model.callback = (...cbArgs) => {
                const value = callback?.apply(model, cbArgs);
                updateLoraNames(this);
                return value;
            };
        }
        updateLoraNames(this);
        return result;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
        onConfigure?.call(this, info);
        migrateLoadLora(this, info);
        updateLoraNames(this);
    };
}

// Graphs saved with two wires (Image Size width/height -> generator width/height) are rewired
// to the single image_size wire; Image Size now has only that output.
function rewireImageSize(graph) {
    const nodes = graph?._nodes ?? [];
    for (const node of nodes) {
        if (!generatorNodes.has(node.type)) continue;
        const target = node.findInputSlot("image_size");
        let source = null;
        for (const name of ["width", "height"]) {
            const slot = node.findInputSlot(name);
            const link = slot >= 0 ? graph.links[node.inputs[slot].link] : null;
            if (!link) continue;
            const origin = graph.getNodeById(link.origin_id);
            if (origin?.type !== "VigenFlowImageSize") continue;
            source = origin;
            node.disconnectInput(slot);
        }
        if (source && target >= 0 && !inputLinked(node, "image_size")) source.connect(0, node, target);
        if (source) updateSizeWidgets(node);
    }
    for (const node of nodes) {   // outputs that saved graphs still list: keep one image_size
        if (node.type !== "VigenFlowImageSize" || !node.outputs?.length) continue;
        let keep = node.outputs.findIndex((o) => o.name === "image_size" && o.links?.length);
        if (keep < 0) keep = node.outputs.findIndex((o) => o.name === "image_size");
        if (keep < 0) keep = 0;
        for (let i = node.outputs.length - 1; i >= 0; i--) {
            if (i !== keep) node.removeOutput(i);
        }
        Object.assign(node.outputs[0], { name: "image_size", type: "VIGENFLOW_SIZE" });
    }
}

app.registerExtension({
    name: "VigenFlow.DenoisingRuntime",

    afterConfigureGraph() {
        rewireImageSize(app.graph);
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "VigenFlowLoadLora") {
            await fetchLoraNames();
            hookLoadLora(nodeType);
            return;
        }
        if (!sizeNodes.has(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            const ratio = findWidget(this, "aspect_ratio");
            if (ratio) {
                const callback = ratio.callback;
                ratio.callback = (...cbArgs) => {
                    const value = callback?.apply(ratio, cbArgs);
                    updateSizeWidgets(this);
                    return value;
                };
            }
            if (nodeData.name === "VigenFlowTextToImage") this.properties.vigenflow_layout = LAYOUT;
            updateSizeWidgets(this);
            updateLoraWidgets(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            onConfigure?.call(this, info);
            migrateOldValues(this, info);
            if (nodeData.name === "VigenFlowTextToImage") {
                this.properties.vigenflow_layout = LAYOUT;
                const latent = this.findInputSlot("latent_size");   // removed input of older graphs
                if (latent >= 0) this.removeInput(latent);
            }
            // Saved workflows can still carry the old text outputs.
            for (const name of ["memory", "timings"]) {
                const slot = this.findOutputSlot(name);
                if (slot >= 0) this.removeOutput(slot);
            }
            updateSizeWidgets(this);
            updateLoraWidgets(this);
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            onConnectionsChange?.apply(this, args);
            const name = args[4]?.name;
            if (["latent_size", "width", "height", "image_size"].includes(name)) updateSizeWidgets(this);
            if (name === "lora_stack" || name === "prompt_text") updateLoraWidgets(this);
        };

        if (!generatorNodes.has(nodeData.name)) return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            onExecuted?.call(this, output);
            const summary = output?.text?.[0];
            if (typeof summary !== "string") return;

            let widget = findWidget(this, widgetName);
            if (!widget) {
                const element = document.createElement("div");
                Object.assign(element.style, {
                    boxSizing: "border-box",
                    padding: "4px 8px",
                    lineHeight: "20px",
                    whiteSpace: "pre-line",
                    color: "var(--input-text, #ddd)",
                });
                widget = this.addDOMWidget(widgetName, "div", element, {
                    serialize: false,
                    hideOnZoom: false,
                    getMinHeight: () => widgetHeight,
                    getMaxHeight: () => widgetHeight,
                });
                widget.serialize = false;
                this.setSize([this.size[0], Math.max(this.size[1], this.computeSize()[1])]);
            }

            widget.element.textContent = summary;
            this.setDirtyCanvas(true, true);
        };
    },
});
