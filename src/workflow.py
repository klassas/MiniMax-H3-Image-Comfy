"""ComfyUI node graph for MiniMax H3 Image, built in code per job.

A faithful transcription of ComfyUI's official templates
(Comfy-Org/workflow_templates: image_minimax_h3_image_t2i.json and
image_minimax_h3_image_edit.json), minus the optional prompt-expander
branch (a second text model behind a "Generate Text" node, off by
default in the t2i template and not wired here — the prompt passes
through exactly as typed):

    UNETLoader ──> KSampler.model
    CLIPLoader ──> MiniMaxH3ImageConditioning.clip
    (prompt is a STRING input to MiniMaxH3ImageConditioning itself —
    it tokenizes and encodes internally; there is NO CLIPTextEncode node)
    VAELoader ──> MiniMaxH3ImageConditioning.vae   (edit mode only)
    LoadImage(s) ──> MiniMaxH3ImageConditioning.images.image_N   (edit mode)
    MiniMaxH3ImageConditioning ──> KSampler.positive / .latent_image
    ConditioningZeroOut ──> KSampler.negative
    KSampler ──> VAEDecode ──> SaveImage

Sampling is the model's distilled schedule: 6 steps, CFG 1, euler /
simple, denoise 1.0. With CFG at 1 there is no classifier-free
guidance pass, so there is no negative prompt: anything to avoid must
be written into the positive prompt (the app's negative_prompt field
is accepted and ignored — see src/validator.py).

Text-to-image: no reference images; width/height default to
2048x2048 (the conditioning node's own default). Reference/edit mode:
one LoadImage per reference (max 16) into the conditioning node;
Picture 1 is the image being edited, the rest supply identity,
props, materials or composition, addressed in the prompt as
<Picture 1> .. <Picture 16>. With references and no explicit size,
width/height are 0 and the node reuses the first reference's size.
The VAE is connected in edit mode so references are anchored as
latents as well as read by the text encoder.

VERIFIED (2026-11-25) against ComfyUI master
comfy_extras/nodes_minimax_h3.py (MiniMaxH3ImageConditioning.define_schema)
and both official templates (Comfy-Org/workflow_templates:
image_minimax_h3_image_t2i.json, image_minimax_h3_image_edit.json):
  1. Image input keys are "images.image_0" .. "images.image_15"
     (autogrow group "images", prefix "image_", 0-based). <Picture 1>
     in the prompt is images.image_0, <Picture 2> is images.image_1,
     and so on. The earlier draft's "image_1" .. "image_16" was wrong
     and is fixed below.
  2. Output order is output 0 = positive conditioning, output 1 =
     latent (empty latent in t2i mode) — as written below.
  3. The node takes "clip" (CLIP) and "prompt" (STRING) directly and
     encodes internally (clip.tokenize + encode_from_tokens_scheduled);
     there is no CLIPTextEncode node in either official template, so
     this worker must not build one either. The earlier draft's
     "conditioning" input did not exist and is removed below.
"""

# Model files, as laid out in Comfy-Org/MiniMax-H3-Image (the Model Cache
# snapshot is symlinked into these exact category dirs by model_link.py).
# int8 set: 21.0 GB DiT + 15.7 GB NVFP4 text encoder + 1.2 GB VAE.
UNET_NAME = "minimax_h3_image_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_NAME = "minimax_h3_image_vae_fp16.safetensors"

OUTPUT_NODE_ID = "save_image"

MAX_REFERENCE_IMAGES = 16


def build_workflow(
    *,
    prompt: str,
    width: int = 2048,
    height: int = 2048,
    seed: int = 0,
    steps: int = 6,
    cfg: float = 1.0,
    image_names: list | None = None,
    filename_prefix: str = "h3_image/job",
) -> dict:
    """Returns the API-format graph: {node_id: {"class_type", "inputs"}}.

    image_names: files already uploaded to ComfyUI's input/ directory,
    in <Picture N> order. Empty/None means text-to-image. width/height
    of 0 mean "inherit the first reference's size" (edit mode only).
    """
    image_names = list(image_names or [])

    nodes: dict[str, dict] = {
        "unet_loader": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": UNET_NAME, "weight_dtype": "default"},
        },
        "clip_loader": {
            "class_type": "CLIPLoader",
            # The Qwen3-VL-32B text encoder loads with type "minimax".
            "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"},
        },
        "vae_loader": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE_NAME},
        },
    }

    # No prompt expander: the prompt reaches the model as typed, as a
    # STRING input to the conditioning node (it encodes internally).
    conditioning_inputs: dict = {
        "clip": ["clip_loader", 0],
        "prompt": prompt,
        "width": width,
        "height": height,
    }
    if image_names:
        # Anchor references as latents as well as text-encoder input.
        conditioning_inputs["vae"] = ["vae_loader", 0]
        for i, name in enumerate(image_names):
            # 0-based API key: <Picture 1> = images.image_0.
            node_id = f"load_image_{i + 1}"
            nodes[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
            }
            conditioning_inputs[f"images.image_{i}"] = [node_id, 0]
    nodes["conditioning"] = {
        "class_type": "MiniMaxH3ImageConditioning",
        "inputs": conditioning_inputs,
    }

    # CFG 1 means no guidance pass; the negative is a zeroed-out copy
    # of the positive conditioning, exactly as the official templates do.
    nodes["zero_out"] = {
        "class_type": "ConditioningZeroOut",
        "inputs": {"conditioning": ["conditioning", 0]},
    }

    nodes["sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["unet_loader", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "simple",
            "positive": ["conditioning", 0],
            "negative": ["zero_out", 0],
            # Verified: output 0 = positive, output 1 = latent.
            "latent_image": ["conditioning", 1],
            "denoise": 1.0,
        },
    }
    nodes["decode"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["sampler", 0], "vae": ["vae_loader", 0]},
    }
    nodes[OUTPUT_NODE_ID] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["decode", 0], "filename_prefix": filename_prefix},
    }
    return nodes
