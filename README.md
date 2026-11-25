# MiniMax H3 Image worker (DRAFT — 2026-11-24)

A RunPod Serverless worker for **MiniMax H3 Image** (open-weights still-image
model, 2026-11-10), built on the same pattern as the Qwen-Image worker in
`../worker-qwen/` (which was built on the fkky MiniMax-H3 video worker):
Model Cache + GitHub worker + B2 storage, no monthly fee, pay only for
generation time. One checkpoint does text-to-image and multi-reference
editing (up to 16 reference images, addressed as `<Picture 1>` ..
`<Picture 16>`), at 6 sampling steps, CFG 1, up to ~16 MP.

## Status

**Draft, not yet deployed.** The model files are downloaded and
uploaded to `klassas/minimax-h3-image-slim` (completed 2026-11-25
~06:12 JST, all 3 files verified by size + SHA256). The graph in
`src/workflow.py` was verified 2026-11-25 against ComfyUI master
`comfy_extras/nodes_minimax_h3.py` and both official templates, and
fixed: image keys are `images.image_0` .. (0-based, `<Picture 1>` =
`images.image_0`), outputs are positive = 0 / latent = 1, and the
prompt goes as a STRING straight into `MiniMaxH3ImageConditioning`
(no `CLIPTextEncode` node — the earlier draft had that wrong, and
the Dockerfile build asserts were fixed to match). Everything else
in this worker is copied from the proven Qwen worker and only renamed.

## Model files (Comfy-Org/MiniMax-H3-Image)

| File | Folder | Size |
| --- | --- | --- |
| `minimax_h3_image_pruned_int8_convrot.safetensors` | `diffusion_models` | 21.0 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders` | 15.7 GB |
| `minimax_h3_image_vae_fp16.safetensors` | `vae` | 1.2 GB |

Do NOT point the endpoint at the full Comfy-Org repo (~142 GB, all
variants) — Model Cache downloads the whole snapshot. Create a slim
repo with only these 3 files (same approach as
`klassas/qwen-image-2.1-slim`), e.g. `klassas/h3-image-slim`
(name not final — the local download folder is
`../h3-image-slim-files/`).

## Setup (same shape as the Qwen endpoint)

1. Hugging Face: create the slim repo above and upload the 3 files
   (the local copies are in `../h3-image-slim-files/` once the download
   finishes; upload via the web UI — the files are too big for a chat
   transfer, and the HF token goes via link, never in chat).
2. GitHub: publish this worker as its own repo, e.g.
   `klassas/MiniMax-H3-Image-Comfy` (branch `main`, Dockerfile at
   `/Dockerfile`). **Note the migration decided 2026-11-24:** new
   endpoints are built from `klassas` repos on the new RunPod account,
   not from `maitianyuan44-arch` — the old account's GitHub connection
   drops rebuilds (see Qwen history).
3. RunPod → **Serverless → New Endpoint → Deploy from a GitHub
   repository** → select the repo, branch `main`, Dockerfile path
   `/Dockerfile` → **Queue** (not Load balancer). If the "could not find
   runpod.serverless.start()" warning appears, choose **Continue anyway**
   (the call is at module level in `src/handler.py`).
4. Configure: name e.g. `MiniMax-H3-Image`; GPU **80 GB** × 1 (the
   38 GB model set needs it); Initialization timeout **420**; enable
   Execution timeout and set **1800**; **Cached model** = the slim repo
   id from step 1; Container disk **20**; Max workers **1**, Active
   workers **0**; FlashBoot off (the free allowance of 3 is already
   used by H3 / R2V / Qwen — first generation will take minutes to
   start, generation itself is normal).
5. **Deploy**, then **Manage → Edit endpoint → Advanced → Data centers**:
   keep only AP-JP-1, AP-IN-1, CA-MTL-1, CA-MTL-3, CA-MTL-4. The
   MiniMax H3 Image Community License excludes EU / UK / South Korea /
   USA — Japan is fine, and these are the same five data centers the
   H3 video endpoint uses. Confirm no network volume is attached.
6. **Edit endpoint → Environment variables** (Raw editor): paste the
   five S3 lines (copied from the H3 endpoint) plus
   `HF_REPO=<slim repo id>` → **Update Variables**. (The env editor
   has silently dropped lines before — re-open it and check they stuck,
   as happened on the H3 and Qwen endpoints.)
7. Copy the new **Endpoint ID** into the app: 設定 tab → add a
   "MiniMax H3 Image" model entry (app work, not done yet — the app
   currently has no H3 Image model; it also needs multi-image sending
   for up to 16 references and the character-sheet creation mode).
8. First test in the app: model **MiniMax H3 Image**, a short prompt,
   no image, 2048×2048. The first job takes minutes (38 GB of weights
   load); after that, a 4 MP image should take well under a minute of
   GPU time (6 steps).

## Extra LoRAs (optional, added 2026-11-25)

Up to 3 LoRA slots, the same mechanism as the MiniMax H3 video worker.
Configure by env vars on the endpoint, then restart the worker (no
rebuild): `EXTRA_LORA_KEY=<object key in the B2 bucket>` (or
`EXTRA_LORA_URL=<direct https URL>` — a set URL wins) plus optional
`EXTRA_LORA_STRENGTH=<0-2, default 0.5>`; slots 2/3 use the `_2` / `_3`
suffixes. Files are fetched at boot into ComfyUI's models/loras; a
configured-but-unfetchable LoRA fails the boot on purpose. The app
steers each slot per request with `lora_strength{,_2,_3}` (0 = slot
skipped for that job). First LoRA fitted: "V.H.S - MMH3I - Real
Pussy.safetensors" (Civitai model 3054468, v2.0) in slot 1 via
EXTRA_LORA_KEY — anatomy close-ups are this model's weak area and the
LoRA is trained for exactly that.

## Job contract

Text-to-image: `{"input": {"prompt": "..."}}` → 2048×2048 by default.
Edit: `{"input": {"prompt": "...", "image_base64": "..."}}` (one image)
or `"image_base64s": [...]` / `"image_urls": [...]` (up to 16, in
`<Picture N>` order). With references and no width/height, the output
inherits the first reference's size; with an explicit width/height pair
it uses that canvas (both multiples of 32, total ≤ 20 MP). `steps`
defaults to 6, `cfg_scale` to 1.0. `negative_prompt` is accepted and
ignored — at CFG 1 the model has no negative pass, so anything to avoid
must be written into the positive prompt. Returns
`{"status": "completed", "image_url": ..., "meta": {...}, "timing": {...}}`
with the PNG in B2 (same presigned-link arrangement as the other
endpoints).

## Notes / risks

- **ComfyUI is pinned to 0.46.2 here, not the H3 video worker's
  0.36.0** — `MiniMaxH3ImageConditioning` doesn't exist before 0.45.0.
  The two workers are separate repos/images, so the pins don't conflict.
- **The cuDNN SDPA opt-out is included from the start**
  (`src/sitecustomize.py`, loaded via PYTHONPATH in `src/handler.py`).
  The Qwen worker only learned after its first failures that PyTorch's
  cuDNN SDPA path breaks on this torch cu130 + cuDNN 9 stack
  (`cuDNN Frontend error: No valid execution plans built`); the H3
  transformer shares the same attention path, so this worker doesn't
  wait for a failure to add it.
- **No prompt expander.** The official templates wire a second text
  model ("Generate Text") to expand short prompts; the t2i template
  ships with it off and the edit template with it on. This worker
  always passes the prompt through as typed, matching what the app
  composes (long tile-based prompts). If short prompts underperform,
  the expander is a future addition, not a first-build requirement.
- **Character-sheet mode is app-side only.** The designer (up to 13
  views / 6-expression sheets, wide canvases) is prompt construction
  on top of the stock reference-image graph — the worker needs nothing
  special beyond what is already here. See `../H3_IMAGE_RESEARCH.md`.
- The H3 video endpoints' lessons apply unchanged: env vars can vanish
  from the editor (re-check after saving), GitHub→RunPod rebuilds can
  silently stop (recreate the endpoint rather than Redeploy), and the
  GPU must be Previous-generation 24 GB for Qwen — for this worker the
  80 GB tier has no such Latest/Previous trap noted yet, but check the
  Overview after creation.
