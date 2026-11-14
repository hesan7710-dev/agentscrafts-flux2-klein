# AgentsCrafts — FLUX.2 Klein Serverless Worker

RunPod Queue Serverless worker for image editing with FLUX.2 Klein 4B.

## Status

Implementation draft; source and prompt JSON have been inspected, but Docker build and GPU inference have not yet been verified. Do not deploy to production until the acceptance checks below pass.

## Model

Default: `black-forest-labs/FLUX.2-klein-4B`. Override with `MODEL_ID` only after verifying that the selected model exposes the Diffusers `Flux2KleinPipeline` image-edit interface. Accept the Hugging Face model terms and provide `HF_TOKEN` as a RunPod secret if required by the model repository.

## Input contract

RunPod job input:

- `image_base64` (required; raw base64 or data URL)
- `prompt_type` (optional; one of `general` or the nine surface IDs in `prompts.json`)
- `custom_prompt` (optional; overrides the preset text)
- `seed` (optional integer)
- `full_reconstruction` (optional boolean, default false)
- `reconstruction_suffix` (required only when full_reconstruction=true; V2 prompt library does not contain an approved suffix, so the worker refuses to invent it)

Output is PNG as a presigned S3 URL when S3 settings are configured; otherwise base64 for development only.

## Persistent output storage

Set `S3_BUCKET`, `S3_ENDPOINT_URL`, `S3_REGION`, `S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY` as endpoint secrets. Optional `S3_PREFIX` and `S3_PRESIGNED_URL_SECONDS`. Never commit credentials.

## Build

Build this directory as a Docker image and use it as a custom RunPod Serverless worker image. Set container disk based on dependency installation needs; attach a persistent model cache/volume if you choose volume-based caching. The default HF cache path expects `/runpod-volume` to be mounted; otherwise override `HF_HOME` to writable local storage.

## Acceptance tests before deployment

1. Docker image builds successfully.
2. Worker starts on a CUDA GPU and loads the exact model ID.
3. A known test image produces a valid PNG and output dimensions.
4. Verify default crop-lock prompt behavior: no extension/completion/invention of cropped or hidden parts.
5. Verify all ten prompt keys and that each surface prompt remains byte-for-byte unchanged from the approved source JSON.
6. Verify full reconstruction is rejected unless the separately approved suffix is supplied.
7. Verify S3 upload and presigned download, including expiry.
8. Verify malformed/oversized inputs fail without crashing the worker.

## License notes

FLUX.2 Klein Base model license and upstream Diffusers dependencies must be reviewed before production use.
