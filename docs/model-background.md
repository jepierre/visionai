# Vision Model Background

## What is a vision-language model?

A vision-language model (VLM) accepts visual input, usually an image, together with natural-language text. It converts visual features and text tokens into a shared reasoning process and generates a text response. Depending on its architecture and training, a VLM may answer questions, describe a scene, read text, locate objects, or produce structured output.

A VLM is not automatically an object detector. Many VLMs are good at describing an image but are not reliable at returning exact object counts, pixel masks, or stable coordinates. Those tasks require explicit grounding capabilities and careful evaluation.

VisionAI therefore separates two jobs:

- **Grounding:** identify instances, locations, boxes, masks, and counts that can be checked against the image.
- **Reasoning:** interpret the image and grounded evidence, answer natural-language questions, and explain results conversationally.

## Falcon Perception

[Falcon Perception](https://github.com/tiiuae/Falcon-Perception) is a natively multimodal, dense autoregressive model with perception heads for natural-language object detection and instance segmentation. A query such as `dog` or `the person on the left` can produce bounding boxes and, for segmentation tasks, pixel-level masks. The repository uses its PyTorch/CUDA path through the `falcon-perception[torch]` package.

### Strengths

- **Explicit spatial grounding:** returns boxes and masks rather than only a prose description.
- **Open-vocabulary queries:** the user can provide a natural-language object or expression instead of selecting from a fixed class list.
- **Deterministic counting support:** the application can count normalized detections directly instead of asking a language model to estimate a count.
- **Useful intermediate evidence:** detections can be rendered, cached, displayed, and passed to a later reasoning step.
- **Focused inference:** for a detection request, it performs a specific perception task rather than generating an unnecessarily long explanation.

### Limitations

- **GPU and runtime requirements:** the current PyTorch path requires CUDA, model downloads, and enough VRAM. Triton may also require Python development headers when compiling CUDA helpers.
- **Narrower conversational ability:** Falcon Perception should be treated as a grounding tool, not the system's primary conversational answer writer.
- **Per-query work:** different object expressions generally require separate inference calls. Comparisons can therefore multiply detection latency.
- **Resolution and mask costs:** higher-resolution visual features improve localization but increase memory use and preprocessing time.
- **Warm-up cost:** model loading and CUDA compilation can dominate the first request.

The upstream Falcon documentation reports optimized paged-engine timings on an H100 of roughly 100 ms for prefill, about 200 ms for uncached upsampling, and about 50 ms for short decode after an initial 10-30 second compile and CUDA-graph capture. Those figures are not a benchmark for VisionAI: this repository currently uses Falcon's batch inference path, and an RTX 3070, image size, query, cache state, and model initialization will produce different results.

## Gemma models for vision inference

[Gemma](https://ai.google.dev/gemma) is Google's family of open models. Vision-capable Gemma variants accept an image and text prompt and generate a natural-language response. In this repository, Gemma is served through the local [Ollama](https://ollama.com/) HTTP API; the default local tag is `gemma3:4b`, but another compatible local vision model can be selected.

Gemma is the reasoning layer in VisionAI. It is appropriate for questions such as:

- What is happening in the scene?
- What color or attribute does the largest detected object have?
- How should several grounded observations be explained to the user?
- What is a useful short description of the image?

### Strengths

- **Flexible language reasoning:** handles open-ended questions, descriptions, comparisons, and explanations better than a perception-only interface.
- **Multimodal context:** can inspect the original image and, in combined workflows, receive a compact summary of Falcon's grounded evidence.
- **Natural user experience:** produces answers instead of exposing only coordinates, masks, or model-specific metadata.
- **Local deployment:** Ollama keeps image data and inference on the local machine and makes model selection straightforward.
- **Model choice:** the Ollama endpoint allows the deployment to trade quality, memory, and latency by selecting a suitable local tag.

### Limitations

- **Counting and localization are not guaranteed:** a VLM can hallucinate objects, miss instances, or give approximate locations. Exact counts and coordinates should come from Falcon when available.
- **Autoregressive latency:** response time grows with prompt processing, image encoding, and generated output tokens. Long answers cost more than short answers.
- **Memory pressure:** a larger or higher-quality vision model can compete with Falcon for GPU memory. VisionAI intentionally tests and warms the models separately.
- **Cold starts:** Ollama may need to load the selected model before answering. The first request is usually much slower than a warm request.
- **Runtime dependency:** the Ollama service and requested model tag must be reachable and already available locally.

Gemma latency does not have one useful universal number. It depends on model size and quantization, image resolution, prompt length, generated token count, GPU or CPU placement, Ollama load state, and whether the model is already resident in memory. The application records Ollama duration and token counts so latency can be measured on the target device rather than inferred from a different benchmark.

## Why VisionAI uses both models

The combination follows a useful principle: use the smallest specialized operation that can produce trustworthy evidence, then use a language model to interpret that evidence.

1. **Falcon grounds the request.** It finds the requested instances and returns boxes, masks, scores, and count indexes.
2. **The application performs reliable operations.** Counts and count comparisons can be computed directly from detections, while annotations make the evidence visible.
3. **Gemma explains the result.** For descriptions and attribute questions, it can use the original image plus the Falcon summary to produce a readable answer.
4. **The planner avoids unnecessary work.** Direct scene questions can use Gemma only; counting and location questions can invoke Falcon first; compound questions can use both.

Using only Falcon would provide strong spatial evidence but a limited conversational interface. Using only Gemma would simplify the stack but make exact counts, masks, and coordinates less dependable. The two-model design makes those failure modes explicit and lets the UI show which model produced each part of the result.

## Latency and edge inference

Latency matters more at the edge because the device has limited compute, memory, power, and thermal headroom, and because there is no remote service to absorb model loading or queueing. A user experiences the total path, not just neural-network time:

```text
request wait
  + image read and preprocessing
  + model load or wake-up
  + vision prefill / feature extraction
  + detection or token generation
  + annotation rendering
  + HTTP and UI update
```

The two models have different latency profiles:

| Stage | Falcon Perception | Gemma vision model |
| --- | --- | --- |
| Main output | Boxes, masks, and detection metadata | Text tokens and explanations |
| Latency driver | Image resolution, segmentation work, CUDA warm-up, and number of queries | Model size, image encoding, prompt length, output tokens, and cold start |
| Typical workflow use | Short, structured, evidence-producing calls | One flexible response, sometimes after grounding |
| Edge risk | VRAM spikes and expensive first-run compilation | Larger memory footprint and variable generation time |
| Optimization levers | Cache detections, serialize GPU access, limit image size, reuse image features where supported | Use a quantized compact model, cap output length, keep the model warm, avoid needless calls |

For this repository, the practical latency strategy is:

- Keep Falcon detection results cached by image fingerprint, query, and annotation mode.
- Use deterministic answers for counts and comparisons instead of a second VLM generation.
- Send direct descriptive questions to Gemma without Falcon when spatial grounding is unnecessary.
- Use Falcon first for questions involving counts, locations, highlighting, or object attributes tied to a selected instance.
- Keep Falcon and Gemma from loading concurrently on small GPUs.
- Measure cold-start and warm-request latency separately, including end-to-end API time.
- Prefer concise prompts and bounded answers for interactive edge use.

The goal is not to minimize the latency of either model in isolation. It is to minimize total user-visible latency while preserving trustworthy grounding. A slower combined request is justified when it prevents a false count or makes an object-specific answer verifiable; a second model call is unnecessary when a direct Gemma response is sufficient.

## References

- [Falcon Perception repository](https://github.com/tiiuae/Falcon-Perception)
- [Falcon Perception technical report](https://arxiv.org/abs/2603.27365)
- [Google Gemma documentation](https://ai.google.dev/gemma)
- [Ollama documentation](https://docs.ollama.com/)
