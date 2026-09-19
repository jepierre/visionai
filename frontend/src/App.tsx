import { FormEvent, useEffect, useMemo, useState } from 'react';

const DEFAULT_QUESTION = 'How many dogs are in this image?';
const MODES = ['mask', 'box', 'combined'] as const;
const EXECUTION_MODES = ['falcon', 'gemma', 'agent'] as const;

type AnnotationMode = (typeof MODES)[number];
type ExecutionMode = (typeof EXECUTION_MODES)[number];

type ImageSummary = {
  id: string;
  name: string;
  width: number;
  height: number;
  image_url: string;
  thumbnail_url: string;
};

type DetectionRecord = {
  label: string;
  score: number | null;
  bbox: number[] | null;
  mask_area: number | null;
  count_index: number;
};

type ModelInfo = {
  name: string;
  role: string;
};

type TraceStep = {
  title: string;
  detail: string;
  model: string | null;
  action?: string | null;
};

type DetectResponse = {
  image_id: string;
  object_query: string;
  annotation_mode: AnnotationMode;
  original_image_url: string;
  annotated_image_url: string | null;
  detections: DetectionRecord[];
  execution_mode: 'falcon';
  models_used: ModelInfo[];
  trace: TraceStep[];
  reasoning: string | null;
  final_output: string | null;
  message: string | null;
  timings: Record<string, number>;
};

type ChatResponse = {
  image_id: string;
  query: string;
  answer: string;
  route: string;
  execution_mode: 'agent' | 'gemma';
  original_image_url: string;
  annotated_image_url: string | null;
  detections: DetectionRecord[];
  models_used: ModelInfo[];
  trace: TraceStep[];
  reasoning: string | null;
  final_output: string | null;
  message: string | null;
  timings: Record<string, number>;
};

type CatalogResponse = {
  images: ImageSummary[];
};

type OllamaModelSummary = {
  name: string;
};

type OllamaModelListResponse = {
  models: OllamaModelSummary[];
  default_model: string;
};

type ApiError = {
  detail?: string;
};

type RunResult = {
  executionMode: ExecutionMode;
  routeLabel: string;
  originalImageUrl: string;
  annotatedImageUrl: string | null;
  modelsUsed: ModelInfo[];
  trace: TraceStep[];
  reasoning: string;
  finalOutput: string;
  detections: DetectionRecord[];
  timings: Record<string, number>;
  message: string | null;
};

async function postJson<TResponse>(url: string, body: Record<string, unknown>): Promise<TResponse> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const responseText = await response.text();
  let payload: (TResponse & ApiError) | null = null;
  try {
    payload = responseText ? (JSON.parse(responseText) as TResponse & ApiError) : null;
  } catch {
    payload = null;
  }
  if (!response.ok) {
    throw new Error(payload?.detail || responseText || `Request failed with status ${response.status}.`);
  }
  if (!payload) {
    throw new Error('The server returned an empty or invalid JSON response.');
  }
  return payload;
}

function formatTimingLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/seconds/g, 's');
}

function App() {
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState('');
  const [catalogState, setCatalogState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [catalogError, setCatalogError] = useState('');
  const [ollamaModelError, setOllamaModelError] = useState('');
  const [selectedImageId, setSelectedImageId] = useState('');
  const [annotationMode, setAnnotationMode] = useState<AnnotationMode>('combined');
  const [executionMode, setExecutionMode] = useState<ExecutionMode>('agent');
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [runResult, setRunResult] = useState<RunResult | null>(null);
  const [runStatus, setRunStatus] = useState('Loading image catalog...');
  const [runError, setRunError] = useState('');
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    let isActive = true;

    async function loadImages() {
      try {
        const response = await fetch('/api/images');
        const payload = (await response.json()) as CatalogResponse & ApiError;
        if (!response.ok) {
          throw new Error(payload.detail || 'Unable to load images.');
        }
        if (!isActive) {
          return;
        }
        setImages(payload.images);
        setSelectedImageId((current) => current || payload.images[0]?.id || '');
        setCatalogState('ready');
        setRunStatus(payload.images.length ? 'Ready.' : 'No images found in the images folder.');
      } catch (error) {
        if (!isActive) {
          return;
        }
        setCatalogState('error');
        setCatalogError(error instanceof Error ? error.message : 'Unable to load images.');
        setRunStatus('Image catalog unavailable.');
      }
    }

    void loadImages();
    return () => {
      isActive = false;
    };
  }, []);

  useEffect(() => {
    let isActive = true;

    async function loadOllamaModels() {
      try {
        const response = await fetch('/api/ollama/models');
        const payload = (await response.json()) as OllamaModelListResponse & ApiError;
        if (!response.ok) {
          throw new Error(payload.detail || 'Unable to load Ollama models.');
        }
        if (!isActive) {
          return;
        }
        const modelNames = payload.models.map((model) => model.name);
        setOllamaModels(modelNames);
        setSelectedOllamaModel((current) => {
          if (current && modelNames.includes(current)) {
            return current;
          }
          if (payload.default_model && modelNames.includes(payload.default_model)) {
            return payload.default_model;
          }
          return modelNames[0] || payload.default_model || '';
        });
        setOllamaModelError('');
      } catch (error) {
        if (!isActive) {
          return;
        }
        setOllamaModels([]);
        setSelectedOllamaModel('');
        setOllamaModelError(error instanceof Error ? error.message : 'Unable to load Ollama models.');
      }
    }

    void loadOllamaModels();
    return () => {
      isActive = false;
    };
  }, []);

  const selectedImage = useMemo(
    () => images.find((image) => image.id === selectedImageId) || null,
    [images, selectedImageId]
  );

  const canRun = Boolean(selectedImageId);
  const routeSummary = useMemo(() => {
    if (!runResult) {
      return 'No workflow has been run yet.';
    }
    return `${runResult.executionMode.toUpperCase()} mode via ${runResult.routeLabel}`;
  }, [runResult]);

  async function handleRun(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canRun) {
      return;
    }

    setIsRunning(true);
    setRunError('');
    setRunStatus(
      executionMode === 'falcon'
        ? `Running Falcon for ${question}...`
        : executionMode === 'gemma'
          ? 'Running Gemma-only reasoning...'
          : 'Running combined Falcon + Gemma workflow...'
    );

    try {
      if (executionMode === 'falcon') {
        const payload = await postJson<DetectResponse>('/api/detect', {
          image_id: selectedImageId,
          object_query: question.trim(),
          annotation_mode: annotationMode,
        });
        setRunResult({
          executionMode: 'falcon',
          routeLabel: payload.execution_mode,
          originalImageUrl: payload.original_image_url,
          annotatedImageUrl: payload.annotated_image_url,
          modelsUsed: payload.models_used,
          trace: payload.trace,
          reasoning: payload.reasoning || 'Falcon-only run completed.',
          finalOutput: payload.final_output || 'Falcon run completed.',
          detections: payload.detections,
          timings: payload.timings,
          message: payload.message,
        });
        setRunStatus(payload.message || payload.final_output || `Detected ${payload.detections.length} match(es).`);
      } else {
        const payload = await postJson<ChatResponse>('/api/chat', {
          image_id: selectedImageId,
          query: question,
          annotation_mode: annotationMode,
          execution_mode: executionMode,
          ollama_model: selectedOllamaModel || null,
        });
        setRunResult({
          executionMode,
          routeLabel: payload.route,
          originalImageUrl: payload.original_image_url,
          annotatedImageUrl: payload.annotated_image_url,
          modelsUsed: payload.models_used,
          trace: payload.trace,
          reasoning: payload.reasoning || 'Run completed.',
          finalOutput: payload.final_output || payload.answer,
          detections: payload.detections,
          timings: payload.timings,
          message: payload.message,
        });
        setRunStatus(payload.message || payload.final_output || `Completed via ${payload.route}.`);
      }
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Run failed.');
      setRunStatus('Run failed.');
    } finally {
      setIsRunning(false);
    }
  }

  function handleImageSelect(imageId: string) {
    setSelectedImageId(imageId);
    setRunResult(null);
    setRunError('');
    setRunStatus('Ready.');
  }

  function handleExecutionModeChange(nextMode: ExecutionMode) {
    setExecutionMode(nextMode);
    if (nextMode === 'falcon' && question === DEFAULT_QUESTION) {
      setQuestion('dog');
    } else if (nextMode !== 'falcon' && question === 'dog') {
      setQuestion(DEFAULT_QUESTION);
    }
  }

  function renderImagePane(title: string, subtitle: string, imageUrl: string | null, fallback: string) {
    return (
      <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
        <div className="mb-3">
          <h3 className="text-base font-semibold">{title}</h3>
          <p className="text-sm text-slate-300/70">{subtitle}</p>
        </div>
        <div className="relative aspect-[16/10] overflow-hidden rounded-[18px] border border-white/12 bg-[rgba(18,31,48,0.9)]">
          {imageUrl ? (
            <img className="block h-full w-full object-cover" src={imageUrl} alt={title} />
          ) : (
            <div className="flex h-full w-full items-center justify-center px-6 text-center text-slate-300/60">{fallback}</div>
          )}
        </div>
      </article>
    );
  }

  function downloadSummary() {
    if (!runResult) {
      return;
    }
    const blob = new Blob([JSON.stringify(runResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `visionai-${selectedImage?.name || 'run'}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="mx-auto max-w-[1580px] px-5 pb-7 pt-8 text-slate-100">
      <header className="mb-6 flex flex-col items-start justify-between gap-4 lg:flex-row">
        <div>
          <p className="mb-1 text-[0.72rem] uppercase tracking-[0.12em] text-vision-gold">VisionAI</p>
          <h1 className="mb-2 text-[clamp(2rem,3vw,3rem)] font-semibold">Agentic Grounded Image Chat</h1>
          <p className="max-w-3xl text-sm text-slate-200/75">
            Inspect Falcon-only, Gemma-only, and combined Falcon-plus-Gemma runs with model badges, workflow trace,
            reasoning, and final output.
          </p>
        </div>
        <span className="pill max-w-[380px] border-vision-gold/35 bg-vision-gold/10 text-[#fff1dc]">{runStatus}</span>
      </header>

      <main className="grid items-start gap-6 2xl:grid-cols-[minmax(260px,320px)_minmax(0,1.45fr)_minmax(340px,420px)]">
        <section className="panel">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">Gallery</h2>
              <p className="text-sm text-slate-200/75">Choose an image from the backend catalog.</p>
            </div>
            <span className="pill whitespace-nowrap border-vision-aqua/30 bg-vision-aqua/15 text-[#e7f3ff]">{images.length} image(s)</span>
          </div>

          {catalogState === 'loading' ? <p className="text-slate-300/70">Loading images...</p> : null}
          {catalogState === 'error' ? (
            <p className="rounded-xl border border-red-400/35 bg-red-400/10 px-4 py-3 text-red-100">{catalogError}</p>
          ) : null}
          {catalogState === 'ready' && !images.length ? (
            <p className="text-slate-300/70">No supported images were found in the images folder.</p>
          ) : null}

          <div className="flex flex-col gap-3">
            {images.map((image) => {
              const isSelected = selectedImageId === image.id;
              return (
                <button
                  key={image.id}
                  type="button"
                  className={[
                    'overflow-hidden rounded-xl border text-left transition duration-150',
                    isSelected
                      ? 'border-vision-gold/90 bg-vision-gold/10 -translate-y-[1px]'
                      : 'border-white/10 bg-white/[0.03] hover:-translate-y-[1px] hover:border-vision-gold/90 hover:bg-vision-gold/10',
                  ].join(' ')}
                  onClick={() => handleImageSelect(image.id)}
                >
                  <img className="block aspect-[16/10] w-full object-cover" src={image.thumbnail_url} alt={image.name} />
                  <span className="block px-3 py-3 text-[0.92rem] text-slate-100">{image.name}</span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="panel">
          <div className="mb-5 flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
            <div>
              <h2 className="text-xl font-semibold">Execution workspace</h2>
              <p className="text-sm text-slate-200/75">Run Falcon, Gemma, or the full agent workflow against the selected image.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              {MODES.map((mode) => (
                <button
                  key={mode}
                  type="button"
                  className={[
                    'cursor-pointer rounded-full border px-3 py-2 text-sm capitalize',
                    annotationMode === mode
                      ? 'border-vision-gold/45 bg-vision-gold/15 text-slate-50'
                      : 'border-white/12 bg-white/[0.02] text-slate-100',
                  ].join(' ')}
                  onClick={() => setAnnotationMode(mode)}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          <form onSubmit={handleRun} className="mb-6 grid gap-4 rounded-[20px] border border-white/10 bg-white/[0.03] p-4 lg:grid-cols-2">
            <div className="lg:col-span-2">
              <p className="mb-2 text-sm font-semibold text-[#f6ead4]">Execution mode</p>
              <div className="flex flex-wrap gap-2">
                {EXECUTION_MODES.map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    className={[
                      'rounded-full border px-4 py-2 text-sm capitalize transition',
                      executionMode === mode
                        ? 'border-vision-gold/45 bg-vision-gold/15 text-slate-50'
                        : 'border-white/12 bg-white/[0.02] text-slate-100',
                    ].join(' ')}
                    onClick={() => handleExecutionModeChange(mode)}
                  >
                    {mode === 'agent' ? 'Gemma + Falcon' : `${mode} only`}
                  </button>
                ))}
              </div>
            </div>

            <label className="flex flex-col gap-2 lg:col-span-2">
              <span className="font-semibold text-[#f6ead4]">
                {executionMode === 'falcon' ? 'Object to detect' : 'Question for the selected workflow'}
              </span>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                rows={3}
                placeholder={executionMode === 'falcon' ? 'dog' : 'Ask a question about the selected image...'}
                className="min-h-[110px] w-full resize-y rounded-xl border border-white/12 bg-[rgba(5,13,20,0.84)] px-3.5 py-3 text-slate-100 outline-none placeholder:text-slate-400"
              />
              <span className="text-xs text-slate-300/60">
                {executionMode === 'falcon'
                  ? 'Falcon-only mode treats this prompt as the object query.'
                  : 'Combined mode lets the planner decide whether Falcon is needed.'}
              </span>
            </label>

            <label className="flex flex-col gap-2 lg:col-span-2">
              <span className="font-semibold text-[#f6ead4]">Ollama model</span>
              <select
                value={selectedOllamaModel}
                onChange={(event) => setSelectedOllamaModel(event.target.value)}
                disabled={executionMode === 'falcon' || !ollamaModels.length}
                className="w-full rounded-xl border border-white/12 bg-[rgba(5,13,20,0.84)] px-3.5 py-3 text-slate-100 outline-none disabled:cursor-not-allowed disabled:opacity-60"
              >
                {ollamaModels.length ? (
                  ollamaModels.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))
                ) : (
                  <option value="">No Ollama models available</option>
                )}
              </select>
              <span className="text-xs text-slate-300/60">
                {executionMode === 'falcon'
                  ? 'Falcon-only mode does not call Ollama.'
                  : 'Gemma-only and combined mode use the selected Ollama model.'}
              </span>
              {ollamaModelError ? <span className="text-xs text-red-200">{ollamaModelError}</span> : null}
            </label>

            <div className="flex flex-col gap-3 lg:col-span-2 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-sm text-slate-300/70">{routeSummary}</p>
              <button
                type="submit"
                className="rounded-xl bg-gradient-to-br from-vision-gold to-vision-coral px-4 py-3 font-bold text-vision-ink transition hover:-translate-y-[1px] hover:shadow-[0_12px_22px_rgba(255,146,84,0.26)] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0 disabled:hover:shadow-none"
                disabled={!canRun || isRunning || !question.trim()}
              >
                {isRunning ? 'Running...' : 'Run workflow'}
              </button>
            </div>
          </form>

          <div>
            {renderImagePane(
              'Falcon annotated output',
              'Masks and or boxes appear here when Falcon participates in the workflow.',
              runResult?.annotatedImageUrl || null,
              'No Falcon overlay is available for this run yet.'
            )}
            {runResult?.annotatedImageUrl ? (
              <div className="mt-3 flex flex-wrap gap-2">
                <a
                  href={runResult.annotatedImageUrl}
                  download
                  className="rounded-xl border border-vision-aqua/30 bg-vision-aqua/15 px-3 py-2 text-sm text-[#e7f3ff]"
                >
                  Download annotated image
                </a>
                <button
                  type="button"
                  onClick={downloadSummary}
                  className="rounded-xl border border-vision-gold/35 bg-vision-gold/10 px-3 py-2 text-sm text-[#fff1dc]"
                >
                  Download run summary
                </button>
              </div>
            ) : null}
          </div>
        </section>

        <section className="panel flex min-h-[72vh] flex-col gap-4">
          <div>
            <h2 className="text-xl font-semibold">Agent run details</h2>
            <p className="text-sm text-slate-200/75">Inspect models used, intermediate steps, reasoning, final output, and grounded detections.</p>
          </div>

          <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-3 text-base font-semibold">Models used</h3>
            {runResult?.modelsUsed.length ? (
              <div className="flex flex-col gap-2">
                {runResult.modelsUsed.map((model) => (
                  <div key={`${model.name}-${model.role}`} className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2.5">
                    <p className="font-medium text-slate-50">{model.name}</p>
                    <p className="text-sm text-slate-300/70">{model.role}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-slate-300/70">Run a workflow to see which models were invoked.</p>
            )}
          </article>

          <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-3 text-base font-semibold">Intermediate loop</h3>
            {runResult?.trace.length ? (
              <ol className="m-0 flex list-decimal flex-col gap-3 pl-5">
                {runResult.trace.map((step, index) => (
                  <li key={`${step.title}-${index}`}>
                    <div className="rounded-xl border border-white/10 bg-white/[0.02] px-3 py-2.5">
                      <p className="font-medium text-slate-50">{step.title}</p>
                      <p className="text-sm text-slate-300/75">{step.detail}</p>
                      {step.action ? <p className="mt-1 text-xs uppercase tracking-[0.08em] text-vision-aqua/90">Action: {step.action}</p> : null}
                      {step.model ? <p className="mt-1 text-xs uppercase tracking-[0.08em] text-vision-gold/90">{step.model}</p> : null}
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-slate-300/70">No execution trace yet.</p>
            )}
          </article>

          <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-3 text-base font-semibold">Reasoning</h3>
            <p className="text-sm leading-6 text-slate-200/85">
              {runResult?.reasoning || 'The agent reasoning summary will appear here after a run.'}
            </p>
          </article>

          <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-3 text-base font-semibold">Final output</h3>
            <p className="text-sm leading-6 text-slate-100">{runResult?.finalOutput || 'The final model output will appear here.'}</p>
            {runResult?.message ? <p className="mt-3 text-xs text-slate-300/70">{runResult.message}</p> : null}
          </article>

          <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-3 text-base font-semibold">Grounded detections</h3>
            {runResult?.detections.length ? (
              <ul className="m-0 flex list-none flex-col gap-2 p-0">
                {runResult.detections.map((detection) => (
                  <li key={`${detection.label}-${detection.count_index}`} className="rounded-xl bg-white/[0.03] px-3 py-2.5">
                    <p className="font-medium text-slate-50">
                      {detection.count_index}. {detection.label}
                    </p>
                    <p className="text-sm text-slate-300/75">
                      {detection.score !== null ? `score ${detection.score.toFixed(3)}` : 'score unavailable'}
                    </p>
                    <p className="text-sm text-slate-300/75">
                      {detection.bbox ? `box ${detection.bbox.map((value) => value.toFixed(0)).join(', ')}` : 'box unavailable'}
                    </p>
                    <p className="text-sm text-slate-300/75">
                      {detection.mask_area !== null ? `mask area ${detection.mask_area}` : 'mask unavailable'}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-300/70">No grounded detections are available for this run.</p>
            )}
          </article>

          <article className="rounded-[20px] border border-white/10 bg-white/[0.03] p-4">
            <h3 className="mb-3 text-base font-semibold">Timings</h3>
            {runResult && Object.keys(runResult.timings).length ? (
              <div className="flex flex-wrap gap-2">
                {Object.entries(runResult.timings).map(([key, value]) => (
                  <span key={key} className="pill border-vision-aqua/30 bg-vision-aqua/15 text-[#e7f3ff]">
                    {formatTimingLabel(key)}: {value.toFixed(2)}s
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-slate-300/70">Timing data will appear after a run.</p>
            )}
          </article>

          {runError ? <p className="rounded-xl border border-red-400/35 bg-red-400/10 px-4 py-3 text-red-100">{runError}</p> : null}
        </section>
      </main>
    </div>
  );
}

export default App;