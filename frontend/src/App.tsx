import { FormEvent, useEffect, useMemo, useState } from 'react';

const DEFAULT_QUESTION = 'How many dogs are in this image?';
const DEFAULT_OBJECT_QUERY = 'dog';
const MODES = ['mask', 'box', 'combined'] as const;

type AnnotationMode = (typeof MODES)[number];

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

type DetectResponse = {
  image_id: string;
  object_query: string;
  annotation_mode: AnnotationMode;
  annotated_image_url: string | null;
  detections: DetectionRecord[];
  message: string | null;
  timings: Record<string, number>;
};

type ChatResponse = {
  image_id: string;
  query: string;
  answer: string;
  route: string;
  annotated_image_url: string | null;
  detections: DetectionRecord[];
  message: string | null;
  timings: Record<string, number>;
};

type ChatTurn = {
  role: 'user' | 'assistant';
  text: string;
  route?: string;
};

type CatalogResponse = {
  images: ImageSummary[];
};

type ApiError = {
  detail?: string;
};

async function postJson<TResponse>(url: string, body: Record<string, unknown>): Promise<TResponse> {
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  const payload = (await response.json()) as TResponse & ApiError;
  if (!response.ok) {
    throw new Error(payload.detail || 'Request failed.');
  }
  return payload;
}

function App() {
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [catalogState, setCatalogState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [catalogError, setCatalogError] = useState('');
  const [selectedImageId, setSelectedImageId] = useState('');
  const [annotationMode, setAnnotationMode] = useState<AnnotationMode>('combined');
  const [objectQuery, setObjectQuery] = useState(DEFAULT_OBJECT_QUERY);
  const [question, setQuestion] = useState(DEFAULT_QUESTION);
  const [annotatedImageUrl, setAnnotatedImageUrl] = useState('');
  const [detectResult, setDetectResult] = useState<DetectResponse | null>(null);
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [runStatus, setRunStatus] = useState('Loading image catalog...');
  const [runError, setRunError] = useState('');
  const [isDetecting, setIsDetecting] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [lastDetectQuery, setLastDetectQuery] = useState('');

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

  const selectedImage = useMemo(
    () => images.find((image) => image.id === selectedImageId) || null,
    [images, selectedImageId]
  );

  const displayedImageUrl = annotatedImageUrl || selectedImage?.image_url || '';
  const canRun = Boolean(selectedImageId);

  async function runDetect(nextQuery = objectQuery, nextMode = annotationMode) {
    if (!canRun) {
      return;
    }
    setIsDetecting(true);
    setRunError('');
    setRunStatus(`Detecting ${nextQuery}...`);
    try {
      const payload = await postJson<DetectResponse>('/api/detect', {
        image_id: selectedImageId,
        object_query: nextQuery,
        annotation_mode: nextMode,
      });
      setDetectResult(payload);
      setAnnotatedImageUrl(payload.annotated_image_url || '');
      setLastDetectQuery(nextQuery);
      setRunStatus(payload.message || `Detected ${payload.detections.length} match(es).`);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Detection failed.');
      setRunStatus('Detection failed.');
    } finally {
      setIsDetecting(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canRun) {
      return;
    }
    setIsChatting(true);
    setRunError('');
    setRunStatus('Running grounded chat...');
    try {
      const payload = await postJson<ChatResponse>('/api/chat', {
        image_id: selectedImageId,
        query: question,
        annotation_mode: annotationMode,
      });
      setAnnotatedImageUrl(payload.annotated_image_url || '');
      setDetectResult(
        payload.detections.length
          ? {
              image_id: payload.image_id,
              object_query: lastDetectQuery || objectQuery,
              annotation_mode: annotationMode,
              annotated_image_url: payload.annotated_image_url,
              detections: payload.detections,
              message: payload.message,
              timings: payload.timings,
            }
          : null
      );
      setChatTurns((turns) => [
        ...turns,
        { role: 'user', text: question },
        { role: 'assistant', text: payload.answer, route: payload.route },
      ]);
      setRunStatus(payload.message || `Completed via ${payload.route}.`);
    } catch (error) {
      setRunError(error instanceof Error ? error.message : 'Chat request failed.');
      setRunStatus('Chat request failed.');
    } finally {
      setIsChatting(false);
    }
  }

  function handleImageSelect(imageId: string) {
    setSelectedImageId(imageId);
    setAnnotatedImageUrl('');
    setDetectResult(null);
    setRunError('');
    setRunStatus('Ready.');
  }

  function handleModeChange(nextMode: AnnotationMode) {
    setAnnotationMode(nextMode);
    if (detectResult && lastDetectQuery) {
      void runDetect(lastDetectQuery, nextMode);
    }
  }

  return (
    <div className="mx-auto max-w-[1440px] px-5 pb-7 pt-8 text-slate-100">
      <header className="mb-6 flex flex-col items-start justify-between gap-4 lg:flex-row">
        <div>
          <p className="mb-1 text-[0.72rem] uppercase tracking-[0.12em] text-vision-gold">VisionAI</p>
          <h1 className="mb-2 text-[clamp(2rem,3vw,3rem)] font-semibold">Grounded Image Chat</h1>
          <p className="max-w-3xl text-sm text-slate-200/75">
            Phase 1-3 prototype with local image catalog, Falcon detection, and Ollama-backed chat.
          </p>
        </div>
        <span className="pill max-w-[360px] border-vision-gold/35 bg-vision-gold/10 text-[#fff1dc]">{runStatus}</span>
      </header>

      <main className="grid items-start gap-6 xl:grid-cols-[minmax(260px,320px)_minmax(0,1.25fr)_minmax(340px,420px)]">
        <section className="panel">
          <div className="mb-4 flex items-start justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold">Gallery</h2>
              <p className="text-sm text-slate-200/75">Images are served by the local catalog API from the repo image folder.</p>
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
          <div className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-xl font-semibold">{selectedImage?.name || 'Viewer'}</h2>
              <p className="text-sm text-slate-200/75">
                {selectedImage ? `${selectedImage.width} × ${selectedImage.height}` : 'Select an image to begin.'}
              </p>
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
                  onClick={() => handleModeChange(mode)}
                >
                  {mode}
                </button>
              ))}
            </div>
          </div>

          <div className="relative overflow-hidden rounded-[18px] border border-white/12 bg-[rgba(18,31,48,0.9)] aspect-[16/10]">
            {displayedImageUrl ? (
              <img className="block h-full w-full object-cover" src={displayedImageUrl} alt={selectedImage?.name || 'Selected scene'} />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-slate-300/60">Choose an image to view it here.</div>
            )}
          </div>

          <div className="mt-5">
            <form
              className="flex flex-col gap-3"
              onSubmit={(event) => {
                event.preventDefault();
                void runDetect();
              }}
            >
              <label className="font-semibold text-[#f6ead4]" htmlFor="objectQuery">
                Detect objects
              </label>
              <input
                id="objectQuery"
                type="text"
                value={objectQuery}
                onChange={(event) => setObjectQuery(event.target.value)}
                placeholder="dog"
                className="w-full rounded-xl border border-white/12 bg-[rgba(5,13,20,0.84)] px-3.5 py-3 text-slate-100 outline-none placeholder:text-slate-400"
              />
              <button
                type="submit"
                className="rounded-xl bg-gradient-to-br from-vision-gold to-vision-coral px-4 py-3 font-bold text-vision-ink transition hover:-translate-y-[1px] hover:shadow-[0_12px_22px_rgba(255,146,84,0.26)] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0 disabled:hover:shadow-none"
                disabled={!canRun || isDetecting || !objectQuery.trim()}
              >
                {isDetecting ? 'Detecting...' : 'Run detection'}
              </button>
            </form>
          </div>

          <div className="mt-[18px] rounded-[14px] border border-white/8 bg-white/[0.04] p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="text-lg font-semibold">Detections</h3>
              {detectResult?.detections.length ? (
                <span className="pill whitespace-nowrap border-vision-aqua/30 bg-vision-aqua/15 text-[#e7f3ff]">
                  {detectResult.detections.length} found
                </span>
              ) : null}
            </div>
            {detectResult?.detections.length ? (
              <ul className="flex list-none flex-col gap-2.5 p-0 m-0">
                {detectResult.detections.map((detection) => (
                  <li key={`${detection.label}-${detection.count_index}`} className="flex flex-col gap-0.5 rounded-xl bg-white/[0.03] px-3 py-2.5">
                    <strong>
                      {detection.count_index}. {detection.label}
                    </strong>
                    <span>{detection.score ? `score ${detection.score.toFixed(3)}` : 'score unavailable'}</span>
                    <span>
                      {detection.bbox ? `box ${detection.bbox.map((value) => value.toFixed(0)).join(', ')}` : 'box unavailable'}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-slate-300/70">Detection results will appear here.</p>
            )}
          </div>
        </section>

        <section className="panel flex min-h-[72vh] flex-col">
          <div className="mb-4">
            <h2 className="text-xl font-semibold">Grounded chat</h2>
            <p className="text-sm text-slate-200/75">
              Counts are answered deterministically after detection; general scene questions go through Ollama.
            </p>
          </div>

          <div className="flex min-h-[280px] flex-col gap-3">
            {!chatTurns.length ? (
              <p className="text-slate-300/70">Ask a question about the selected image to start the session.</p>
            ) : (
              chatTurns.map((turn, index) => (
                <article
                  key={`${turn.role}-${index}`}
                  className={[
                    'max-w-full rounded-2xl border px-4 py-3.5',
                    turn.role === 'user'
                      ? 'border-vision-aqua/20 bg-vision-aqua/15'
                      : 'border-white/8 bg-white/[0.05]',
                  ].join(' ')}
                >
                  <span className="mb-1.5 block text-[0.76rem] uppercase tracking-[0.08em] text-slate-200/70">
                    {turn.role === 'user' ? 'You' : turn.route || 'Assistant'}
                  </span>
                  <p className="m-0">{turn.text}</p>
                </article>
              ))
            )}
          </div>

          {runError ? <p className="mt-4 rounded-xl border border-red-400/35 bg-red-400/10 px-4 py-3 text-red-100">{runError}</p> : null}

          <form onSubmit={handleSubmit} className="mt-auto flex flex-col gap-3 pt-4">
            <label className="sr-only" htmlFor="question">
              Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={3}
              placeholder="Ask a question about the selected image..."
              className="min-h-[110px] w-full resize-y rounded-xl border border-white/12 bg-[rgba(5,13,20,0.84)] px-3.5 py-3 text-slate-100 outline-none placeholder:text-slate-400"
            />
            <button
              type="submit"
              className="rounded-xl bg-gradient-to-br from-vision-gold to-vision-coral px-4 py-3 font-bold text-vision-ink transition hover:-translate-y-[1px] hover:shadow-[0_12px_22px_rgba(255,146,84,0.26)] disabled:cursor-not-allowed disabled:opacity-55 disabled:hover:translate-y-0 disabled:hover:shadow-none"
              disabled={!canRun || isChatting || !question.trim()}
            >
              {isChatting ? 'Submitting...' : 'Submit'}
            </button>
          </form>
        </section>
      </main>
    </div>
  );
}

export default App;