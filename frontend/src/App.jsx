import { useMemo, useState } from 'react';

const SAMPLE_IMAGES = [
  'https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=1200&q=80',
  'https://images.unsplash.com/photo-1493246507139-91e8fad9978e?auto=format&fit=crop&w=1200&q=80',
  'https://images.unsplash.com/photo-1501594907352-04cda38ebc29?auto=format&fit=crop&w=1200&q=80',
];

function App() {
  const [selectedImage, setSelectedImage] = useState(SAMPLE_IMAGES[0]);
  const [question, setQuestion] = useState('How many cars are in this image?');
  const [response, setResponse] = useState('The backend is not ready yet. Please check back once the local Python inference pipeline is connected.');

  const statusText = useMemo(
    () => 'Backend not connected — this interface is a placeholder until the Phase 1+ service is implemented.',
    []
  );

  const handleSubmit = (event) => {
    event.preventDefault();
    setResponse(
      'This UI is ready, but the backend is not available yet. Once the local FastAPI service and model pipeline are implemented, responses will be shown here.'
    );
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">VisionAI</p>
          <h1>Grounded Image Chat</h1>
        </div>
        <span className="status-pill">{statusText}</span>
      </header>

      <main className="content-grid">
        <section className="panel image-panel">
          <div className="panel-header">
            <h2>Image</h2>
          </div>

          <div className="image-stage">
            <img src={selectedImage} alt="Selected scene" />
          </div>

          <div className="thumbnail-row">
            {SAMPLE_IMAGES.map((imageUrl) => (
              <button
                key={imageUrl}
                type="button"
                className={selectedImage === imageUrl ? 'thumb active' : 'thumb'}
                onClick={() => setSelectedImage(imageUrl)}
              >
                <img src={imageUrl} alt="Select scene thumbnail" />
              </button>
            ))}
          </div>
        </section>

        <section className="panel chat-panel">
          <div className="panel-header">
            <h2>Ask about the image</h2>
          </div>

          <form onSubmit={handleSubmit} className="query-form">
            <label className="field-label" htmlFor="question">
              Question
            </label>
            <textarea
              id="question"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows="4"
            />

            <button type="submit" className="submit-button">
              Submit
            </button>
          </form>

          <div className="response-box">
            <h3>Response</h3>
            <p>{response}</p>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
