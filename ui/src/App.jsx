import { useState, useRef, useEffect, useCallback } from 'react';
import './index.css';
import { predict, health, classColor, ApiError } from './lib/api.js';
import { createSyntheticFrame, canvasToBlob, CLASS_NAMES } from './lib/synthetic.js';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  
  // Surfaced in the UI instead of an alert(), and cleared on the next success.
  const [error, setError] = useState(null);
  // What the server is actually running, so a stale or missing checkpoint is
  // visible before you send a frame.
  const [serverInfo, setServerInfo] = useState(null);

  // Webcam & Stream references
  const videoRef = useRef(null);
  const streamCanvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const streamIntervalRef = useRef(null);
  // Cancels the previous request when a newer frame arrives.
  const abortRef = useRef(null);
  // True while a stream request is in flight, so slow frames get dropped
  // rather than queued.
  const inFlightRef = useRef(false);

  useEffect(() => {
    // Best-effort: a failed probe is shown, not thrown. Aborted on unmount so
    // a slow probe cannot setState on a component that is already gone.
    const controller = new AbortController();
    health({ signal: controller.signal })
      .then(setServerInfo)
      .catch((err) => {
        if (err.name === 'AbortError') return;
        setError(err instanceof ApiError ? err.message : 'Backend unreachable');
      });
    return () => controller.abort();
  }, []);

  const generateDemoImage = () => {
    if (isStreaming || isWebcamActive) stopAllStreams();
    const canvas = document.createElement('canvas');
    createSyntheticFrame(canvas);
    
    canvasToBlob(canvas).then((blob) => {
      if (!blob) {
        setError('Could not encode the demo frame.');
        return;
      }
      const file = new File([blob], 'demo.png', { type: 'image/png' });
      handleFileChange({ target: { files: [file] } });
    });
  };

  const handleFileChange = (e) => {
    if (isStreaming || isWebcamActive) stopAllStreams();
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      // Release the previous blob URL, otherwise each upload leaks one.
      setPreview((prev) => {
        if (prev && prev.startsWith('blob:')) URL.revokeObjectURL(prev);
        return URL.createObjectURL(file);
      });
      setResults(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    try {
      setResults(await predict(selectedFile));
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Send one stream frame. Shared by the synthetic and webcam modes.
   *
   * Returns false when the tick was dropped or the frame could not be encoded,
   * so the caller never leaves the in-flight guard stuck on.
   */
  const sendStreamFrame = useCallback(async (canvas) => {
    const blob = await canvasToBlob(canvas);
    if (!blob) {
      setError('Could not encode the capture frame.');
      return false;
    }

    // A newer frame supersedes whatever is still pending.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const file = new File([blob], 'frame.png', { type: 'image/png' });
      setResults(await predict(file, { signal: controller.signal }));
      setError(null);
      return true;
    } catch (err) {
      // Aborts are expected here — they mean a newer frame arrived.
      if (err.name !== 'AbortError') setError(err.message);
      return false;
    }
  }, []);

  // Live Stream Logic
  const toggleSyntheticStream = () => {
    if (isStreaming) {
      stopAllStreams();
    } else {
      stopAllStreams();
      setIsStreaming(true);
      setError(null);
      streamIntervalRef.current = setInterval(async () => {
        // Drop the tick instead of queueing it. Without this guard a slow
        // backend makes requests pile up and they resolve out of order, so a
        // stale frame can overwrite a newer one on screen.
        if (inFlightRef.current) return;
        inFlightRef.current = true;
        try {
          const canvas = document.createElement('canvas');
          createSyntheticFrame(canvas);
          setPreview(canvas.toDataURL('image/png'));
          await sendStreamFrame(canvas);
        } finally {
          inFlightRef.current = false;
        }
      }, 500); // 2 FPS
    }
  };

  // Webcam Logic
  const toggleWebcam = async () => {
    if (isWebcamActive) {
      stopAllStreams();
      return;
    }
    stopAllStreams();

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Webcam capture needs a secure context (https or localhost).');
      return;
    }

    try {
      // Width/height here are ideals, not guarantees — real cameras return
      // their native resolution, so the capture canvas follows whatever the
      // track actually delivers instead of assuming 128x128.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
      });
      if (!videoRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      videoRef.current.srcObject = stream;
      await videoRef.current.play().catch(() => undefined);
      setIsWebcamActive(true);
      setError(null);

      streamIntervalRef.current = setInterval(async () => {
        const video = videoRef.current;
        const canvas = streamCanvasRef.current;
        if (!video || !canvas || video.readyState < 2 || inFlightRef.current) return;

        // Keep the sensor's aspect ratio; the backend letterboxes to square.
        const scale = Math.min(1, 512 / Math.max(video.videoWidth, 1));
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);

        setPreview(canvas.toDataURL('image/png'));
        inFlightRef.current = true;
        try {
          await sendStreamFrame(canvas);
        } finally {
          inFlightRef.current = false;
        }
      }, 1000); // 1 FPS
    } catch (err) {
      console.error('Webcam error:', err);
      setError(
        err?.name === 'NotAllowedError'
          ? 'Camera permission denied.'
          : 'Camera unavailable on this device.'
      );
    }
  };

  const stopAllStreams = () => {
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    streamIntervalRef.current = null;
    abortRef.current?.abort();
    abortRef.current = null;
    inFlightRef.current = false;
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      // Leave the camera light off in the UI, not just stopped underneath.
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    setIsWebcamActive(false);
  };

  useEffect(() => {
    return () => stopAllStreams(); // Cleanup on unmount
  }, []);

  // Boxes the API returned, filtered by the confidence slider. The backend
  // already normalises them to the ORIGINAL image, so they line up with the
  // preview regardless of its aspect ratio.
  const detections = results?.detections ?? [];
  const visibleDetections = detections.filter((d) => d.score >= confidenceThreshold);

  return (
    <div className="app-container">
      <header className="header">
        <h1>Per-Camera PoC Intelligence</h1>
        <p>Real-time autonomous perception command center.</p>
        {/* Shows which checkpoint is actually loaded. Without this, a stale or
            randomly-initialised model is indistinguishable from a good one. */}
        <div className="server-badge" data-testid="server-badge">
          {serverInfo
            ? `${serverInfo.arch_version ?? 'unknown'} · epoch ${serverInfo.checkpoint_epoch ?? '?'} · ${serverInfo.device ?? '?'}`
            : 'connecting to backend…'}
        </div>
      </header>

      <main className="main-content">
        <div className="dashboard-grid">
          {/* Controls */}
          <div className="control-panel glass-panel">
            <h2>Input Feed</h2>
            <div className="upload-section">
              <input type="file" id="file-upload" accept="image/*" onChange={handleFileChange} style={{display:'none'}} />
              <label htmlFor="file-upload" className="btn btn-primary">Upload Frame</label>
              <button onClick={generateDemoImage} className="btn btn-secondary">Generate Demo</button>
            </div>
            
            <div className="upload-section" style={{marginTop: '1rem'}}>
              <button onClick={toggleSyntheticStream} className={`btn ${isStreaming ? 'btn-accent' : 'btn-secondary'}`}>
                {isStreaming ? 'Stop Stream' : 'Live Synthetic Stream'}
              </button>
              <button onClick={toggleWebcam} className={`btn ${isWebcamActive ? 'btn-accent' : 'btn-secondary'}`}>
                {isWebcamActive ? 'Stop Webcam' : 'Use Webcam'}
              </button>
            </div>
            
            <div className="slider-section" style={{width: '100%', marginTop: '1.5rem'}}>
              <label htmlFor="confidence-threshold">
                Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
              </label>
              <input
                id="confidence-threshold"
                aria-label="Confidence threshold"
                type="range"
                min="0" max="1" step="0.05"
                value={confidenceThreshold}
                onChange={e => setConfidenceThreshold(parseFloat(e.target.value))}
                style={{width: '100%', marginTop: '0.5rem'}}
              />
            </div>

            <video ref={videoRef} autoPlay playsInline muted style={{display:'none'}} />
            <canvas ref={streamCanvasRef} style={{display:'none'}} />

            {preview && !isStreaming && !isWebcamActive && (
              <div className="preview-container" style={{marginTop: '1.5rem'}}>
                <img src={preview} alt="Input preview" className="preview-img" />
                <button 
                  onClick={handlePredict} 
                  className={`btn btn-accent run-btn ${loading ? 'loading' : ''}`}
                  disabled={loading}
                >
                  {loading ? 'Analyzing...' : 'Run Perception'}
                </button>
              </div>
            )}
          </div>

          {/* Backend / camera errors. Outside the results block so it still
              shows on the first failed request, when there is nothing to render. */}
          {error && (
            <div className="error-banner glass-panel" role="alert">
              <strong>Error:</strong> {error}
            </div>
          )}

          {/* Results */}
          {results && (
            <div className="results-panel glass-panel slide-up">
              <h2>Telemetry Results</h2>
              <div className="stats-bar">
                  <div className="stat">
                    <span>Detections</span>
                    <strong>{visibleDetections.length} shown / {detections.length} found</strong>
                  </div>
                  <div className="stat">
                    <span>Top Score</span>
                    <strong>
                      {detections.length > 0
                        ? `${(Math.max(...detections.map((d) => d.score)) * 100).toFixed(1)}%`
                        : 'No object detected'}
                    </strong>
                  </div>
              </div>

              {/* Per-class counts over the filtered set, so a model that has
                  quietly stopped seeing one class is visible at a glance. */}
              {visibleDetections.length > 0 && (
                <div className="class-breakdown">
                  {CLASS_NAMES.map((name, i) => {
                    const n = visibleDetections.filter((d) => d.class_id === i).length;
                    return (
                      <span
                        key={name}
                        className="class-chip"
                        style={{ borderColor: classColor(i), opacity: n ? 1 : 0.4 }}
                      >
                        <i style={{ background: classColor(i) }} />
                        {name} <b>{n}</b>
                      </span>
                    );
                  })}
                </div>
              )}
              
              <div className="grid-display">
                <div className="grid-item">
                  <h3>Input + Detection</h3>
                  <div style={{position: 'relative', width: '100%'}}>
                    <img src={preview || `data:image/png;base64,${results.input_b64}`} alt="Detection" />
                    {visibleDetections.map((det, i) => {
                      const color = classColor(det.class_id);
                      return (
                        <div
                          key={i}
                          style={{
                            position: 'absolute',
                            left: `${det.x * 100}%`,
                            top: `${det.y * 100}%`,
                            width: `${det.w * 100}%`,
                            height: `${det.h * 100}%`,
                            border: `2px solid ${color}`,
                            pointerEvents: 'none'
                          }}
                        >
                          <span
                            style={{
                              color: '#fff',
                              background: color,
                              fontSize: '10px',
                              padding: '0 3px',
                              position: 'absolute',
                              top: '-15px',
                              left: '-2px',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            {det.class ?? 'object'} {det.score.toFixed(2)}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
                <div className="grid-item">
                  <h3>Depth Map</h3>
                  <img src={`data:image/png;base64,${results.depth_b64}`} alt="Depth" />
                </div>
                <div className="grid-item">
                  <h3>Segmentation Mask</h3>
                  <img src={`data:image/png;base64,${results.seg_b64}`} alt="Segmentation" />
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
