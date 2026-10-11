import { useState, useRef, useEffect } from 'react';
import './index.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  
  // Webcam & Stream references
  const videoRef = useRef(null);
  const streamCanvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const streamIntervalRef = useRef(null);

  // Generate synthetic frame on a canvas
  const createSyntheticFrame = (canvas) => {
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    
    // Background
    ctx.fillStyle = '#1e1e1e';
    ctx.fillRect(0, 0, 128, 128);
    
    // Random rect
    const w = Math.floor(Math.random() * 28) + 12;
    const h = Math.floor(Math.random() * 28) + 12;
    const x = Math.floor(Math.random() * (128 - w));
    const y = Math.floor(Math.random() * (128 - h));
    
    const r = Math.floor(Math.random() * 195) + 60;
    const g = Math.floor(Math.random() * 195) + 60;
    const b = Math.floor(Math.random() * 195) + 60;
    
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y, w, h);
    return canvas;
  };

  const generateDemoImage = () => {
    if (isStreaming || isWebcamActive) stopAllStreams();
    const canvas = document.createElement('canvas');
    createSyntheticFrame(canvas);
    
    canvas.toBlob((blob) => {
      const file = new File([blob], 'demo.png', { type: 'image/png' });
      handleFileChange({ target: { files: [file] } });
    });
  };

  const handleFileChange = (e) => {
    if (isStreaming || isWebcamActive) stopAllStreams();
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
    }
  };

  const performPrediction = async (fileToPredict) => {
    const formData = new FormData();
    formData.append('file', fileToPredict);
    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Prediction failed');
      return await res.json();
    } catch (err) {
      console.error(err);
      return null;
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    const data = await performPrediction(selectedFile);
    if (data) setResults(data);
    else alert('Error connecting to backend.');
    setLoading(false);
  };

  // Live Stream Logic
  const toggleSyntheticStream = () => {
    if (isStreaming) {
      stopAllStreams();
    } else {
      stopAllStreams();
      setIsStreaming(true);
      streamIntervalRef.current = setInterval(async () => {
        const canvas = document.createElement('canvas');
        createSyntheticFrame(canvas);
        setPreview(canvas.toDataURL('image/png'));
        canvas.toBlob(async (blob) => {
          const file = new File([blob], 'stream.png', { type: 'image/png' });
          const data = await performPrediction(file);
          if (data) setResults(data);
        });
      }, 500); // 2 FPS
    }
  };

  // Webcam Logic
  const toggleWebcam = async () => {
    if (isWebcamActive) {
      stopAllStreams();
    } else {
      stopAllStreams();
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 128, height: 128 } });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setIsWebcamActive(true);
          
          streamIntervalRef.current = setInterval(() => {
            if (!videoRef.current || !streamCanvasRef.current) return;
            const canvas = streamCanvasRef.current;
            const ctx = canvas.getContext('2d');
            canvas.width = 128;
            canvas.height = 128;
            ctx.drawImage(videoRef.current, 0, 0, 128, 128);
            
            setPreview(canvas.toDataURL('image/png'));
            canvas.toBlob(async (blob) => {
              const file = new File([blob], 'webcam.png', { type: 'image/png' });
              const data = await performPrediction(file);
              if (data) setResults(data);
            });
          }, 1000); // 1 FPS
        }
      } catch (err) {
        console.error("Webcam error:", err);
        alert("Webcam access denied or unavailable.");
      }
    }
  };

  const stopAllStreams = () => {
    if (streamIntervalRef.current) clearInterval(streamIntervalRef.current);
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
    setIsStreaming(false);
    setIsWebcamActive(false);
  };

  useEffect(() => {
    return () => stopAllStreams(); // Cleanup on unmount
  }, []);

  const shouldShowDet = results && results.score >= confidenceThreshold;

  return (
    <div className="app-container">
      <header className="header">
        <h1>Per-Camera PoC Intelligence</h1>
        <p>Real-time autonomous perception command center.</p>
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
              <label>Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%</label>
              <input 
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

          {/* Results */}
          {results && (
            <div className="results-panel glass-panel slide-up">
              <h2>Telemetry Results</h2>
              <div className="stats-bar">
                  <div className="stat"><span>Detection Score</span> <strong>{(results.score * 100).toFixed(1)}%</strong></div>
                  <div className="stat">
                    <span>BBox [x,y,w,h]</span> 
                    <strong>
                      {shouldShowDet 
                        ? `[${results.bbox.x.toFixed(2)}, ${results.bbox.y.toFixed(2)}, ${results.bbox.w.toFixed(2)}, ${results.bbox.h.toFixed(2)}]` 
                        : "Suppressed (Low Confidence)"}
                    </strong>
                  </div>
              </div>
              
              <div className="grid-display">
                <div className="grid-item">
                  <h3>Input + Detection</h3>
                  <div style={{position: 'relative', width: '100%'}}>
                    <img src={preview || `data:image/png;base64,${results.input_b64}`} alt="Detection" />
                    {shouldShowDet && (
                      <div style={{
                        position: 'absolute',
                        left: `${(results.bbox.x - results.bbox.w/2) * 100}%`,
                        top: `${(results.bbox.y - results.bbox.h/2) * 100}%`,
                        width: `${results.bbox.w * 100}%`,
                        height: `${results.bbox.h * 100}%`,
                        border: '2px solid red',
                        pointerEvents: 'none'
                      }}>
                        <span style={{color:'red', fontSize:'10px', position:'absolute', top:'-15px', left:'-2px'}}>Det: {results.score.toFixed(2)}</span>
                      </div>
                    )}
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
