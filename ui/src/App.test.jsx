import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App.jsx';

const HEALTH = {
  status: 'ready',
  arch_version: 'big-per-camera-net-v3',
  checkpoint_epoch: 19,
  device: 'cpu',
  classes: ['block', 'sphere', 'triangle'],
};

const PREDICT_BODY = {
  input_b64: 'AAAA',
  seg_b64: 'AAAA',
  depth_b64: 'AAAA',
  image_size: { width: 128, height: 128 },
  score: 0.92,
  detections: [
    { x: 0.1, y: 0.1, w: 0.2, h: 0.2, score: 0.92, class: 'block', class_id: 0 },
    { x: 0.4, y: 0.4, w: 0.2, h: 0.2, score: 0.71, class: 'sphere', class_id: 1 },
    { x: 0.7, y: 0.7, w: 0.2, h: 0.2, score: 0.31, class: 'triangle', class_id: 2 },
  ],
};

function json(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  };
}

/**
 * Stub fetch and route by path. The real api.js module runs underneath, so
 * these tests cover the client and the component together rather than a mock
 * standing in for the thing being tested.
 */
function stubBackend({ health: healthBody = HEALTH, healthStatus = 200, predict: predictBody = PREDICT_BODY, predictStatus = 200 } = {}) {
  const calls = { health: 0, predict: 0 };
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url) => {
      if (url.endsWith('/health')) {
        calls.health += 1;
        return json(healthBody, healthStatus);
      }
      calls.predict += 1;
      return json(predictBody, predictStatus);
    })
  );
  return calls;
}

async function uploadAndPredict(user) {
  const file = new File(['png-bytes'], 'frame.png', { type: 'image/png' });
  await user.upload(screen.getByLabelText('Upload Frame'), file);
  await user.click(screen.getByRole('button', { name: /run perception/i }));
}

describe('App', () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it('shows which checkpoint the backend loaded', async () => {
    stubBackend();
    render(<App />);

    expect(screen.getByTestId('server-badge')).toHaveTextContent(/connecting/i);

    await waitFor(() => {
      expect(screen.getByTestId('server-badge')).toHaveTextContent(
        'big-per-camera-net-v3 · epoch 19 · cpu'
      );
    });
  });

  it('surfaces an unreachable backend instead of failing silently', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch');
      })
    );
    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Backend unreachable');
    });
  });

  it('shows the API detail when a prediction is rejected', async () => {
    stubBackend({
      predict: { detail: 'Unsupported image format' },
      predictStatus: 400,
    });
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Unsupported image format');
    });
  });

  it('renders results and counts detections', async () => {
    stubBackend();
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);

    await waitFor(() => {
      expect(screen.getByText('Telemetry Results')).toBeInTheDocument();
    });
    // Default threshold is 0.5, so the 0.31 triangle is filtered out.
    expect(screen.getByText(/2 shown \/ 3 found/)).toBeInTheDocument();
  });

  it('filters detections by the confidence threshold', async () => {
    stubBackend();
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);
    await waitFor(() => screen.getByText(/2 shown \/ 3 found/));

    // Dropping the threshold to 0 admits the 0.31 triangle. Range inputs do
    // not respond to typing, so drive the change event directly.
    fireEvent.change(screen.getByLabelText('Confidence threshold'), {
      target: { value: '0' },
    });

    await waitFor(() => {
      expect(screen.getByText(/3 shown \/ 3 found/)).toBeInTheDocument();
    });
  });

  it('labels boxes by class name and colours them per class', async () => {
    stubBackend();
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);
    await waitFor(() => screen.getByText(/2 shown \/ 3 found/));

    const overlay = screen.getByAltText('Detection').parentElement;
    const boxes = [...overlay.querySelectorAll('div[style*="border"]')];

    expect(boxes).toHaveLength(2);
    const styles = boxes.map((b) => b.getAttribute('style'));
    expect(styles.some((s) => s.includes('rgb(239, 68, 68)'))).toBe(true); // block
    expect(styles.some((s) => s.includes('rgb(34, 197, 94)'))).toBe(true); // sphere
    expect(styles.some((s) => s.includes('rgb(59, 130, 246)'))).toBe(false); // triangle filtered

    const labels = boxes.map((b) => b.querySelector('span').textContent);
    expect(labels.some((t) => t.startsWith('block'))).toBe(true);
    expect(labels.some((t) => t.startsWith('sphere'))).toBe(true);
  });

  it('reports per-class counts so a blind class is visible', async () => {
    stubBackend();
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);
    await waitFor(() => screen.getByText(/2 shown \/ 3 found/));

    const chips = [...document.querySelectorAll('.class-chip')].map((c) => c.textContent);
    expect(chips).toContain('block 1');
    expect(chips).toContain('sphere 1');
    expect(chips).toContain('triangle 0');
  });

  it('says so explicitly when nothing is detected', async () => {
    stubBackend({ predict: { ...PREDICT_BODY, detections: [], score: 0 } });
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);

    await waitFor(() => {
      expect(screen.getByText('No object detected')).toBeInTheDocument();
    });
    // Regression: an empty frame must not leave a stale error on screen.
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('clears a previous error once a prediction succeeds', async () => {
    let fail = true;
    vi.stubGlobal(
      'fetch',
      vi.fn(async (url) => {
        if (url.endsWith('/health')) return json(HEALTH);
        if (fail) {
          fail = false;
          return json({ detail: 'Backend returned 503' }, 503);
        }
        return json(PREDICT_BODY);
      })
    );
    const user = userEvent.setup();
    render(<App />);

    await uploadAndPredict(user);
    await waitFor(() => screen.getByRole('alert'));

    await user.click(screen.getByRole('button', { name: /run perception/i }));

    await waitFor(() => {
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
  });

  it('offers no predict button until a frame is selected', async () => {
    const calls = stubBackend();
    render(<App />);

    // Let the health probe settle first, otherwise its state update lands
    // after the assertions and React flags it as an unwrapped update.
    await waitFor(() => {
      expect(screen.getByTestId('server-badge')).toHaveTextContent(/v3/);
    });

    // The button is gated on having a frame, so there is nothing to click
    // rather than a disabled control that looks broken.
    expect(
      screen.queryByRole('button', { name: /run perception/i })
    ).not.toBeInTheDocument();
    expect(calls.predict).toBe(0);
  });
});
