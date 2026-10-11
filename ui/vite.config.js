import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Empty prefix so non-VITE_ variables are readable here too. This config
  // runs in Node, not the browser.
  const env = loadEnv(mode, '.', '')

  return {
    plugins: [react()],
    server: {
      // Matches the port documented in README.md. `host: true` binds 0.0.0.0
      // so the dashboard is reachable from outside the machine/container.
      port: 3000,
      host: true,
      strictPort: true,
      // Vite rejects unknown Host headers to guard against DNS rebinding.
      // This is a dev-only server; set PER_CAMERA_ALLOWED_HOSTS to a
      // comma-separated list to restrict it, or leave unset to accept any.
      allowedHosts: env.PER_CAMERA_ALLOWED_HOSTS
        ? env.PER_CAMERA_ALLOWED_HOSTS.split(',').map((h) => h.trim())
        : true,
      // The browser must never call the API origin directly: it is a different
      // host in every real deployment, and mixed-content/CORS rules bite. Dev
      // requests go to /api and are proxied to the backend.
      proxy: {
        '/api': {
          target: env.PER_CAMERA_API_TARGET ?? 'http://127.0.0.1:8000',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
