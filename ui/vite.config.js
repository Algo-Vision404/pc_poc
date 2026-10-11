import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Matches the port documented in README.md. `host: true` binds 0.0.0.0 so
    // the dashboard is reachable from outside the machine/container.
    port: 3000,
    host: true,
    strictPort: true,
  },
})
