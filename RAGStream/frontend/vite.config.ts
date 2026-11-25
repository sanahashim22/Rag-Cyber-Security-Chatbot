import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Local dev only: proxies /api to the FastAPI backend (uvicorn api.main:app
// --port 8502) so the dev server can run on its own port with hot reload.
// Anything meant to be reachable from another PC (via ngrok) instead uses
// the production build, served same-origin by FastAPI itself - see
// api/main.py and the project's ngrok-config.yml.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8502',
        changeOrigin: true,
      },
    },
  },
})
