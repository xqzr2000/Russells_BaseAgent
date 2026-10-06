import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The UI calls /api/*; in development Vite forwards those requests to the
// Python server, so only this one port needs to be opened in Codespaces.
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
});
