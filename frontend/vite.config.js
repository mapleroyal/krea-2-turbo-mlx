import { reactRouter } from "@react-router/dev/vite";
import tailwindcss from "@tailwindcss/vite";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const isVitest = process.env.VITEST === "true";

export default defineConfig({
  esbuild: isVitest ? { jsx: "automatic" } : undefined,
  plugins: [!isVitest && reactRouter(), tailwindcss()].filter(Boolean),
  server: {
    host: true,
  },
  test: {
    environment: "jsdom",
    globals: true,
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./app"),
    },
  },
});
