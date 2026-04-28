import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwind from '@astrojs/tailwind';

// The repo will be served at https://icemap.github.io/duckdb-vector-index/
// so every asset must be prefixed with /duckdb-vector-index/.
export default defineConfig({
  site: 'https://icemap.github.io',
  base: '/duckdb-vector-index',
  trailingSlash: 'ignore',
  integrations: [react(), tailwind({ applyBaseStyles: false })],
  build: {
    format: 'directory',
  },
});
