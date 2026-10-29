import {defineConfig} from 'vite';
export default defineConfig({
  server: {proxy: {'/api': 'http://127.0.0.1:8018', '/images': 'http://127.0.0.1:8018'}},
});
