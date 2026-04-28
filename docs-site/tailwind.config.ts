import type { Config } from 'tailwindcss';

// Palette lifted verbatim from ref/design/53aba90c-...html so every surface
// color matches the mock 1:1.
export default {
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      colors: {
        base: '#2B2B2B',          // --bg-base  (page + section-panel bg)
        panel: '#1E1E1E',         // --bg-panel (code container bg)
        strip: '#1A1A1A',         // install-strip bg
        line: '#404040',          // --border-color
        mid: '#333333',           // muted strip bg (feature-matrices header)
        ink: {
          black: '#000000',
          900: '#1A1A1A',
          800: '#1E1E1E',
          700: '#242424',
          600: '#333333',
          500: '#404040',
        },
        accent: {
          blue: '#29C5FF',        // --accent-blue
          green: '#98F2D1',       // --accent-green
          yellow: '#F0CF65',      // --accent-yellow
        },
        fg: {
          DEFAULT: '#D4D4D4',     // --text-main
          dim: '#999999',         // --text-dim
          on: '#1A1A1A',          // --text-on-accent
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
