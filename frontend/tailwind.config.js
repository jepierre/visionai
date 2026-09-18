/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['IBM Plex Sans', 'Segoe UI', 'sans-serif'],
      },
      boxShadow: {
        glass: '0 24px 60px rgba(0, 0, 0, 0.28)',
      },
      colors: {
        vision: {
          ink: '#091017',
          steel: '#102230',
          slate: '#1a3041',
          gold: '#ffd36f',
          coral: '#ff9254',
          aqua: '#3ab4ff',
        },
      },
      backgroundImage: {
        hero: 'radial-gradient(circle at top, rgba(255, 145, 77, 0.18), transparent 32%), linear-gradient(140deg, #091017 0%, #102230 52%, #1a3041 100%)',
      },
    },
  },
  plugins: [],
};