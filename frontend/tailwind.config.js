/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        navy: {
          950: '#050b18',
          900: '#0a1224',
          800: '#0f1b33',
          700: '#162544',
          600: '#1f3258',
        },
      },
    },
  },
  plugins: [],
}
