/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./pages/**/*.{js,jsx}', './components/**/*.{js,jsx}', './hooks/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          500: '#1e66f5',
          700: '#1148b7',
        },
      },
    },
  },
  plugins: [],
};
