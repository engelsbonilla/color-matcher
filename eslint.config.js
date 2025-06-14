module.exports = [
  {
    files: ["**/*.{js,jsx,html}"],
    languageOptions: {
      parserOptions: {
        ecmaVersion: "latest",
        sourceType: "module",
        ecmaFeatures: {
          jsx: true
        }
      }
    },
    plugins: {
      html: require('eslint-plugin-html')
    },
    rules: {
    }
  }
];
