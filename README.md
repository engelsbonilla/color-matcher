# Color Matcher Pro

This project contains a self‑contained HTML application for calibrating
colors using a 50‑patch system enhanced with AI. The page is built with
React, TailwindCSS and TensorFlow.js, all loaded from CDNs. The calibration
tool can be run entirely in a modern web browser.

## Prerequisites

- **Node.js** (v18 or later) &ndash; optional, used only when serving the page
  from a local HTTP server.
- **Modern web browser** &ndash; required to run the application (Chrome,
  Firefox, Safari or Edge are recommended).

## Usage

1. Clone this repository:

   ```bash
   git clone <repo-url>
   cd color-matcher
   ```

2. Start a simple local server (recommended):

   ```bash
   npx http-server
   ```

   The page will be available at `http://localhost:8080/`.

   Alternatively, open the HTML file directly in your browser, though some
   browsers may restrict certain functionality when run from the `file://`
   scheme.

3. Load `ColorMatcherPro_CALIBRATION_FIXED_v2.1.8.html` in the browser. It
   provides the AI driven 50 patch calibration workflow where you can capture
   patches, refine color matches and export profiles.

## Testing

Run the Jest test suite with:

```bash
npm test
```

This command executes all tests in the `__tests__` directory.

## Purpose of `ColorMatcherPro_CALIBRATION_FIXED_v2.1.8.html`

`ColorMatcherPro_CALIBRATION_FIXED_v2.1.8.html` is the main tool of this
repository. It contains the calibration UI and logic for measuring and learning
color patches to build accurate profiles. The “FIXED” tag denotes improvements
in handling device data, while version **2.1.8** includes the latest adjustments
for more reliable color matching.

## Configuration

The `AIColorModel` accepts optional thresholds for ΔL, Δa and Δb when
instantiated. These thresholds determine how sensitive the model is when
converting LAB errors to CMYK adjustments. In version **2.1.8** the
constructor defaults to **0.5** for each threshold so very small LAB
differences are ignored by default. To account for every difference,
override the thresholds to **0** when creating the model:

```javascript
const ai = new AIColorModel({
  deltaLThreshold: 0,
  deltaAThreshold: 0,
  deltaBThreshold: 0
});
```

Adjust these values if you want to dampen very small adjustments. Raising the thresholds makes the model ignore minor LAB differences when calculating CMYK changes.

## Development

The project ships with an ESLint setup that checks HTML files using the
`eslint-plugin-html` plugin. Run `npm install` to install the required
development dependencies such as Jest and ESLint before invoking
`npm run lint` or `npm test`.
