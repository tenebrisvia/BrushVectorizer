# Brush Vectorizer Studio

**Brush Vectorizer Studio** is a **100% client-side WebAssembly (Pyodide)** interactive web application that converts 2D raster artwork and photographs into **multi-layer brushwork vectors** for two output targets:

1. **Painterly Multi-Layer SVG (`Painterly Multi-Layer` mode)**: Monitor-targeted variable-width brush strokes with automatically extracted color palettes, 3-stop tonal gradients, dry-brush bristle filaments, paper grain, and creative/abstract brush mutations.
2. **Multi-Layer Acrylic Plotter SVG & JSON (`Plotter` mode)**: Physical centerline toolpaths decomposed into the **same $K$ color/tonal paint layers**, with physical brush stepover subdivision for wide acrylic washes, bristle-drag kinematics compensation, 3-axis $Z$/feedrate profiling, per-layer TSP path ordering, and a toggleable **Opaque Acrylic Footprint Preview**.

Everything runs directly inside the user's browser via a background **Pyodide Web Worker** (`numpy`, `scipy`, `pillow`, `networkx`, `pydantic`). No backend server is required, and uploaded images never leave the user's device.

---

## How the Vectorization Pipeline Works

Instead of tracing outer shape contours like standard bitmap vectorizers, Brush Vectorizer Studio:
1. **Decomposes artwork into $K$ color/tonal paint layers** (`1_faint_wash` … `K_black_accents`) via joint `(R, G, B, Density)` K-Means clustering and cumulative underpainting masks (`Stage 1`).
2. **Extracts medial-axis centerlines** via Euclidean Distance Transform and Zhang-Suen thinning (`Stage 2`) and **untangles crossing strokes** (`Stage 3`) using momentum-preserving junction resolution so the brush flows naturally through intersections.
3. **Subdivides broad washes** (`Stage 4`) into parallel flowing passes scaled to your brush's physical splay radius ($R_{\max}$) and stepover factor.
4. **Resolves stroke polarity** (`Stage 5`) from heavy entry press ($t=0$) to feathered exit whip ($t=1$).
5. **Applies physical brush kinematics & per-layer TSP ordering** (`Stage 6`) to compensate for trailing bristle drag ($\alpha$) and minimize non-drawing rapid air travel within each paint layer.

---

## Building & Running the Static Web App

Because the entire application runs in the browser via Pyodide, building the static site bundle (`dist/`) only uses Python's standard library (`zipfile`, `shutil`, `pathlib`, `http.server`)—no third-party packages or virtual environment are needed to build or serve the web app.

### 1. Build & Launch Locally
From the repository root, run:

```bash
PYTHONPATH=src python3 -m brush_vectorizer.web.build_static --serve
```

Then open **[http://127.0.0.1:8080](http://127.0.0.1:8080)** in Chrome, Safari, Firefox, or Edge.

### 2. Build Only (For Static Hosting)
To generate the self-contained `dist/` folder without starting a local server:

```bash
PYTHONPATH=src python3 -m brush_vectorizer.web.build_static
```

This creates:
- `dist/index.html` — The interactive Web Studio UI.
- `dist/pyodide_worker.js` — Background Web Worker that loads Pyodide from the jsDelivr CDN and executes the vectorization pipeline in WebAssembly.
- `dist/brush_vectorizer.zip` — Compact (~60 KB) archive of the Python pipeline mounted into Pyodide's in-memory filesystem.
- `dist/samples/` — Bundled sample images and `manifest.json`.

### 3. Deploying to GitHub Pages / Cloudflare Pages / Netlify / Vercel
To deploy automatically on every `git push`, configure your static host with:
- **Build Command:** `PYTHONPATH=src python3 -m brush_vectorizer.web.build_static`
- **Publish / Output Directory:** `dist`

---

## Using the Interactive Web Studio

### Top Toolbar
- **Source Image Picker & Upload:** Switch between built-in sample images or click **Upload Image...** (you can also drag-and-drop any `.png` or `.jpg` directly onto the center canvas).
- **Fast Preview & Full-Res Exports:** Keep **Fast Preview** checked while adjusting sliders for responsive 480px updates. Click **Render Full-Res** to preview at 100% original pixel resolution, or click **Download SVG** / **JSON** to export full-resolution files.

### Left Sidebar (Mode, Presets & Parameter Controls)
- **Hover Tooltips:** Hover over any control row, dropdown, or mode button in the left sidebar to see a floating tooltip explaining what that parameter controls.
- **Pipeline Output Mode:** Toggle between **Painterly Multi-Layer** and **Plotter** at the top of the left sidebar. Both modes share the same $K$ color/tonal layers, extracted/custom color palettes, presets, topology controls, and brush spine mutations.
- **Presets Dropdown & Settings Export/Import:** Choose from 24 built-in presets. Use **Export Settings** and **Import Settings** directly below the preset selector to save and restore all left-sidebar parameters together with right-sidebar layer colors, opacities, blurs, and visibilities as a `.json` file.
- **Physical Plotter & Acrylic Brush Panel** *(when **Plotter** is selected)*:
  - **Plotter Viewport Preview:** Switch instantly between:
    - **Overlay** *(default)*: Crisp plotter centerline toolpaths overlaid on a 55% translucent acrylic footprint preview.
    - **Opaque Acrylic**: Hides the centerline wireframes and renders the variable-width physical brush footprints at **100% opacity**, letting you inspect how your physical brush radius ($R_{\max}$) and stepover factor cover the canvas across all $K$ paint layers.
    - **Centerlines**: Displays only the bare plotter centerline toolpaths.
  - **Physical Calibration Sliders:** Set **Canvas Max Width/Height (mm)**, **Max Brush Radius $R_{\max}$ (mm)**, **Acrylic Pass Stepover Factor**, and **Bristle Drag Coeff ($\alpha$)**.

### Center Viewport
- **Scroll wheel** to zoom toward your cursor; **click and drag** to pan; use the bottom HUD buttons (**`-`**, **`+`**, **`Fit`**, **`1:1`**) to reset your view.

### Right Sidebar (Render Metrics & Interactive SVG Layers)
- Click any layer's **Color Swatch** to customize its paint color (or paper canvas tone) at 60fps—updating both Painterly strokes and Plotter centerlines/acrylic footprints in real time.
- Toggle **Visible**, **Solo** a single layer, or adjust **Opacity** and per-layer **Blur** sliders.
- Click **Reset Colors** to restore the automatically extracted source palette, or **Show All** to reset layer visibilities.

---

## Plotting Multi-Layer Acrylics

1. **Layer-by-Layer Color Plotting (`1_faint_wash` … `K_black_accents`):**
   - Each paint layer in the exported Plotter SVG is a numbered Inkscape layer (`1_faint_wash`, `2_light_wash`, `3_soft_midtone`, …) containing `<path fill="none" stroke="..."/>` centerlines ordered via banded TSP.
   - Using Inkscape's plotter layer controls, you can plot one paint color layer at a time from back (lightest underpainting wash) to front (darkest detail accents).
   - Layers prefixed with `0_` (`0_paper_canvas` and `0_preview_acrylic_footprint`) are non-plotting reference layers and are automatically ignored when plotting layers `1` through `K` by number.
2. **Physical Brush Calibration:**
   - **Max Brush Radius ($R_{\max}$):** Measure the half-width of your acrylic brush mark at full pressure. Regions wider than this are automatically filled with parallel centerline strokes spaced by $\text{Stepover} \times R_{\max}$.
   - **Bristle Drag Compensation ($\alpha$):** Keep around `0.75–1.0` for flexible synthetic/sable acrylic brushes so the carriage leads the trailing bristle tip through curves, or set to `0.0` for rigid acrylic paint markers.

---

## Running Unit Tests (For Contributors)

If you have a local Python environment with `numpy`, `scipy`, `networkx`, `pillow`, and `pydantic` installed, you can run the automated test suite:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

---

## License

This project is released under the [MIT License](LICENSE).

---

Vibe-coded with Gemini 4 Argon 