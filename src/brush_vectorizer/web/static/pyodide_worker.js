/**
 * Pure Client-Side WebAssembly Worker for Brush Vectorizer Studio via Pyodide.
 * Runs the full Python multi-layer Painterly & Plotter pipelines in a background Web Worker.
 */

const PYODIDE_INDEX_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/";
importScripts(`${PYODIDE_INDEX_URL}pyodide.js`);

let pyodide = null;
let initPromise = null;
let pendingRender = null;
let actionQueue = [];
let isProcessing = false;

function postStatus(busy, text) {
  self.postMessage({ type: "status", busy, text });
}

async function ensureInitialized() {
  if (pyodide) return pyodide;
  if (initPromise) return initPromise;

  initPromise = (async () => {
    try {
      postStatus(true, "Loading Pyodide WebAssembly runtime...");
      const py = await loadPyodide({ indexURL: PYODIDE_INDEX_URL });

      postStatus(true, "Loading NumPy, SciPy, Pillow, NetworkX & Pydantic...");
      await py.loadPackage(["numpy", "scipy", "pillow", "pydantic", "networkx"]);

      postStatus(true, "Mounting Brush Vectorizer pipeline...");
      const zipResp = await fetch("./brush_vectorizer.zip", { cache: "no-cache" });
      if (!zipResp.ok) {
        throw new Error(`Failed to fetch brush_vectorizer.zip (HTTP ${zipResp.status})`);
      }
      const zipBuffer = await zipResp.arrayBuffer();
      py.FS.mkdirTree("/tmp/site-packages");
      py.FS.mkdirTree("/tmp/studio_workspace/samples");
      py.unpackArchive(zipBuffer, "zip", { extractDir: "/tmp/site-packages" });

      // Optionally pre-populate bundled sample images from ./samples/manifest.json if present
      try {
        const manifestResp = await fetch("./samples/manifest.json", { cache: "no-cache" });
        if (manifestResp.ok) {
          const manifest = await manifestResp.json();
          const files = Array.isArray(manifest.files) ? manifest.files : [];
          for (const filename of files) {
            try {
              const imgResp = await fetch(`./samples/${encodeURIComponent(filename)}`);
              if (imgResp.ok) {
                const buf = new Uint8Array(await imgResp.arrayBuffer());
                py.FS.writeFile(`/tmp/studio_workspace/samples/${filename}`, buf);
              }
            } catch (_e) {
              // Ignore optional sample load failure
            }
          }
        }
      } catch (_e) {
        // Optional manifest not present
      }

      py.runPython(`
import json
import sys
from pathlib import Path

if "/tmp/site-packages" not in sys.path:
    sys.path.insert(0, "/tmp/site-packages")

from brush_vectorizer.web.engine import (
    StudioState,
    _execute_render_sync,
    get_export_payload,
)

_STUDIO_STATE = StudioState(Path("/tmp/studio_workspace"))

def _py_get_samples() -> str:
    return json.dumps({"samples": _STUDIO_STATE.list_samples()})

def _py_register_upload(filename: str, raw_bytes) -> str:
    body = bytes(raw_bytes.to_py())
    res = _STUDIO_STATE.register_uploaded_bytes(filename, body)
    return json.dumps(res)

def _py_render(payload_json: str) -> str:
    payload = json.loads(payload_json)
    res = _execute_render_sync(
        state=_STUDIO_STATE,
        image_id=payload.get("image_id", "color_landscape_demo.png"),
        mode=payload.get("mode", "painterly"),
        resolution=payload.get("resolution", "preview"),
        raw_params=payload.get("params", {}),
    )
    return json.dumps(res)

def _py_export(fmt: str, params_json: str) -> str:
    target_params = json.loads(params_json) if params_json else None
    res = get_export_payload(_STUDIO_STATE, fmt, target_params)
    return json.dumps(res)
`);

      pyodide = py;
      return py;
    } catch (err) {
      initPromise = null;
      throw err;
    }
  })();

  return initPromise;
}

async function drainQueue() {
  if (isProcessing) return;
  isProcessing = true;

  try {
    const py = await ensureInitialized();

    while (actionQueue.length > 0 || pendingRender !== null) {
      // Process non-render actions (getSamples, upload, export) first in FIFO order
      while (actionQueue.length > 0) {
        const msg = actionQueue.shift();
        await handleAction(py, msg);
      }

      // Yield a macrotask tick so any burst of queued slider messages coalesces into pendingRender
      if (pendingRender !== null) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }

      if (pendingRender !== null) {
        const renderMsg = pendingRender;
        pendingRender = null;
        await handleRender(py, renderMsg);
      }
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      error: err && err.message ? err.message : String(err),
    });
  } finally {
    isProcessing = false;
    if (actionQueue.length > 0 || pendingRender !== null) {
      drainQueue();
    }
  }
}

async function handleAction(py, msg) {
  const { type, callId } = msg;
  try {
    if (type === "getSamples") {
      const fn = py.globals.get("_py_get_samples");
      const jsonStr = fn();
      fn.destroy();
      self.postMessage({ type: "response", callId, data: JSON.parse(jsonStr) });
    } else if (type === "upload") {
      const fn = py.globals.get("_py_register_upload");
      const uint8 = new Uint8Array(msg.buffer);
      const jsonStr = fn(msg.filename, uint8);
      fn.destroy();
      self.postMessage({ type: "response", callId, data: JSON.parse(jsonStr) });
    } else if (type === "export") {
      const fn = py.globals.get("_py_export");
      const jsonStr = fn(msg.fmt, JSON.stringify(msg.params || null));
      fn.destroy();
      self.postMessage({ type: "response", callId, data: JSON.parse(jsonStr) });
    }
  } catch (err) {
    self.postMessage({
      type: "response",
      callId,
      error: err && err.message ? err.message : String(err),
    });
  }
}

async function handleRender(py, msg) {
  const { reqId, payload } = msg;
  try {
    const fn = py.globals.get("_py_render");
    const jsonStr = fn(JSON.stringify(payload));
    fn.destroy();
    self.postMessage({
      type: "renderResult",
      reqId,
      data: JSON.parse(jsonStr),
    });
  } catch (err) {
    self.postMessage({
      type: "renderResult",
      reqId,
      data: { error: err && err.message ? err.message : String(err) },
    });
  }
}

self.onmessage = (event) => {
  const msg = event.data;
  if (!msg || !msg.type) return;

  if (msg.type === "render") {
    // Coalesce rapid slider renders: supersede any unstarted render request
    pendingRender = msg;
  } else {
    actionQueue.push(msg);
  }
  drainQueue();
};
