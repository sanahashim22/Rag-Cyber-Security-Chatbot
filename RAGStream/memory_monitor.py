"""
memory_monitor.py
Live memory monitoring table for the Hotel RAG Assistant.

FIX FOR REMOTE ACCESS (LAN / instructor's PC):
  The old version hardcoded "127.0.0.1:8502" in both the server bind address
  and the JavaScript fetch URL.  When the instructor opened the app via the
  office PC's LAN IP (e.g. http://192.168.x.x:8501), their browser's
  "127.0.0.1" pointed to THEIR OWN laptop – which has no stats server – so
  the table showed only dashes.

  Fix applied here:
    1. Server now binds to "0.0.0.0" (all interfaces) so it is reachable
       from ANY device on the same network, not just localhost.
    2. The JavaScript fetch URL is built dynamically from
       window.parent.location.hostname, so the request always targets the
       SAME host the app is being viewed from (localhost for local use,
       192.168.x.x for LAN use). Both cases hit the same stats server on the
       office PC.

  IMPORTANT – firewall:  port 8502 must be allowed through Windows Firewall,
  just like port 8501 was.  Run once in an elevated PowerShell:
      New-NetFirewallRule -DisplayName "RAG Stats Server" -Direction Inbound `
          -Protocol TCP -LocalPort 8502 -Action Allow

HOW IT WORKS (unchanged from original design):
  A lightweight background HTTP server (Python built-in http.server, one
  daemon thread started on first app rerun) listens on 0.0.0.0:8502 and
  serves live psutil / Ollama readings as JSON at /mem-stats.

  A Streamlit HTML component injects a tiny JS snippet that:
    - Builds the floating table once (no duplication on reruns).
    - Polls /mem-stats every 2 seconds via fetch() and updates only the
      text nodes – no Streamlit rerun, no page blink.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import psutil
import requests
import streamlit.components.v1 as components

OLLAMA_API = "http://localhost:11434/api/ps"
STATS_PORT = 8502


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------

def _ollama_process_memory_mb() -> float:
    """Return total RSS (MB) of all processes whose name contains 'ollama'."""
    total = 0.0
    for proc in psutil.process_iter(["name"]):
        try:
            if proc.info["name"] and "ollama" in proc.info["name"].lower():
                total += proc.memory_info().rss / (1024 ** 2)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _loaded_model_memory() -> list:
    """Return list of {name, size_mb} dicts from Ollama's /api/ps endpoint."""
    try:
        r = requests.get(OLLAMA_API, timeout=1)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [
            {"name": m.get("name"), "size_mb": m.get("size", 0) / (1024 ** 2)}
            for m in models
        ]
    except Exception:
        return []


def _get_snapshot() -> dict:
    """Collect all memory readings and return as a single dict."""
    vm = psutil.virtual_memory()
    return {
        "models": _loaded_model_memory(),
        "ollama_process_mb": _ollama_process_memory_mb(),
        "system_used_mb": vm.used / (1024 ** 2),
        "system_total_mb": vm.total / (1024 ** 2),
        "system_percent": vm.percent,
    }


# ---------------------------------------------------------------------------
# Lightweight HTTP server (runs in a daemon thread)
# ---------------------------------------------------------------------------

class _StatsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/mem-stats":
            payload = json.dumps(_get_snapshot()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            # Allow any origin so the browser's fetch() succeeds regardless of
            # which hostname was used to access the Streamlit app.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # Browser closed the connection mid-response (e.g. on page
                # refresh). Harmless – just swallow the error silently.
                pass
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress the per-request log lines in the terminal.
        pass


_server_lock = threading.Lock()
_server_started = False


def _start_stats_server_once():
    """Start the stats HTTP server exactly once across all Streamlit reruns."""
    global _server_started
    with _server_lock:
        if _server_started:
            return

        def _run():
            try:
                # KEY FIX: bind to "0.0.0.0" so the server is reachable from
                # other devices on the LAN, not just from localhost.
                HTTPServer(("0.0.0.0", STATS_PORT), _StatsHandler).serve_forever()
            except OSError:
                # Port already in use (e.g. a previous Streamlit worker still
                # running). Nothing to do – the existing server keeps serving.
                pass

        threading.Thread(target=_run, daemon=True).start()
        _server_started = True


# ---------------------------------------------------------------------------
# Streamlit component
# ---------------------------------------------------------------------------

def render_memory_bar(current_model: str = None):
    """
    Inject the floating memory-monitor table into the Streamlit page.

    Call this once per rerun, AFTER the model dropdown has been read so that
    `current_model` is already known.

    Args:
        current_model: The model name currently selected in the sidebar
                       (e.g. "llama3.2"). Pass None when no model is selected.
    """
    _start_stats_server_once()
    model_json = json.dumps(current_model)  # produces "null" or '"llama3.2"' etc.

    html = f"""
    <script>
    (function() {{
        const doc = window.parent.document;

        // Build the table only once; subsequent reruns just update the data.
        let box = doc.getElementById("mem-monitor-box");
        if (!box) {{
            box = doc.createElement("div");
            box.id = "mem-monitor-box";
            box.style = (
                "position:fixed; top:60px; right:8px; z-index:9999; " +
                "background:#111827; color:#f3f4f6; border-radius:8px; " +
                "padding:8px 10px; font-family:monospace; font-size:11px; " +
                "box-shadow:0 2px 8px rgba(0,0,0,0.35); width:290px;"
            );
            box.innerHTML = `
                <table style="width:100%; border-collapse:collapse; font-size:11px;">
                  <tr style="border-bottom:1px solid #374151;">
                    <th style="text-align:left;  padding:3px 4px;">Memory usage</th>
                    <th style="text-align:right; padding:3px 4px;">%</th>
                    <th style="text-align:right; padding:3px 4px;">GB</th>
                  </tr>
                  <tr>
                    <td id="mm-model-name"
                        style="padding:3px 4px; white-space:nowrap;
                               overflow:hidden; text-overflow:ellipsis;
                               max-width:130px;"></td>
                    <td id="mm-model-pct"  style="text-align:right; padding:3px 4px;">-</td>
                    <td id="mm-model-gb"   style="text-align:right; padding:3px 4px;">-</td>
                  </tr>
                  <tr>
                    <td style="padding:3px 4px;">Ollama</td>
                    <td id="mm-ollama-pct" style="text-align:right; padding:3px 4px;">-</td>
                    <td id="mm-ollama-gb"  style="text-align:right; padding:3px 4px;">-</td>
                  </tr>
                  <tr>
                    <td style="padding:3px 4px;">Total system</td>
                    <td id="mm-sys-pct"   style="text-align:right; padding:3px 4px;">-</td>
                    <td id="mm-sys-gb"    style="text-align:right; padding:3px 4px;">-</td>
                  </tr>
                </table>`;
            doc.body.appendChild(box);
        }}

        const currentModel = {model_json};
        doc.getElementById("mm-model-name").textContent =
            currentModel || "no model selected";

        // KEY FIX: derive the host from the actual page URL instead of
        // hardcoding "127.0.0.1".
        //
        //   Local access  → window.parent.location.hostname === "localhost"
        //                    fetch → http://localhost:{STATS_PORT}/mem-stats   ✓
        //
        //   LAN access    → window.parent.location.hostname === "192.168.x.x"
        //                    fetch → http://192.168.x.x:{STATS_PORT}/mem-stats ✓
        //
        // In both cases the request reaches the stats server on the OFFICE PC,
        // so the instructor's browser will show the same live readings as yours.
        const statsHost = window.parent.location.hostname;
        const statsUrl  = "http://" + statsHost + ":{STATS_PORT}/mem-stats";

        async function tick() {{
            try {{
                const res  = await fetch(statsUrl);
                const data = await res.json();
                const totalMb = data.system_total_mb || 1;

                // --- Model row ---
                let modelGb = 0, modelPct = 0;
                if (currentModel) {{
                    const m = data.models.find(
                        x => x.name && x.name.includes(currentModel)
                    );
                    if (m) {{
                        modelGb  = m.size_mb / 1024;
                        modelPct = (m.size_mb / totalMb) * 100;
                    }}
                }}
                doc.getElementById("mm-model-pct").textContent =
                    modelPct.toFixed(1) + "%";
                doc.getElementById("mm-model-gb").textContent =
                    modelGb.toFixed(2);

                // --- Ollama row ---
                const ollamaGb  = data.ollama_process_mb / 1024;
                const ollamaPct = (data.ollama_process_mb / totalMb) * 100;
                doc.getElementById("mm-ollama-pct").textContent =
                    ollamaPct.toFixed(1) + "%";
                doc.getElementById("mm-ollama-gb").textContent =
                    ollamaGb.toFixed(2);

                // --- System row ---
                const sysGb = data.system_used_mb / 1024;
                doc.getElementById("mm-sys-pct").textContent =
                    data.system_percent.toFixed(1) + "%";
                doc.getElementById("mm-sys-gb").textContent =
                    sysGb.toFixed(2);

            }} catch (e) {{
                // Network error or server not yet ready – silently retry next tick.
                console.warn("mem-monitor fetch failed:", e);
            }}
        }}

        // Fire immediately, then every 2 seconds.
        tick();
        if (window._memMonitorInterval) clearInterval(window._memMonitorInterval);
        window._memMonitorInterval = setInterval(tick, 2000);
    }})();
    </script>
    """
    components.html(html, height=0)
