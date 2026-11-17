"""
alert_receiver.py
------------------
Standalone listener for IDS-Dashboard's "Send" feature (see its
dashboard_api.py: POST to IDS_LLM_URL with the selected alerts' export as
the body). Run this alongside the Streamlit dashboard - it just writes
whatever it receives into alert_inputs/, which dashboard_app.py's live
folder-watch already picks up automatically. Doesn't touch dashboard_app.py
or snort_ranking.py at all.

Usage:
    python alert_receiver.py [--port 8600] [--token SOME_SHARED_SECRET]

Then on the IDS-Dashboard side (its .env):
    IDS_LLM_URL=http://<this-pc's-ip>:8600/alerts
    IDS_LLM_TOKEN=SOME_SHARED_SECRET      # only if you passed --token here too

No format handling/validation of the received JSON happens here - it's
saved as-is. Whether dashboard_app.py's folder scan can use it depends on
its shape, same as any manually-dropped file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALERT_INPUTS_DIR = Path(__file__).resolve().parent / "alert_inputs"
ALERT_INPUTS_DIR.mkdir(exist_ok=True)

MAX_BODY_BYTES = 200 * 1024 * 1024  # 200 MB safety cap against a runaway/garbled Content-Length


def _safe_filename(raw_name: str | None) -> str:
    """Takes just the basename (no directory components - blocks path
    traversal via a crafted X-Filename header), strips anything that isn't
    alphanumeric/dot/dash/underscore, and guarantees a .json extension."""
    name = Path(raw_name).name if raw_name else ""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name) or "received"
    if not name.lower().endswith(".json"):
        name += ".json"
    return name


def _unique_path(name: str) -> Path:
    """Avoids clobbering an existing file of the same name by inserting a
    timestamp before the extension on collision."""
    candidate = ALERT_INPUTS_DIR / name
    if not candidate.exists():
        return candidate
    stem, ext = name.rsplit(".", 1)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ALERT_INPUTS_DIR / f"{stem}_{stamp}.{ext}"


def make_handler(expected_token: str | None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter default logging
            print(f"[alert_receiver] {self.address_string()} - {fmt % args}")

        def _reject(self, code: int, message: str):
            body = json.dumps({"ok": False, "error": message}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if expected_token is not None:
                auth = self.headers.get("Authorization", "")
                if auth != f"Bearer {expected_token}":
                    self._reject(401, "missing/incorrect bearer token")
                    return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._reject(400, "invalid Content-Length")
                return

            if length <= 0:
                self._reject(400, "empty body")
                return
            if length > MAX_BODY_BYTES:
                self._reject(413, "body too large")
                return

            raw_bytes = self.rfile.read(length)

            try:
                raw_bytes.decode("utf-8")  # just validating it's text/JSON-ish, not re-encoding
            except UnicodeDecodeError:
                self._reject(400, "body is not valid UTF-8")
                return

            filename_header = self.headers.get("X-Filename")
            dest = _unique_path(_safe_filename(filename_header))
            dest.write_bytes(raw_bytes)

            print(
                f"[alert_receiver] saved {dest.name} ({length} bytes) "
                f"from {self.address_string()}, path={self.path}"
            )

            body = json.dumps({"ok": True, "saved_as": dest.name, "bytes": length}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            # Simple liveness check, e.g. GET http://<host>:<port>/ in a browser.
            body = json.dumps({
                "ok": True,
                "message": "alert_receiver is up - POST alerts JSON here, any path",
                "alert_inputs_dir": str(ALERT_INPUTS_DIR),
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--token", default=None, help="If set, require Authorization: Bearer <token> on incoming POSTs")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(args.token))
    print(f"[alert_receiver] listening on 0.0.0.0:{args.port}, saving into {ALERT_INPUTS_DIR}")
    print(f"[alert_receiver] point IDS_LLM_URL at: http://<this-pc-ip>:{args.port}/alerts")
    if args.token:
        print("[alert_receiver] bearer token auth is ON - set IDS_LLM_TOKEN to match on the sender side")
    else:
        print("[alert_receiver] no token configured - any POST on this port is accepted (fine for a lab LAN)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[alert_receiver] shutting down")
        server.shutdown()


if __name__ == "__main__":
    sys.exit(main())
