from __future__ import annotations

"""Browser-based map editor for advanced curriculum configs."""

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import urlparse

from edit_advanced_curriculum_maps import TOOL_ORDER, MapModel


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced Curriculum Map Editor</title>
  <style>
    :root {
      --bg: #f7f7f2;
      --panel: #fffdf8;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d1d5db;
      --wall: #334155;
      --free: #f8fafc;
      --start: #2563eb;
      --goal: #f59e0b;
      --key: #10b981;
      --door: #a16207;
      --hazard: #ef4444;
      --bonus: #f97316;
      --accent: #0f766e;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", "Helvetica Neue", sans-serif; background: linear-gradient(135deg, #f3efe5, #f7f7f2); color: var(--ink); }
    .app { max-width: 1200px; margin: 0 auto; padding: 24px; display: grid; grid-template-columns: 280px 1fr; gap: 24px; }
    .panel { background: var(--panel); border: 1px solid #e5e7eb; border-radius: 18px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); padding: 18px; }
    h1 { margin: 0 0 6px; font-size: 28px; line-height: 1.1; }
    .sub { color: var(--muted); margin-bottom: 18px; }
    .toolbar h2, .meta h2 { margin: 0 0 12px; font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
    .tool-grid, .button-grid { display: grid; gap: 8px; }
    button, select { width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; background: white; color: var(--ink); font: inherit; cursor: pointer; }
    button.active { background: var(--accent); color: white; border-color: var(--accent); }
    button.secondary { background: #f8fafc; }
    .status { margin-top: 12px; min-height: 24px; color: var(--muted); font-size: 14px; }
    .board-wrap { overflow: auto; }
    .board { display: grid; gap: 2px; background: var(--line); padding: 2px; width: max-content; border-radius: 16px; }
    .cell { width: 56px; height: 56px; border: 0; border-radius: 10px; font-weight: 700; font-size: 18px; transition: transform 0.08s ease, box-shadow 0.08s ease; }
    .cell:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(15, 23, 42, 0.12); }
    .legend { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; margin-top: 16px; }
    .legend-item { display: flex; align-items: center; gap: 8px; font-size: 14px; }
    .swatch { width: 18px; height: 18px; border-radius: 5px; border: 1px solid rgba(0,0,0,0.08); }
    .meta-list { display: grid; gap: 8px; margin-top: 10px; font-size: 14px; }
    @media (max-width: 900px) { .app { grid-template-columns: 1fr; } .cell { width: 44px; height: 44px; font-size: 16px; } .legend { grid-template-columns: repeat(2, minmax(120px, 1fr)); } }
  </style>
</head>
<body>
  <div class="app">
    <aside class="panel">
      <h1>Map Editor</h1>
      <div class="sub">Browser GUI for advanced curriculum maps</div>
      <div class="toolbar">
        <h2>Level</h2>
        <select id="levelSelect"></select>
        <h2 style="margin-top:16px;">Tools</h2>
        <div class="tool-grid" id="toolGrid"></div>
        <h2 style="margin-top:16px;">Actions</h2>
        <div class="button-grid">
          <button class="secondary" id="saveBtn">Save YAML</button>
          <button class="secondary" id="resetBtn">Reset Level</button>
          <button class="secondary" id="reloadBtn">Reload File</button>
          <button class="secondary" id="refreshBtn">Refresh View</button>
        </div>
        <div class="status" id="status"></div>
      </div>
    </aside>

    <main class="panel">
      <div class="meta">
        <h2>Grid</h2>
        <div class="meta-list" id="metaList"></div>
      </div>
      <div class="board-wrap">
        <div class="board" id="board"></div>
      </div>
      <div class="legend">
        <div class="legend-item"><span class="swatch" style="background:var(--free)"></span>Free</div>
        <div class="legend-item"><span class="swatch" style="background:var(--wall)"></span>Wall</div>
        <div class="legend-item"><span class="swatch" style="background:var(--start)"></span>Start</div>
        <div class="legend-item"><span class="swatch" style="background:var(--goal)"></span>Goal</div>
        <div class="legend-item"><span class="swatch" style="background:var(--key)"></span>Key</div>
        <div class="legend-item"><span class="swatch" style="background:var(--door)"></span>Door</div>
        <div class="legend-item"><span class="swatch" style="background:var(--hazard)"></span>Hazard</div>
        <div class="legend-item"><span class="swatch" style="background:var(--bonus)"></span>Bonus</div>
      </div>
    </main>
  </div>

  <script>
    const TOOL_LABELS = {
      toggle_obstacle: "Wall",
      key: "Key",
      door: "Door",
      hazard: "Hazard",
      bonus: "Bonus",
      erase_special: "Erase"
    };
    const CELL_COLORS = {
      ".": "var(--free)",
      "#": "var(--wall)",
      "S": "var(--start)",
      "G": "var(--goal)",
      "K": "var(--key)",
      "D": "var(--door)",
      "H": "var(--hazard)",
      "B": "var(--bonus)"
    };
    let state = null;

    async function api(path, method = "GET", body = null) {
      const res = await fetch(path, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : null
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      return await res.json();
    }

    function setStatus(text) {
      document.getElementById("status").textContent = text || "";
    }

    function renderTools() {
      const toolGrid = document.getElementById("toolGrid");
      toolGrid.innerHTML = "";
      state.tool_order.forEach(tool => {
        const btn = document.createElement("button");
        btn.textContent = TOOL_LABELS[tool];
        btn.className = tool === state.current_tool ? "active" : "";
        btn.onclick = async () => {
          state = await api("/api/set-tool", "POST", { tool });
          renderAll();
        };
        toolGrid.appendChild(btn);
      });
    }

    function renderLevelSelect() {
      const sel = document.getElementById("levelSelect");
      sel.innerHTML = "";
      state.level_names.forEach((name, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = name;
        if (idx === state.current_level_idx) opt.selected = true;
        sel.appendChild(opt);
      });
      sel.onchange = async e => {
        state = await api("/api/set-level", "POST", { index: Number(e.target.value) });
        renderAll();
      };
    }

    function renderMeta() {
      const meta = document.getElementById("metaList");
      meta.innerHTML = `
        <div><strong>Config:</strong> ${state.config_path}</div>
        <div><strong>Level:</strong> ${state.current_level_name}</div>
        <div><strong>Tool:</strong> ${TOOL_LABELS[state.current_tool]}</div>
        <div><strong>Grid:</strong> ${state.rows} x ${state.cols}</div>
        <div><strong>Specials:</strong> key, door, hazard, and bonus all support multiple cells</div>
      `;
    }

    function renderBoard() {
      const board = document.getElementById("board");
      board.style.gridTemplateColumns = `repeat(${state.cols}, 56px)`;
      board.innerHTML = "";
      state.grid.forEach((row, r) => {
        row.forEach((label, c) => {
          const btn = document.createElement("button");
          btn.className = "cell";
          btn.style.background = CELL_COLORS[label];
          btn.style.color = label === "G" ? "#111827" : "white";
          btn.textContent = label === "." || label === "#" ? "" : label;
          btn.title = `(${r}, ${c})`;
          btn.onclick = async () => {
            state = await api("/api/apply", "POST", { row: r, col: c });
            renderAll();
          };
          board.appendChild(btn);
        });
      });
    }

    function renderAll() {
      renderLevelSelect();
      renderTools();
      renderMeta();
      renderBoard();
      setStatus(state.status);
    }

    async function refreshState() {
      state = await api("/api/state");
      renderAll();
    }

    async function bindActions() {
      document.getElementById("saveBtn").onclick = async () => {
        state = await api("/api/save", "POST");
        renderAll();
      };
      document.getElementById("resetBtn").onclick = async () => {
        state = await api("/api/reset-level", "POST");
        renderAll();
      };
      document.getElementById("reloadBtn").onclick = async () => {
        state = await api("/api/reload", "POST");
        renderAll();
      };
      document.getElementById("refreshBtn").onclick = refreshState;
    }

    bindActions().then(refreshState).catch(err => setStatus(err.message));
  </script>
</body>
</html>
"""


def serialize_state(model: MapModel) -> Dict[str, Any]:
    grid = [
        [model.cell_label((row, col)) for col in range(model.cols)]
        for row in range(model.rows)
    ]
    return {
        "config_path": str(model.config_path),
        "rows": model.rows,
        "cols": model.cols,
        "start": list(model.start),
        "goal": list(model.goal),
        "level_names": model.level_names,
        "current_level_idx": model.current_level_idx,
        "current_level_name": model.current_level_name,
        "current_tool": model.current_tool,
        "tool_order": TOOL_ORDER,
        "status": model.status,
        "grid": grid,
    }


class EditorHandler(BaseHTTPRequestHandler):
    model: MapModel

    def _json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return
        if parsed.path == "/api/state":
            self._send_json(serialize_state(self.model))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        body = self._json_body()

        try:
            if parsed.path == "/api/set-tool":
                self.model.set_tool(str(body["tool"]))
            elif parsed.path == "/api/set-level":
                index = int(body["index"])
                self.model.current_level_idx = index
                self.model.status = f"Switched to {self.model.current_level_name}"
            elif parsed.path == "/api/apply":
                self.model.apply_tool((int(body["row"]), int(body["col"])))
            elif parsed.path == "/api/save":
                self.model.save()
            elif parsed.path == "/api/reset-level":
                self.model.reset_level()
            elif parsed.path == "/api/reload":
                self.model.reload()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json(serialize_state(self.model))

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run browser GUI for advanced curriculum map editing")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/experiment4.yaml",
        help="Path to advanced curriculum YAML config",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    args = parser.parse_args()

    model = MapModel(Path(args.config).resolve())
    model.status = "Web editor ready"

    EditorHandler.model = model
    server = ThreadingHTTPServer((args.host, args.port), EditorHandler)
    print(f"Map editor running at http://{args.host}:{args.port}")
    print("Open this URL in your browser. Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
