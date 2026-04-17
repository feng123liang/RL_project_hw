from __future__ import annotations

"""Map editor for advanced curriculum configs."""

import argparse
import copy
import curses
import os
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


Position = Tuple[int, int]

CELL_SIZE = 48
GRID_LINE_COLOR = "#CBD5E1"
FREE_COLOR = "#F8FAFC"
OBSTACLE_COLOR = "#334155"
START_COLOR = "#2563EB"
GOAL_COLOR = "#F59E0B"
KEY_COLOR = "#10B981"
DOOR_COLOR = "#A16207"
HAZARD_COLOR = "#EF4444"
BONUS_COLOR = "#F97316"

PALETTE = [
    ("toggle_obstacle", "Wall"),
    ("key", "Key"),
    ("door", "Door"),
    ("hazard", "Hazard"),
    ("bonus", "Bonus"),
    ("erase_special", "Erase"),
]
TOOL_ORDER = [tool_id for tool_id, _ in PALETTE]
TOOL_LABELS = {tool_id: label for tool_id, label in PALETTE}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


class MapModel:
    """Shared editing model used by GUI, TUI, and web modes."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.cfg = load_yaml(config_path)
        self.original_cfg = copy.deepcopy(self.cfg)

        shared = self.cfg["shared_env"]
        self.rows = int(shared["rows"])
        self.cols = int(shared["cols"])
        self.start = tuple(shared["start"])
        self.goal = tuple(shared["goal"])
        self.level_names = list(self.cfg["levels"].keys())
        self.current_level_idx = 0
        self.current_tool = TOOL_ORDER[0]
        self.status = ""

        self._normalize_all_levels()

    @property
    def current_level_name(self) -> str:
        return self.level_names[self.current_level_idx]

    def current_level_data(self) -> Dict:
        return self.cfg["levels"][self.current_level_name]

    def _normalize_multi_field(self, level: Dict, singular_key: str, plural_key: str) -> None:
        if plural_key in level:
            level[plural_key] = [list(cell) for cell in sorted({tuple(cell) for cell in level.get(plural_key, [])})]
            return
        value = level.get(singular_key)
        if value is None:
            level[plural_key] = []
        else:
            level[plural_key] = [[int(value[0]), int(value[1])]]
        level.pop(singular_key, None)

    def _normalize_all_levels(self) -> None:
        for level_name in self.level_names:
            self._ensure_obstacle_layout(level_name)
            level = self.cfg["levels"][level_name]
            self._normalize_multi_field(level, "key_pos", "key_cells")
            self._normalize_multi_field(level, "bonus_pos", "bonus_cells")
            level.setdefault("door_cells", [])
            level.setdefault("hazard_cells", [])
            level.pop("recharge_pos", None)

    def _ensure_obstacle_layout(self, level_name: str) -> None:
        level = self.cfg["levels"][level_name]
        all_cells = {(r, c) for r in range(self.rows) for c in range(self.cols)}

        if "free_cells" in level:
            free_cells = {tuple(cell) for cell in level["free_cells"]}
            obstacles = sorted(all_cells - free_cells)
            level["obstacles"] = [list(cell) for cell in obstacles]
            level.pop("free_cells", None)
        else:
            obstacles = {tuple(cell) for cell in level.get("obstacles", [])}
            level["obstacles"] = [list(cell) for cell in sorted(obstacles)]

    def set_level_by_delta(self, delta: int) -> None:
        self.current_level_idx = (self.current_level_idx + delta) % len(self.level_names)
        self.status = f"Switched to {self.current_level_name}"

    def cycle_tool(self, delta: int) -> None:
        idx = TOOL_ORDER.index(self.current_tool)
        self.current_tool = TOOL_ORDER[(idx + delta) % len(TOOL_ORDER)]
        self.status = f"Tool: {TOOL_LABELS[self.current_tool]}"

    def set_tool(self, tool: str) -> None:
        self.current_tool = tool
        self.status = f"Tool: {TOOL_LABELS[self.current_tool]}"

    def _toggle_multi_position(self, field: str, pos: Position) -> None:
        level = self.current_level_data()
        current = {tuple(cell) for cell in level.get(field, [])}
        if pos in current:
            current.remove(pos)
        else:
            current.add(pos)
        level[field] = [list(cell) for cell in sorted(current)]

    def _remove_special_at(self, pos: Position) -> None:
        level = self.current_level_data()
        for field in ("key_cells", "door_cells", "hazard_cells", "bonus_cells"):
            current = {tuple(cell) for cell in level.get(field, [])}
            if pos in current:
                current.remove(pos)
                level[field] = [list(cell) for cell in sorted(current)]

    def _toggle_obstacle(self, pos: Position) -> None:
        if pos in (self.start, self.goal):
            self.status = "Cannot turn start/goal into walls"
            return

        level = self.current_level_data()
        obstacles = {tuple(cell) for cell in level.get("obstacles", [])}
        if pos in obstacles:
            obstacles.remove(pos)
        else:
            obstacles.add(pos)
            self._remove_special_at(pos)
        level["obstacles"] = [list(cell) for cell in sorted(obstacles)]

    def apply_tool(self, pos: Position) -> None:
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return

        level = self.current_level_data()
        tool = self.current_tool

        if tool == "toggle_obstacle":
            self._toggle_obstacle(pos)
        else:
            obstacles = {tuple(cell) for cell in level.get("obstacles", [])}
            if pos in obstacles or pos in (self.start, self.goal):
                self.status = "Cannot place special cell on wall/start/goal"
                return

            if tool == "key":
                self._toggle_multi_position("key_cells", pos)
            elif tool == "door":
                self._toggle_multi_position("door_cells", pos)
            elif tool == "hazard":
                self._toggle_multi_position("hazard_cells", pos)
            elif tool == "bonus":
                self._toggle_multi_position("bonus_cells", pos)
            elif tool == "erase_special":
                self._remove_special_at(pos)

        self.status = f"Edited {self.current_level_name} at {pos}"

    def cell_label(self, pos: Position) -> str:
        level = self.current_level_data()
        if pos == self.start:
            return "S"
        if pos == self.goal:
            return "G"
        if pos in {tuple(cell) for cell in level.get("key_cells", [])}:
            return "K"
        if pos in {tuple(cell) for cell in level.get("bonus_cells", [])}:
            return "B"
        if pos in {tuple(cell) for cell in level.get("door_cells", [])}:
            return "D"
        if pos in {tuple(cell) for cell in level.get("hazard_cells", [])}:
            return "H"
        if pos in {tuple(cell) for cell in level.get("obstacles", [])}:
            return "#"
        return "."

    def reset_level(self) -> None:
        level_name = self.current_level_name
        self.cfg["levels"][level_name] = copy.deepcopy(self.original_cfg["levels"][level_name])
        self._ensure_obstacle_layout(level_name)
        self._normalize_all_levels()
        self.status = f"Reset {level_name}"

    def reload(self) -> None:
        self.cfg = load_yaml(self.config_path)
        self.original_cfg = copy.deepcopy(self.cfg)
        self._normalize_all_levels()
        self.status = "Reloaded from disk"

    def _level_for_write(self, level_name: str) -> Dict:
        level = copy.deepcopy(self.cfg["levels"][level_name])
        obstacles = {tuple(cell) for cell in level.get("obstacles", [])}
        all_cells = {(r, c) for r in range(self.rows) for c in range(self.cols)}
        free_cells = sorted(all_cells - obstacles)
        return {
            "free_cells": [list(cell) for cell in free_cells],
            "key_cells": level.get("key_cells", []),
            "door_cells": level.get("door_cells", []),
            "hazard_cells": level.get("hazard_cells", []),
            "bonus_cells": level.get("bonus_cells", []),
        }

    def save(self) -> None:
        output_cfg = copy.deepcopy(self.cfg)
        output_cfg["levels"] = {
            level_name: self._level_for_write(level_name)
            for level_name in self.level_names
        }
        dump_yaml(self.config_path, output_cfg)
        self.original_cfg = copy.deepcopy(output_cfg)
        self.cfg = copy.deepcopy(output_cfg)
        self._normalize_all_levels()
        self.status = f"Saved to {self.config_path}"


def run_gui_editor(model: MapModel) -> None:
    import tkinter as tk
    from tkinter import messagebox, ttk

    class GuiEditor:
        def __init__(self, root: tk.Tk, model: MapModel) -> None:
            self.root = root
            self.model = model
            self.current_level = tk.StringVar(value=self.model.current_level_name)
            self.current_tool = tk.StringVar(value=self.model.current_tool)
            self.status_var = tk.StringVar(value=self.model.status)
            self._build_ui()
            self._render()

        def _build_ui(self) -> None:
            self.root.title(f"Advanced Curriculum Map Editor - {self.model.config_path.name}")

            top = ttk.Frame(self.root, padding=10)
            top.pack(fill="x")

            ttk.Label(top, text="Level").pack(side="left")
            ttk.OptionMenu(
                top,
                self.current_level,
                self.current_level.get(),
                *self.model.level_names,
                command=self._change_level,
            ).pack(side="left", padx=(8, 16))

            ttk.Label(top, text="Tool").pack(side="left")
            ttk.OptionMenu(
                top,
                self.current_tool,
                self.current_tool.get(),
                *TOOL_ORDER,
                command=self._change_tool,
            ).pack(side="left", padx=(8, 0))

            tool_help = ttk.Frame(self.root, padding=(10, 0, 10, 6))
            tool_help.pack(fill="x")
            ttk.Label(
                tool_help,
                text="Click grid to edit. Start/Goal fixed. Key, door, hazard, and bonus all support multiple cells.",
            ).pack(side="left")

            self.canvas = tk.Canvas(
                self.root,
                width=self.model.cols * CELL_SIZE,
                height=self.model.rows * CELL_SIZE,
                bg="white",
                highlightthickness=0,
            )
            self.canvas.pack(padx=10, pady=6)
            self.canvas.bind("<Button-1>", self._on_click)

            bottom = ttk.Frame(self.root, padding=10)
            bottom.pack(fill="x")
            ttk.Button(bottom, text="Save YAML", command=self._save).pack(side="left")
            ttk.Button(bottom, text="Reset Level", command=self._reset_level).pack(side="left", padx=8)
            ttk.Button(bottom, text="Reload File", command=self._reload).pack(side="left")
            ttk.Label(bottom, textvariable=self.status_var).pack(side="right")

        def _change_level(self, value: str) -> None:
            self.model.current_level_idx = self.model.level_names.index(value)
            self._sync()

        def _change_tool(self, value: str) -> None:
            self.model.set_tool(value)
            self._sync()

        def _on_click(self, event: tk.Event) -> None:
            row = int(event.y // CELL_SIZE)
            col = int(event.x // CELL_SIZE)
            self.model.apply_tool((row, col))
            self._sync()

        def _render(self) -> None:
            self.canvas.delete("all")
            for row in range(self.model.rows):
                for col in range(self.model.cols):
                    pos = (row, col)
                    label = self.model.cell_label(pos)
                    x0 = col * CELL_SIZE
                    y0 = row * CELL_SIZE
                    x1 = x0 + CELL_SIZE
                    y1 = y0 + CELL_SIZE

                    fill = FREE_COLOR
                    if label == "#":
                        fill = OBSTACLE_COLOR
                    elif label == "S":
                        fill = START_COLOR
                    elif label == "G":
                        fill = GOAL_COLOR
                    elif label == "K":
                        fill = KEY_COLOR
                    elif label == "D":
                        fill = DOOR_COLOR
                    elif label == "H":
                        fill = HAZARD_COLOR
                    elif label == "B":
                        fill = BONUS_COLOR

                    self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=GRID_LINE_COLOR, width=1)
                    if label != ".":
                        text_color = "black" if label == "G" else "white"
                        self.canvas.create_text(
                            x0 + CELL_SIZE / 2,
                            y0 + CELL_SIZE / 2,
                            text=label,
                            fill=text_color,
                            font=("Helvetica", 14, "bold"),
                        )

        def _sync(self) -> None:
            self.current_level.set(self.model.current_level_name)
            self.current_tool.set(self.model.current_tool)
            self.status_var.set(self.model.status)
            self._render()

        def _save(self) -> None:
            self.model.save()
            self._sync()
            messagebox.showinfo("Saved", self.model.status)

        def _reset_level(self) -> None:
            self.model.reset_level()
            self._sync()

        def _reload(self) -> None:
            self.model.reload()
            self._sync()

    root = tk.Tk()
    editor = GuiEditor(root, model)
    editor.status_var.set(
        f"Loaded {model.config_path.name} with levels: {', '.join(model.level_names)}"
    )
    root.mainloop()


def draw_tui(stdscr, model: MapModel, cursor: Position) -> None:
    stdscr.erase()
    stdscr.addstr(0, 0, f"Config: {model.config_path}")
    stdscr.addstr(1, 0, f"Level: {model.current_level_name}    Tool: {TOOL_LABELS[model.current_tool]}")
    stdscr.addstr(2, 0, "Arrows move  Space/Enter apply  Tab next tool  [ ] change level")
    stdscr.addstr(3, 0, "1 wall 2 key 3 door 4 hazard 5 bonus 6 erase  s save  r reset  l reload  q quit")

    top = 5
    for row in range(model.rows):
        line = []
        for col in range(model.cols):
            label = model.cell_label((row, col))
            line.append(f" {label} ")
        stdscr.addstr(top + row, 0, "".join(line))

    cursor_y = top + cursor[0]
    cursor_x = cursor[1] * 3 + 1
    stdscr.move(cursor_y, cursor_x)

    legend_top = top + model.rows + 1
    stdscr.addstr(legend_top, 0, "Legend: . free  # wall  S start  G goal  K key  D door  H hazard  B bonus")
    stdscr.addstr(legend_top + 1, 0, f"Status: {model.status}")
    stdscr.refresh()


def run_tui_editor(model: MapModel) -> None:
    def _main(stdscr) -> None:
        curses.curs_set(1)
        stdscr.keypad(True)
        cursor = [0, 0]
        model.status = f"Loaded {model.config_path.name} in terminal mode"

        keymap = {
            ord("1"): "toggle_obstacle",
            ord("2"): "key",
            ord("3"): "door",
            ord("4"): "hazard",
            ord("5"): "bonus",
            ord("6"): "erase_special",
        }

        while True:
            draw_tui(stdscr, model, (cursor[0], cursor[1]))
            ch = stdscr.getch()

            if ch in (ord("q"), 27):
                break
            if ch == curses.KEY_UP:
                cursor[0] = max(0, cursor[0] - 1)
            elif ch == curses.KEY_DOWN:
                cursor[0] = min(model.rows - 1, cursor[0] + 1)
            elif ch == curses.KEY_LEFT:
                cursor[1] = max(0, cursor[1] - 1)
            elif ch == curses.KEY_RIGHT:
                cursor[1] = min(model.cols - 1, cursor[1] + 1)
            elif ch in (ord(" "), 10, 13):
                model.apply_tool((cursor[0], cursor[1]))
            elif ch == ord("\t"):
                model.cycle_tool(1)
            elif ch == ord("["):
                model.set_level_by_delta(-1)
            elif ch == ord("]"):
                model.set_level_by_delta(1)
            elif ch == ord("s"):
                model.save()
            elif ch == ord("r"):
                model.reset_level()
            elif ch == ord("l"):
                model.reload()
            elif ch in keymap:
                model.set_tool(keymap[ch])

    curses.wrapper(_main)


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit advanced curriculum maps with GUI or terminal mode")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/experiment3.yaml",
        help="Path to advanced curriculum YAML config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("auto", "gui", "tui"),
        default="auto",
        help="Editor mode. auto uses GUI when DISPLAY is available, otherwise TUI.",
    )
    args = parser.parse_args()

    model = MapModel(Path(args.config).resolve())
    mode = args.mode
    if mode == "auto":
        mode = "gui" if os.environ.get("DISPLAY") else "tui"

    if mode == "gui":
        try:
            run_gui_editor(model)
        except Exception as exc:
            print(f"GUI mode failed ({exc}), falling back to terminal mode.")
            run_tui_editor(model)
    else:
        run_tui_editor(model)


if __name__ == "__main__":
    main()
