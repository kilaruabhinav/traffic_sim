"""Professional Tkinter visualization for trained DQN and PPO controllers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk
from typing import Any

import numpy as np
from stable_baselines3 import DQN, PPO

from traffic_sim.environment import TrafficEnv


CANVAS_SIZE = 740
CENTER = CANVAS_SIZE // 2
INTERSECTION_HALF = 82
QUEUE_SPACING = 27
VISIBLE_QUEUE_LIMIT = 10

COLORS = {
    "window": "#08111f",
    "panel": "#101a2a",
    "card": "#162235",
    "card_border": "#26364d",
    "ground": "#102a27",
    "road": "#303846",
    "road_edge": "#556070",
    "intersection": "#394353",
    "muted_marking": "#8f9aaa",
    "text": "#edf2f7",
    "muted": "#9eabbc",
    "blue": "#4da3ff",
    "green": "#35d07f",
    "red": "#ff5d67",
    "yellow": "#ffc857",
}

CAR_COLORS = ("#5dade2", "#ec8d5b", "#a995e8", "#e8c75d", "#65c9a7", "#d87d9d")


@dataclass
class QueuedVehicle:
    """Visual-only identity whose position eases toward a queue slot."""

    lane: int
    color: str
    x: float
    y: float
    target_x: float
    target_y: float


@dataclass
class MovingVehicle:
    """Visual-only identity traveling through and away from the intersection."""

    lane: int
    color: str
    start_x: float
    start_y: float
    departure_offset: int = 0
    progress: float = 0.0
    source: QueuedVehicle | None = None


def load_controller_model(algorithm: str, model_path: Path) -> DQN | PPO:
    """Load a controller with the matching Stable-Baselines3 algorithm class."""
    if algorithm == "dqn":
        return DQN.load(str(model_path))
    if algorithm == "ppo":
        return PPO.load(str(model_path))
    raise ValueError(f"unsupported algorithm: {algorithm}")


class TrafficVisualizer:
    """Render DQN-controlled environment events without changing simulation state."""

    def __init__(
        self,
        root: tk.Tk,
        model_path: Path,
        algorithm: str = "dqn",
    ) -> None:
        self.root = root
        self.algorithm = algorithm.lower()
        self.algorithm_label = self.algorithm.upper()
        self.root.title(f"RL Traffic Control — {self.algorithm_label}")
        self.root.geometry("1240x790")
        self.root.minsize(1160, 760)
        self.root.configure(background=COLORS["window"])

        self.env = TrafficEnv()
        self.model = load_controller_model(self.algorithm, model_path)
        self.observation = np.zeros(15, dtype=np.float32)
        self.info: dict[str, Any] = {}
        self.current_reward = 0.0
        self.reward_sum = 0.0
        self.selected_action: int | None = None
        self.episode_done = False
        self.running = False
        self.step_job: str | None = None
        self.visual_queues: list[list[QueuedVehicle]] = [[] for _ in range(4)]
        self.moving_vehicles: list[MovingVehicle] = []
        self.next_vehicle_color = 0
        self.decision_flash_until = 0.0
        self.last_frame_time = time.monotonic()

        self.speed_var = tk.DoubleVar(value=3.0)
        self.speed_text_var = tk.StringVar()
        self.status_var = tk.StringVar(value="PAUSED")
        self.simulation_var = tk.StringVar()
        self.signal_var = tk.StringVar()
        self.decision_var = tk.StringVar()
        self.performance_var = tk.StringVar()
        self.progress_text_var = tk.StringVar()

        self._configure_style()
        self._build_ui()
        self.reset_episode()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(16, self._animate)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=COLORS["window"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Card.TFrame", background=COLORS["card"])
        style.configure(
            "Card.TLabelframe",
            background=COLORS["card"],
            bordercolor=COLORS["card_border"],
            relief="solid",
            borderwidth=1,
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLORS["card"],
            foreground=COLORS["muted"],
            font=("Helvetica", 10, "bold"),
        )
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["text"])
        style.configure(
            "Muted.Card.TLabel", background=COLORS["card"], foreground=COLORS["muted"]
        )
        style.configure(
            "Title.Panel.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["text"],
            font=("Helvetica", 20, "bold"),
        )
        style.configure(
            "Status.Panel.TLabel",
            background=COLORS["panel"],
            foreground=COLORS["muted"],
            font=("Helvetica", 10, "bold"),
        )
        style.configure(
            "Decision.Card.TLabel",
            background=COLORS["card"],
            foreground=COLORS["blue"],
            font=("Helvetica", 12, "bold"),
        )
        style.configure("Control.TButton", font=("Helvetica", 10, "bold"), padding=(10, 7))
        style.configure(
            "Episode.Horizontal.TProgressbar",
            troughcolor="#263348",
            background=COLORS["blue"],
            bordercolor="#263348",
            lightcolor=COLORS["blue"],
            darkcolor=COLORS["blue"],
        )

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=12)
        shell.pack(fill=tk.BOTH, expand=True)
        shell.columnconfigure(0, weight=3)
        shell.columnconfigure(1, weight=2)
        shell.rowconfigure(0, weight=1)

        canvas_shell = tk.Frame(shell, background="#1e2a3a", padx=1, pady=1)
        canvas_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        self.canvas = tk.Canvas(
            canvas_shell,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            background=COLORS["ground"],
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        panel = ttk.Frame(shell, style="Panel.TFrame", padding=14)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="RL TRAFFIC CONTROL", style="Title.Panel.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(panel, textvariable=self.status_var, style="Status.Panel.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 10)
        )

        controls = ttk.Frame(panel, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        for column in range(4):
            controls.columnconfigure(column, weight=1)
        self.start_button = ttk.Button(controls, text="START", style="Control.TButton", command=self.start)
        self.pause_button = ttk.Button(controls, text="PAUSE", style="Control.TButton", command=self.pause)
        self.step_button = ttk.Button(controls, text="STEP", style="Control.TButton", command=self.step_once)
        self.reset_button = ttk.Button(controls, text="RESET", style="Control.TButton", command=self.reset_episode)
        for column, button in enumerate(
            (self.start_button, self.pause_button, self.step_button, self.reset_button)
        ):
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 3, 0))

        settings = ttk.LabelFrame(panel, text=" SIMULATION ", style="Card.TLabelframe", padding=10)
        settings.grid(row=3, column=0, sticky="ew", pady=(10, 7))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, textvariable=self.simulation_var, style="Card.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(settings, text="Episode Progress", style="Muted.Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(7, 2)
        )
        ttk.Label(settings, textvariable=self.progress_text_var, style="Muted.Card.TLabel").grid(
            row=1, column=1, sticky="e", pady=(7, 2)
        )
        self.progress = ttk.Progressbar(
            settings,
            maximum=self.env.config.episode_length,
            style="Episode.Horizontal.TProgressbar",
        )
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew")

        speed_row = ttk.Frame(settings, style="Card.TFrame")
        speed_row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        speed_row.columnconfigure(1, weight=1)
        ttk.Label(speed_row, text="Slow", style="Muted.Card.TLabel").grid(row=0, column=0)
        self.speed_scale = ttk.Scale(
            speed_row,
            from_=0.5,
            to=12.0,
            variable=self.speed_var,
            command=self._speed_changed,
            orient=tk.HORIZONTAL,
        )
        self.speed_scale.grid(row=0, column=1, sticky="ew", padx=7)
        ttk.Label(speed_row, text="Fast", style="Muted.Card.TLabel").grid(row=0, column=2)
        ttk.Label(speed_row, textvariable=self.speed_text_var, style="Card.TLabel").grid(
            row=1, column=0, columnspan=3, pady=(3, 0)
        )

        middle = ttk.Frame(panel, style="Panel.TFrame")
        middle.grid(row=4, column=0, sticky="ew")
        middle.columnconfigure(0, weight=1)
        middle.columnconfigure(1, weight=1)

        signal_card = ttk.LabelFrame(middle, text=" SIGNAL ", style="Card.TLabelframe", padding=10)
        signal_card.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        ttk.Label(signal_card, textvariable=self.signal_var, style="Card.TLabel").pack(anchor="w")

        decision_card = ttk.LabelFrame(
            middle, text=" MODEL REQUEST ", style="Card.TLabelframe", padding=10
        )
        decision_card.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.decision_label = ttk.Label(
            decision_card, textvariable=self.decision_var, style="Decision.Card.TLabel"
        )
        self.decision_label.pack(anchor="w")

        traffic_card = ttk.LabelFrame(panel, text=" TRAFFIC ", style="Card.TLabelframe", padding=8)
        traffic_card.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        self.queue_canvas = tk.Canvas(
            traffic_card, height=116, background=COLORS["card"], highlightthickness=0
        )
        self.queue_canvas.pack(fill=tk.X, expand=True)

        performance_card = ttk.LabelFrame(
            panel, text=" PERFORMANCE ", style="Card.TLabelframe", padding=10
        )
        performance_card.grid(row=6, column=0, sticky="ew", pady=(7, 0))
        ttk.Label(
            performance_card,
            textvariable=self.performance_var,
            style="Card.TLabel",
            font=("Menlo", 10),
        ).pack(anchor="w")

        legend = ttk.LabelFrame(panel, text=" LEGEND ", style="Card.TLabelframe", padding=8)
        legend.grid(row=7, column=0, sticky="ew", pady=(7, 0))
        ttk.Label(
            legend,
            text="● Green active   ● Red stopped   ● Yellow clearance\n"
            "▣ Blue outline model request   ▰ queued   → departing",
            style="Muted.Card.TLabel",
            font=("Helvetica", 9),
        ).pack(anchor="w")
        self._speed_changed()

    def reset_episode(self) -> None:
        """Reset simulation metrics and every visual-only animation state."""
        self.pause()
        self.observation, self.info = self.env.reset()
        self.current_reward = 0.0
        self.reward_sum = 0.0
        self.selected_action = None
        self.episode_done = False
        self.visual_queues = [[] for _ in range(4)]
        self.moving_vehicles.clear()
        self.next_vehicle_color = 0
        self.decision_flash_until = 0.0
        self.status_var.set("PAUSED  •  FRESH TRAFFIC")
        self._update_controls()
        self._update_dashboard()
        self._render()

    def start(self) -> None:
        if self.episode_done or self.running:
            return
        self.running = True
        self.status_var.set(f"RUNNING  •  {self.algorithm_label} CONTROL")
        self._update_controls()
        self._schedule_step(0)

    def pause(self) -> None:
        self.running = False
        if self.step_job is not None:
            self.root.after_cancel(self.step_job)
            self.step_job = None
        if not self.episode_done and self.info:
            self.status_var.set("PAUSED")
        if hasattr(self, "start_button"):
            self._update_controls()

    def step_once(self) -> None:
        self.pause()
        self._perform_step()

    def _update_controls(self) -> None:
        if self.episode_done:
            self.start_button.state(["disabled"])
            self.pause_button.state(["disabled"])
            self.step_button.state(["disabled"])
        elif self.running:
            self.start_button.state(["disabled"])
            self.pause_button.state(["!disabled"])
            self.step_button.state(["disabled"])
        else:
            self.start_button.state(["!disabled"])
            self.pause_button.state(["disabled"])
            self.step_button.state(["!disabled"])
        self.reset_button.state(["!disabled"])

    def _speed_changed(self, _value: str | None = None) -> None:
        self.speed_text_var.set(f"{self.speed_var.get():.1f} steps / second")

    def _schedule_step(self, delay: int | None = None) -> None:
        milliseconds = delay if delay is not None else int(1000 / max(self.speed_var.get(), 0.1))
        self.step_job = self.root.after(milliseconds, self._continuous_step)

    def _continuous_step(self) -> None:
        self.step_job = None
        if not self.running:
            return
        self._perform_step()
        if self.running and not self.episode_done:
            self._schedule_step()

    def _perform_step(self) -> None:
        if self.episode_done:
            return
        action, _ = self.model.predict(self.observation, deterministic=True)
        self.selected_action = int(action)
        observation, reward, terminated, truncated, info = self.env.step(self.selected_action)
        self._apply_visual_events(info)
        self.observation = observation
        self.current_reward = float(reward)
        self.reward_sum += self.current_reward
        self.info = info
        self.decision_flash_until = time.monotonic() + 0.45
        self.episode_done = terminated or truncated
        if self.episode_done:
            self.running = False
            self.status_var.set("EPISODE COMPLETE")
        elif not self.running:
            self.status_var.set("PAUSED")
        self._update_controls()
        self._update_dashboard()
        self._render()

    def _new_car_color(self) -> str:
        color = CAR_COLORS[self.next_vehicle_color % len(CAR_COLORS)]
        self.next_vehicle_color += 1
        return color

    @staticmethod
    def _spawn_position(lane: int) -> tuple[float, float]:
        return (
            (CENTER - 32, -18),
            (CANVAS_SIZE + 18, CENTER - 32),
            (CENTER + 32, CANVAS_SIZE + 18),
            (-18, CENTER + 32),
        )[lane]

    @staticmethod
    def _front_position(lane: int) -> tuple[float, float]:
        return (
            (CENTER - 32, CENTER - INTERSECTION_HALF - 16),
            (CENTER + INTERSECTION_HALF + 16, CENTER - 32),
            (CENTER + 32, CENTER + INTERSECTION_HALF + 16),
            (CENTER - INTERSECTION_HALF - 16, CENTER + 32),
        )[lane]

    def _queue_target(self, lane: int, index: int) -> tuple[float, float]:
        front_x, front_y = self._front_position(lane)
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[lane]
        return front_x + dx * index * QUEUE_SPACING, front_y + dy * index * QUEUE_SPACING

    def _make_queued_vehicle(self, lane: int) -> QueuedVehicle:
        spawn_x, spawn_y = self._spawn_position(lane)
        return QueuedVehicle(lane, self._new_car_color(), spawn_x, spawn_y, spawn_x, spawn_y)

    def _retarget_queues(self) -> None:
        for lane, vehicles in enumerate(self.visual_queues):
            for index, vehicle in enumerate(self._waiting_vehicles(lane)):
                vehicle.target_x, vehicle.target_y = self._queue_target(lane, index)

    def _waiting_vehicles(self, lane: int) -> list[QueuedVehicle]:
        moving_source_ids = {
            id(vehicle.source)
            for vehicle in self.moving_vehicles
            if vehicle.source is not None
        }
        return [
            vehicle
            for vehicle in self.visual_queues[lane]
            if id(vehicle) not in moving_source_ids
        ]

    def _apply_visual_events(self, info: dict[str, Any]) -> None:
        for lane, arrivals in enumerate(info["arrivals"]):
            for _ in range(int(arrivals)):
                self.visual_queues[lane].append(self._make_queued_vehicle(lane))
        departure_lane = int(info["current_green_lane"])
        available_departures = self._waiting_vehicles(departure_lane)
        for departure_index in range(int(info["vehicles_passed"])):
            source = available_departures.pop(0) if available_departures else None
            color = source.color if source is not None else self._new_car_color()
            # Every reported departure becomes its own visible identity. Parallel
            # offsets keep simultaneous vehicles from overlapping into one shape.
            start_x, start_y = self._queue_target(departure_lane, departure_index)
            self.moving_vehicles.append(
                MovingVehicle(
                    departure_lane,
                    color,
                    start_x,
                    start_y,
                    departure_offset=departure_index,
                    source=source,
                )
            )
        for lane, expected in enumerate(info["queue_lengths"]):
            target = int(expected)
            waiting = self._waiting_vehicles(lane)
            while len(waiting) < target:
                self.visual_queues[lane].append(self._make_queued_vehicle(lane))
                waiting = self._waiting_vehicles(lane)
            for extra in waiting[target:]:
                self.visual_queues[lane].remove(extra)
        self._retarget_queues()

    def _update_dashboard(self) -> None:
        step = int(self.info["elapsed_steps"])
        self.simulation_var.set(
            f"Step  {step:3d} / {self.env.config.episode_length}     "
            f"Speed  {self.speed_var.get():.1f}×"
        )
        self.progress["value"] = step
        self.progress_text_var.set(f"{step} / {self.env.config.episode_length}")
        green = int(self.info["current_green_lane"]) + 1
        if self.info["is_switching"]:
            pending = int(self.info["pending_lane"]) + 1
            self.signal_var.set(
                f"Current Green   LANE {green}\n"
                f"Green Time      {self.info['green_time']} / {self.env.config.max_green_time}\n"
                f"Status          CLEARANCE\nSwitching       → LANE {pending}"
            )
        else:
            self.signal_var.set(
                f"Current Green   LANE {green}\n"
                f"Green Time      {self.info['green_time']} / {self.env.config.max_green_time}\n"
                "Status          GREEN\nSwitching       —"
            )
        if self.selected_action is None:
            self.decision_var.set("Awaiting first step\nAction    —\nActive    LANE 1\n")
        else:
            selected = self.selected_action + 1
            relation = "Matches active signal" if selected == green else "Timing transition applies"
            self.decision_var.set(
                f"Selected  LANE {selected}\nAction    {self.selected_action}\n"
                f"Active    LANE {green}\n{relation}"
            )
        self.performance_var.set(
            f"Vehicles Passed       {self.info['cumulative_vehicles_passed']:4d}\n"
            f"Current Reward        {self.current_reward:+7.3f}\n"
            f"Violations Now        {self.info['queue_limit_violations']:4d}\n"
            f"Total Violations      {self.info['cumulative_queue_limit_violations']:4d}"
        )
        self._draw_queue_bars()

    def _draw_queue_bars(self) -> None:
        canvas = self.queue_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 320)
        queues = [int(value) for value in self.info["queue_lengths"]]
        hard_limit = self.env.config.max_queue_lengths[0]
        for lane, queue in enumerate(queues):
            y = 15 + lane * 27
            canvas.create_text(5, y, text=f"Lane {lane + 1}", anchor="w", fill=COLORS["text"], font=("Menlo", 10))
            x1, x2 = 78, width - 42
            canvas.create_rectangle(x1, y - 7, x2, y + 7, fill="#263348", outline="")
            ratio = min(queue / hard_limit, 1.0)
            color = COLORS["blue"] if self.selected_action == lane else "#718096"
            canvas.create_rectangle(x1, y - 7, x1 + (x2 - x1) * ratio, y + 7, fill=color, outline="")
            canvas.create_text(width - 5, y, text=str(queue), anchor="e", fill=COLORS["text"], font=("Menlo", 10, "bold"))

    def _animate(self) -> None:
        now = time.monotonic()
        elapsed = min(now - self.last_frame_time, 0.1)
        self.last_frame_time = now
        ease = 1.0 - pow(0.02, elapsed)
        for vehicles in self.visual_queues:
            for vehicle in vehicles:
                vehicle.x += (vehicle.target_x - vehicle.x) * ease
                vehicle.y += (vehicle.target_y - vehicle.y) * ease
        for vehicle in self.moving_vehicles:
            vehicle.progress += elapsed / 0.8
        self._remove_finished_departures()
        self._render()
        self.root.after(16, self._animate)

    def _remove_finished_departures(self) -> None:
        finished = [vehicle for vehicle in self.moving_vehicles if vehicle.progress >= 1.0]
        for vehicle in finished:
            if (
                vehicle.source is not None
                and vehicle.source in self.visual_queues[vehicle.lane]
            ):
                self.visual_queues[vehicle.lane].remove(vehicle.source)
        self.moving_vehicles = [vehicle for vehicle in self.moving_vehicles if vehicle.progress < 1.0]
        if finished:
            self._retarget_queues()

    def _render(self) -> None:
        self.canvas.delete("all")
        self._draw_roads()
        self._draw_selected_lane()
        self._draw_stop_lines_and_crosswalks()
        self._draw_queued_vehicles()
        self._draw_moving_vehicles()
        self._draw_signals()
        self._draw_lane_labels()
        if self.episode_done:
            self._draw_episode_complete()

    def _draw_roads(self) -> None:
        c = self.canvas
        c.create_rectangle(0, 0, CANVAS_SIZE, CANVAS_SIZE, fill=COLORS["ground"], outline="")
        road_left, road_right = CENTER - 90, CENTER + 90
        c.create_rectangle(road_left, 0, road_right, CANVAS_SIZE, fill=COLORS["road"], outline=COLORS["road_edge"], width=2)
        c.create_rectangle(0, road_left, CANVAS_SIZE, road_right, fill=COLORS["road"], outline=COLORS["road_edge"], width=2)
        c.create_rectangle(
            CENTER - INTERSECTION_HALF, CENTER - INTERSECTION_HALF,
            CENTER + INTERSECTION_HALF, CENTER + INTERSECTION_HALF,
            fill=COLORS["intersection"], outline="#667386", width=2,
        )
        for coordinates in (
            (CENTER, 0, CENTER, CENTER - INTERSECTION_HALF),
            (CENTER, CENTER + INTERSECTION_HALF, CENTER, CANVAS_SIZE),
            (0, CENTER, CENTER - INTERSECTION_HALF, CENTER),
            (CENTER + INTERSECTION_HALF, CENTER, CANVAS_SIZE, CENTER),
        ):
            c.create_line(*coordinates, fill=COLORS["muted_marking"], dash=(12, 12), width=2)
        c.create_rectangle(CENTER - 62, CENTER - 62, CENTER + 62, CENTER + 62, outline="#7b8797", dash=(5, 5))

    def _draw_selected_lane(self) -> None:
        if self.selected_action is None:
            return
        flash = max(0.0, self.decision_flash_until - time.monotonic()) / 0.45
        rectangles = (
            (CENTER - 88, 2, CENTER - 1, CENTER - INTERSECTION_HALF),
            (CENTER + INTERSECTION_HALF, CENTER - 88, CANVAS_SIZE - 2, CENTER - 1),
            (CENTER + 1, CENTER + INTERSECTION_HALF, CENTER + 88, CANVAS_SIZE - 2),
            (2, CENTER + 1, CENTER - INTERSECTION_HALF, CENTER + 88),
        )
        self.canvas.create_rectangle(*rectangles[self.selected_action], outline=COLORS["blue"], width=3 + int(3 * flash))
        if flash > 0:
            x, y = ((318, 92), (648, 318), (422, 648), (92, 422))[self.selected_action]
            self.canvas.create_text(x, y, text=f"AI DECISION\nLANE {self.selected_action + 1}", fill=COLORS["blue"], font=("Helvetica", 11, "bold"), justify="center")

    def _draw_stop_lines_and_crosswalks(self) -> None:
        c, h = self.canvas, INTERSECTION_HALF
        c.create_line(CENTER - 88, CENTER - h - 4, CENTER - 2, CENTER - h - 4, fill="white", width=5)
        c.create_line(CENTER + h + 4, CENTER - 88, CENTER + h + 4, CENTER - 2, fill="white", width=5)
        c.create_line(CENTER + 2, CENTER + h + 4, CENTER + 88, CENTER + h + 4, fill="white", width=5)
        c.create_line(CENTER - h - 4, CENTER + 2, CENTER - h - 4, CENTER + 88, fill="white", width=5)
        for offset in range(-70, 71, 20):
            c.create_line(CENTER + offset, CENTER - h + 7, CENTER + offset + 10, CENTER - h + 7, fill="#b8c0cc", width=3)
            c.create_line(CENTER + offset, CENTER + h - 7, CENTER + offset + 10, CENTER + h - 7, fill="#b8c0cc", width=3)
            c.create_line(CENTER - h + 7, CENTER + offset, CENTER - h + 7, CENTER + offset + 10, fill="#b8c0cc", width=3)
            c.create_line(CENTER + h - 7, CENTER + offset, CENTER + h - 7, CENTER + offset + 10, fill="#b8c0cc", width=3)

    def _draw_vehicle(self, lane: int, x: float, y: float, color: str) -> None:
        c = self.canvas
        if lane in (0, 2):
            c.create_rectangle(x - 7, y - 11, x + 7, y + 11, fill=color, outline="#07111d", width=2)
            windshield_y = y - 5 if lane == 0 else y + 5
            c.create_rectangle(x - 5, windshield_y - 2, x + 5, windshield_y + 2, fill="#b9dcf1", outline="")
            c.create_line(x - 8, y - 6, x - 8, y + 6, fill="#111827", width=2)
            c.create_line(x + 8, y - 6, x + 8, y + 6, fill="#111827", width=2)
        else:
            c.create_rectangle(x - 11, y - 7, x + 11, y + 7, fill=color, outline="#07111d", width=2)
            windshield_x = x - 5 if lane == 1 else x + 5
            c.create_rectangle(windshield_x - 2, y - 5, windshield_x + 2, y + 5, fill="#b9dcf1", outline="")
            c.create_line(x - 6, y - 8, x + 6, y - 8, fill="#111827", width=2)
            c.create_line(x - 6, y + 8, x + 6, y + 8, fill="#111827", width=2)

    def _draw_queued_vehicles(self) -> None:
        for lane in range(4):
            vehicles = self._waiting_vehicles(lane)
            for vehicle in vehicles[:VISIBLE_QUEUE_LIMIT]:
                self._draw_vehicle(lane, vehicle.x, vehicle.y, vehicle.color)
            hidden = len(vehicles) - VISIBLE_QUEUE_LIMIT
            if hidden > 0:
                x, y = self._queue_target(lane, VISIBLE_QUEUE_LIMIT - 1)
                self.canvas.create_oval(x - 17, y - 17, x + 17, y + 17, fill="#0d1725", outline=COLORS["yellow"], width=2)
                self.canvas.create_text(x, y, text=f"+{hidden}", fill="white", font=("Helvetica", 9, "bold"))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = max(0.0, min(value, 1.0))
        return value * value * (3.0 - 2.0 * value)

    def _moving_position(self, vehicle: MovingVehicle) -> tuple[float, float]:
        t = self._smoothstep(vehicle.progress)
        end_x, end_y = (
            (CENTER - 32, CANVAS_SIZE + 25),
            (-25, CENTER - 32),
            (CENTER + 32, -25),
            (CANVAS_SIZE + 25, CENTER + 32),
        )[vehicle.lane]
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[vehicle.lane]
        end_x += dx * vehicle.departure_offset * QUEUE_SPACING
        end_y += dy * vehicle.departure_offset * QUEUE_SPACING
        return vehicle.start_x + (end_x - vehicle.start_x) * t, vehicle.start_y + (end_y - vehicle.start_y) * t

    def _draw_moving_vehicles(self) -> None:
        for vehicle in self.moving_vehicles:
            x, y = self._moving_position(vehicle)
            self._draw_vehicle(vehicle.lane, x, y, vehicle.color)

    def _draw_signals(self) -> None:
        positions = ((278, 257), (483, 278), (462, 483), (257, 462))
        current = int(self.info["current_green_lane"])
        switching = bool(self.info["is_switching"])
        dim = {"red": "#52252b", "yellow": "#564927", "green": "#1d5037"}
        for lane, (x, y) in enumerate(positions):
            state = "yellow" if lane == current and switching else "green" if lane == current else "red"
            self.canvas.create_rectangle(x - 14, y - 31, x + 14, y + 31, fill="#0a0f16", outline="#778191", width=2)
            for offset, bulb in ((-19, "red"), (0, "yellow"), (19, "green")):
                fill = COLORS[bulb] if bulb == state else dim[bulb]
                self.canvas.create_oval(x - 7, y + offset - 7, x + 7, y + offset + 7, fill=fill, outline="")
        if switching:
            pending = int(self.info["pending_lane"]) + 1
            self.canvas.create_rectangle(CENTER - 92, CENTER - 25, CENTER + 92, CENTER + 25, fill="#111827", outline=COLORS["yellow"], width=2)
            self.canvas.create_text(CENTER, CENTER - 7, text="CLEARANCE", fill=COLORS["yellow"], font=("Helvetica", 12, "bold"))
            self.canvas.create_text(CENTER, CENTER + 10, text=f"SWITCHING → LANE {pending}", fill="white", font=("Helvetica", 9, "bold"))

    def _draw_lane_labels(self) -> None:
        for x, y, text in (
            (CENTER - 32, 20, "LANE 1  ↓"),
            (CANVAS_SIZE - 55, CENTER - 32, "LANE 2  ←"),
            (CENTER + 32, CANVAS_SIZE - 20, "LANE 3  ↑"),
            (55, CENTER + 32, "LANE 4  →"),
        ):
            self.canvas.create_text(x, y, text=text, fill="white", font=("Helvetica", 11, "bold"))

    def _draw_episode_complete(self) -> None:
        step = max(int(self.info["elapsed_steps"]), 1)
        mean_reward = self.reward_sum / step
        self.canvas.create_rectangle(145, 190, 595, 550, fill="#0b1422", outline=COLORS["blue"], width=3)
        self.canvas.create_text(CENTER, 235, text="EPISODE COMPLETE", fill="white", font=("Helvetica", 24, "bold"))
        self.canvas.create_text(
            CENTER, 355,
            text=(
                f"Vehicles Passed              {self.info['cumulative_vehicles_passed']}\n"
                f"Mean Step Reward             {mean_reward:+.3f}\n"
                f"Maximum Queue                {self.info['episode_max_queue']}\n"
                f"Total Violations             {self.info['cumulative_queue_limit_violations']}"
            ),
            fill=COLORS["text"], font=("Menlo", 12), justify="left",
        )
        self.canvas.create_rectangle(285, 470, 455, 515, fill=COLORS["blue"], outline="", tags="overlay_reset")
        self.canvas.create_text(CENTER, 492, text="RESET", fill="#06101d", font=("Helvetica", 12, "bold"), tags="overlay_reset")
        self.canvas.tag_bind("overlay_reset", "<Button-1>", lambda _event: self.reset_episode())

    def close(self) -> None:
        self.pause()
        self.env.close()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/dqn_traffic_bounded"))
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), default="dqn")
    args = parser.parse_args()
    root = tk.Tk()
    TrafficVisualizer(root, args.model_path, args.algorithm)
    root.mainloop()


if __name__ == "__main__":
    main()
