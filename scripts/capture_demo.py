"""Capture actual Tkinter simulator screenshots and an optional animated GIF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
import tkinter as tk

from PIL import Image, ImageGrab

from traffic_sim.visualizer import TrafficVisualizer


def grab_window(root: tk.Tk) -> Image.Image:
    try:
        from AppKit import NSApplicationActivateIgnoringOtherApps, NSRunningApplication

        application = NSRunningApplication.runningApplicationWithProcessIdentifier_(os.getpid())
        application.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)
        time.sleep(0.12)
    except ImportError:
        pass
    root.update_idletasks()
    root.update()
    try:
        import Quartz

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly, Quartz.kCGNullWindowID
        )
        candidates = [
            window for window in windows
            if window.get(Quartz.kCGWindowOwnerPID) == os.getpid()
            and window.get(Quartz.kCGWindowLayer) == 0
        ]
        bounds = max(
            (window[Quartz.kCGWindowBounds] for window in candidates),
            key=lambda item: item["Width"] * item["Height"],
        )
        x, y = int(bounds["X"]), int(bounds["Y"])
        width, height = int(bounds["Width"]), int(bounds["Height"])
    except (ImportError, ValueError):
        x, y = root.winfo_rootx(), root.winfo_rooty()
        width, height = root.winfo_width(), root.winfo_height()
    return ImageGrab.grab((x, y, x + width, y + height))


def advance(visualizer: TrafficVisualizer, root: tk.Tk, steps: int) -> None:
    for _ in range(steps):
        if visualizer.episode_done:
            break
        visualizer._perform_step()
        for _ in range(3):
            root.update()
            time.sleep(0.018)


def capture(
    algorithm: str,
    model_path: Path,
    screenshot: Path,
    gif_path: Path | None,
    warmup_steps: int,
    gif_frames: int,
) -> None:
    root = tk.Tk()
    visualizer = TrafficVisualizer(root, model_path, algorithm)
    try:
        # TrafficVisualizer sets its size during construction; set the capture
        # position afterwards so the complete native window remains on screen.
        root.geometry("1240x790+20+20")
        root.lift()
        root.attributes("-topmost", True)
        root.update()
        advance(visualizer, root, warmup_steps)
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        image = grab_window(root)
        image.save(screenshot)

        width, height = image.size
        image.crop((0, 0, int(width * 0.61), height)).save(
            screenshot.parent / "simulator-traffic-queues.png"
        )
        image.crop((int(width * 0.60), 0, width, height)).save(
            screenshot.parent / "simulator-metrics-panel.png"
        )

        if gif_path is not None:
            frames: list[Image.Image] = []
            for _ in range(gif_frames):
                advance(visualizer, root, 1)
                frames.append(grab_window(root).convert("P", palette=Image.Palette.ADAPTIVE))
            gif_path.parent.mkdir(parents=True, exist_ok=True)
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=140,
                loop=0,
                optimize=True,
            )
    finally:
        root.attributes("-topmost", False)
        visualizer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("dqn", "ppo"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--warmup-steps", type=int, default=90)
    parser.add_argument("--gif-frames", type=int, default=28)
    args = parser.parse_args()
    capture(
        args.algorithm, args.model_path, args.screenshot, args.gif,
        args.warmup_steps, args.gif_frames,
    )


if __name__ == "__main__":
    main()
