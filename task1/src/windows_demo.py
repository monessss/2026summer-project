#!/usr/bin/env python3
"""Run the trained detector from a Windows camera without requiring ROS 2."""

from __future__ import annotations

import argparse
import platform
import time
from collections import deque
from pathlib import Path
from typing import Any

from common import (
    portable_path,
    require_ascii_project_path_on_windows,
    resolve_dataset,
    resolve_project_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--output-fps", type=float, default=15.0)
    parser.add_argument("--save-video", default="")
    return parser.parse_args()


def open_camera(camera_index: int, width: int, height: int) -> Any:
    import cv2

    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        capture = cv2.VideoCapture(camera_index, backend)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _frame = capture.read()
        if ok:
            return capture
        capture.release()
    raise RuntimeError(f"Cannot open Windows camera index {camera_index}")


def create_video_writer(path: Path, width: int, height: int, fps: float) -> Any:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, fps),
        (width, height),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create video: {path}")
    return writer


def main() -> int:
    args = parse_args()
    if platform.system() != "Windows":
        raise RuntimeError("windows_demo.py is intended for Windows")
    if args.headless and args.max_frames <= 0:
        raise ValueError("--headless requires --max-frames greater than 0")
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    if args.duration < 0:
        raise ValueError("--duration cannot be negative")
    if args.output_fps <= 0:
        raise ValueError("--output-fps must be greater than 0")

    require_ascii_project_path_on_windows()
    model_path = resolve_project_path(args.model)
    data_yaml = resolve_project_path(args.data)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install dependencies with requirements-train.txt") from exc

    data, _split_paths = resolve_dataset(data_yaml)
    model = YOLO(str(model_path))
    model_class_names = {int(index): str(name) for index, name in model.names.items()}
    if model_class_names != data["names"]:
        raise ValueError(
            f"Model classes {model_class_names} do not match dataset classes {data['names']}"
        )

    capture = open_camera(args.camera, args.width, args.height)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = None
    video_path: Path | None = None
    if args.save_video:
        video_path = resolve_project_path(args.save_video)
        writer = create_video_writer(
            video_path,
            actual_width,
            actual_height,
            args.output_fps,
        )

    model_sha256 = sha256_file(model_path)
    print(f"Model: {portable_path(model_path)}")
    print(f"SHA-256: {model_sha256}")
    print(f"Camera: {args.camera}, {actual_width}x{actual_height}, reported {camera_fps:.1f} FPS")
    if not args.headless:
        print("Press q or Esc in the preview window to stop.")

    frame_times: deque[float] = deque(maxlen=30)
    frame_count = 0
    written_frame_count = 0
    started = time.perf_counter()
    previous_frame_time = started
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")

            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            annotated = result.plot()

            now = time.perf_counter()
            frame_times.append(now - previous_frame_time)
            previous_frame_time = now
            frame_count += 1
            smoothed_fps = len(frame_times) / sum(frame_times) if sum(frame_times) > 0 else 0.0
            detection_count = len(result.boxes) if result.boxes is not None else 0
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 34), (25, 25, 25), -1)
            elapsed = now - started
            recording_text = (
                f" | REC {min(elapsed, args.duration):.1f}/{args.duration:.0f}s"
                if args.duration > 0
                else ""
            )
            cv2.putText(
                annotated,
                f"Windows demo | FPS {smoothed_fps:.1f} | objects {detection_count}{recording_text}",
                (8, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                target_written_frames = max(1, int(elapsed * args.output_fps))
                while written_frame_count < target_written_frames:
                    writer.write(annotated)
                    written_frame_count += 1
            if not args.headless:
                cv2.imshow("Task1 YOLO26 Windows Demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
            if args.duration > 0 and elapsed >= args.duration:
                break
            if frame_count % 30 == 0:
                print(f"Frames: {frame_count}, FPS: {smoothed_fps:.1f}, objects: {detection_count}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    mean_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"Completed: {frame_count} frames in {elapsed:.2f}s, mean loop FPS {mean_fps:.2f}")
    if writer is not None:
        print(f"Video frames written: {written_frame_count} at {args.output_fps:.2f} FPS")
    if video_path is not None:
        metadata_path = video_path.with_suffix(".json")
        write_json(
            metadata_path,
            {
                "video": portable_path(video_path),
                "video_sha256": sha256_file(video_path),
                "video_bytes": video_path.stat().st_size,
                "video_frames": written_frame_count,
                "video_fps": args.output_fps,
                "video_duration_seconds": written_frame_count / args.output_fps,
                "resolution": {"width": actual_width, "height": actual_height},
                "source_frames_processed": frame_count,
                "recording_elapsed_seconds": elapsed,
                "mean_loop_fps": mean_fps,
                "camera_index": args.camera,
                "model": portable_path(model_path),
                "model_sha256": model_sha256,
                "parameters": vars(args),
                "scope": "Windows camera preview; not Jetson or ROS2 acceptance evidence",
            },
        )
        print(f"Video metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
