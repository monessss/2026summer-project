#!/usr/bin/env python3
"""Run real-time YOLO detection on Jetson and publish JSON over ROS 2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from common import portable_path, resolve_dataset, resolve_project_path, sha256_file


REQUIRED_WARMUP_FRAMES = 30
REQUIRED_MEASURED_FRAMES = 500


def artifact_contract(
    model_path: Path,
    video_path: Path | None,
    jsonl_path: Path | None,
    metrics_path: Path,
) -> str:
    """Identify one of the two fixed Jetson result layouts."""
    pytorch_output_contract = (
        portable_path(model_path) == "models/best.pt"
        and video_path is not None
        and portable_path(video_path) == "results/videos/jetson_demo.mp4"
        and jsonl_path is not None
        and portable_path(jsonl_path) == "results/jetson_detections.jsonl"
        and portable_path(metrics_path) == "results/fps_jetson.csv"
    )
    if pytorch_output_contract:
        return "pytorch_fp16"
    tensorrt_output_contract = (
        portable_path(model_path) == "models/best.engine"
        and video_path is not None
        and portable_path(video_path) == "results/videos/jetson_demo_engine.mp4"
        and jsonl_path is not None
        and portable_path(jsonl_path) == "results/jetson_detections_engine.jsonl"
        and portable_path(metrics_path) == "results/fps_jetson_engine.csv"
    )
    return "tensorrt_fp16" if tensorrt_output_contract else "mismatch"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", dest="half", action="store_true")
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.set_defaults(half=True)
    parser.add_argument("--topic", default="/yolo/detections")
    parser.add_argument("--node-name", default="task1_yolo_detector")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-video", default="results/videos/jetson_demo.mp4")
    parser.add_argument("--save-jsonl", default="results/jetson_detections.jsonl")
    parser.add_argument("--metrics-csv", default="results/fps_jetson.csv")
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--max-measured-frames",
        type=int,
        default=500,
        help="Stop after this many post-warmup frames; 0 runs until q/Ctrl-C",
    )
    return parser


def open_camera(cv2: Any, camera: int) -> Any:
    if platform.system() == "Linux":
        capture = cv2.VideoCapture(camera, cv2.CAP_V4L2)
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(camera)


def main() -> int:
    parser = build_parser()
    args, ros_args = parser.parse_known_args()

    import cv2
    import rclpy
    import torch
    import ultralytics
    from rclpy.node import Node
    from std_msgs.msg import String
    from ultralytics import YOLO

    model_path = resolve_project_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    rclpy.init(args=ros_args)
    node = Node(args.node_name)
    publisher = node.create_publisher(String, args.topic, 10)
    model = YOLO(str(model_path))
    class_names = {int(index): str(name) for index, name in model.names.items()}
    expected_data, _split_paths = resolve_dataset(resolve_project_path(args.data))
    if class_names != expected_data["names"]:
        node.destroy_node()
        rclpy.shutdown()
        raise ValueError(
            f"Model classes {class_names} do not match dataset classes {expected_data['names']}"
        )

    capture = open_camera(cv2, args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    if not capture.isOpened():
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    camera_properties = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }

    video_writer = None
    video_path = resolve_project_path(args.save_video) if args.save_video else None
    if video_path:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.unlink(missing_ok=True)
    jsonl_path = resolve_project_path(args.save_jsonl) if args.save_jsonl else None
    jsonl_file = None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = jsonl_path.open("w", encoding="utf-8")
    metrics_path = resolve_project_path(args.metrics_csv) if args.metrics_csv else None
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.unlink(missing_ok=True)
        metrics_path.with_suffix(".json").unlink(missing_ok=True)
    metrics_rows: list[dict[str, Any]] = []
    measured_intervals: list[float] = []
    video_encoding_fps = float(max(args.camera_fps, 1))

    node.get_logger().info(f"Model: {model_path}")
    node.get_logger().info(f"Publishing std_msgs/String JSON on {args.topic}")
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    frame_id = 0
    consecutive_read_failures = 0
    exit_code = 1 if metrics_path else 0
    try:
        while rclpy.ok():
            ok, frame = capture.read()
            if not ok:
                consecutive_read_failures += 1
                node.get_logger().warning("Camera frame read failed")
                if consecutive_read_failures >= 30:
                    raise RuntimeError("Camera failed to provide 30 consecutive frames")
                continue
            consecutive_read_failures = 0
            frame_id += 1
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                half=args.half,
                verbose=False,
            )[0]
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].cpu().tolist())
                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_names.get(class_id, str(class_id)),
                            "confidence": round(float(box.conf[0].item()), 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        }
                    )

            current_time = time.perf_counter()
            frame_interval = max(current_time - previous_time, 1e-9)
            current_fps = 1.0 / frame_interval
            previous_time = current_time
            smoothed_fps = current_fps if smoothed_fps == 0.0 else smoothed_fps * 0.9 + current_fps * 0.1
            if frame_id > args.warmup_frames and (metrics_path or args.max_measured_frames):
                measured_intervals.append(frame_interval)
                metrics_rows.append(
                    {
                        "frame_id": frame_id,
                        "total_ms": round(frame_interval * 1000.0, 3),
                        "fps": round(current_fps, 3),
                        "smoothed_fps": round(smoothed_fps, 3),
                        "object_count": len(detections),
                    }
                )
            payload = {
                "timestamp": time.time(),
                "frame_id": frame_id,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "fps": round(smoothed_fps, 2),
                "object_count": len(detections),
                "objects": detections,
            }
            message = String()
            message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            publisher.publish(message)
            if jsonl_file:
                jsonl_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                jsonl_file.flush()

            annotated = result.plot()
            cv2.putText(
                annotated,
                f"FPS: {smoothed_fps:.1f}  ROS2: {len(detections)} objects",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if video_path and video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    video_encoding_fps,
                    (annotated.shape[1], annotated.shape[0]),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {video_path}")
            if video_writer:
                video_writer.write(annotated)
            if not args.headless:
                cv2.imshow("Task1 YOLO ROS2 Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            rclpy.spin_once(node, timeout_sec=0.0)
            if frame_id % 100 == 0:
                node.get_logger().info(f"frame={frame_id} fps={smoothed_fps:.1f} objects={len(detections)}")
            if args.max_measured_frames and len(metrics_rows) >= args.max_measured_frames:
                node.get_logger().info(f"Completed {len(metrics_rows)} measured frames")
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if video_writer:
            video_writer.release()
        if jsonl_file:
            jsonl_file.close()
        if metrics_path and metrics_rows:
            with metrics_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(metrics_rows[0]))
                writer.writeheader()
                writer.writerows(metrics_rows)
            fps_values = [float(row["fps"]) for row in metrics_rows]
            mean_fps = len(measured_intervals) / sum(measured_intervals)
            selected_artifact_contract = artifact_contract(
                model_path, video_path, jsonl_path, metrics_path
            )
            protocol_matches = (
                args.camera == 0
                and args.width == 640
                and args.height == 480
                and args.camera_fps == 30
                and camera_properties["width"] == 640
                and camera_properties["height"] == 480
                and args.imgsz == 640
                and args.conf == 0.50
                and args.iou == 0.70
                and str(args.device) == "0"
                and args.half
                and args.topic == "/yolo/detections"
                and args.node_name == "task1_yolo_detector"
                and args.warmup_frames == REQUIRED_WARMUP_FRAMES
                and args.max_measured_frames == REQUIRED_MEASURED_FRAMES
                and selected_artifact_contract != "mismatch"
            )
            summary = {
                "measured_frames": len(metrics_rows),
                "warmup_frames": args.warmup_frames,
                "mean_fps": mean_fps,
                "median_fps": statistics.median(fps_values),
                "min_fps": min(fps_values),
                "max_fps": max(fps_values),
                "acceptance_threshold_fps": 5.0,
                "required_warmup_frames": REQUIRED_WARMUP_FRAMES,
                "required_measured_frames": REQUIRED_MEASURED_FRAMES,
                "artifact_contract": selected_artifact_contract,
                "protocol_matches": protocol_matches,
                "passed": (
                    len(metrics_rows) == REQUIRED_MEASURED_FRAMES
                    and protocol_matches
                    and mean_fps >= 5.0
                ),
                "measurement_definition": (
                    "full processing-loop FPS including camera read, inference, drawing, "
                    "ROS2 publication and result recording"
                ),
                "model": portable_path(model_path),
                "model_sha256": sha256_file(model_path),
                "data": portable_path(resolve_project_path(args.data)),
                "classes": class_names,
                "parameters": vars(args),
                "outputs": {
                    "video": portable_path(video_path) if video_path else None,
                    "video_encoding_fps": video_encoding_fps,
                    "detections_jsonl": portable_path(jsonl_path) if jsonl_path else None,
                    "metrics_csv": portable_path(metrics_path),
                },
                "environment": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "python": sys.version.split()[0],
                    "opencv": cv2.__version__,
                    "ultralytics": ultralytics.__version__,
                    "pytorch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device": (
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
                    ),
                    "ros_distro": os.environ.get("ROS_DISTRO", "not_exported"),
                    "camera_width": camera_properties["width"],
                    "camera_height": camera_properties["height"],
                    "camera_fps": camera_properties["fps"],
                },
            }
            exit_code = 0 if summary["passed"] else 1
            metrics_path.with_suffix(".json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
