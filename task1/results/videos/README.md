# 结果视频

在 Jetson 上运行时使用 `--save-video results/videos/jetson_demo.mp4` 保存结果视频。

验收视频固定展示：

1. 至少两类物体同时出现。
2. 检测框、类别和置信度。
3. 实时 FPS。
4. ROS2 `topic echo` 同步录屏证据。
5. 明暗、距离、角度变化和部分遮挡。

Windows 训练电脑已使用 Osmo Action 5 Pro 录制 40.0 秒、1280×720、15 FPS 的本地演示，画面包含检测框、类别、置信度和完整循环 FPS，并出现键盘与手机同时检测。根据本次提交要求，`windows_dji_demo.mp4` 及其本地元数据不上传 GitHub；可使用 `src/windows_demo.py` 和运行说明中的命令重新生成。

ROS2 实机发布已由使用者确认成功，但本次没有提交 Jetson 视频、`topic echo`、JSONL 或 FPS 文件。Windows 视频不作为 Jetson FPS 证据。PyTorch 基准后续写入 `jetson_demo.mp4` 和 `results/fps_jetson.json`；TensorRT 后备测试写入 `jetson_demo_engine.mp4` 和 `results/fps_jetson_engine.json`。两份 FPS JSON 都记录对应模型 SHA-256、运行参数和实测速度。
