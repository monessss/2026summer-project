# 结果视频

在 Jetson 上运行时使用 `--save-video results/videos/jetson_demo.mp4` 保存结果视频。

验收视频固定展示：

1. 至少两类物体同时出现。
2. 检测框、类别和置信度。
3. 实时 FPS。
4. ROS2 `topic echo` 同步录屏证据。
5. 明暗、距离、角度变化和部分遮挡。

当前状态：Jetson 实测尚未执行，结果视频尚未生成。PyTorch 基准写入 `jetson_demo.mp4` 和 `results/fps_jetson.json`；TensorRT 后备测试写入 `jetson_demo_engine.mp4` 和 `results/fps_jetson_engine.json`。两份 FPS JSON 都记录对应模型 SHA-256、运行参数和实测速度。
