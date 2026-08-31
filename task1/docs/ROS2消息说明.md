# ROS2 消息说明

运行状态：已确认实机能够通过 `/yolo/detections` 发布消息。

## 节点与话题

- 节点名称：`task1_yolo_detector`
- 话题名称：`/yolo/detections`
- 消息类型：`std_msgs/msg/String`
- QoS 队列深度：10
- 编码：UTF-8 JSON

## JSON 示例

```json
{
  "timestamp": 1788163200.12,
  "frame_id": 125,
  "image_width": 640,
  "image_height": 480,
  "fps": 18.6,
  "object_count": 2,
  "objects": [
    {
      "class_id": 2,
      "class_name": "phone",
      "confidence": 0.9321,
      "bbox": {"x1": 120, "y1": 80, "x2": 340, "y2": 420}
    }
  ]
}
```

| 字段 | 类型 | 含义 |
|---|---|---|
| `timestamp` | float | Unix 时间戳，秒 |
| `frame_id` | int | 启动后的递增帧号 |
| `image_width/height` | int | 实际输入帧尺寸 |
| `fps` | float | 指数平滑的实时处理速度 |
| `object_count` | int | 当前帧的检测数量 |
| `class_id/name` | int/string | 类别编号和名称 |
| `confidence` | float | 模型置信度 |
| `bbox` | object | 原始图像像素坐标 `xyxy` |

验证：

```bash
ros2 topic type /yolo/detections
ros2 topic echo /yolo/detections
ros2 topic hz /yolo/detections
```

本实验固定使用 `std_msgs/msg/String` 承载 UTF-8 JSON；验收端按上表字段解析。
