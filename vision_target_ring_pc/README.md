# 靶环视觉模块

这个目录只负责初赛/决赛流程中的靶环识别，当前模型类别为 `target_1`、`target_2`、`target_3`。物料识别和串口通信保持在目录外，通过控制接口接入。

## 运行环境

需要 Python 3.9+、`opencv-python` 和 `numpy`。当前实现使用 OpenCV DNN 加载 YOLO ONNX 模型，不依赖 `ultralytics` 或 `onnxruntime`。

```powershell
python -m pip install -r requirements.txt
```

## 单张图片测试

```powershell
python run_target_ring.py `
  --model ..\bullseyebig.v1i.yolov11\best.onnx `
  --image ..\bullseyebig.v1i.yolov11\test\images\0_f000374_jpg.rf.de0e3f339f9ea71fd00546fe1ff07e96.jpg `
  --save annotated.jpg
```

输出 JSON 中始终包含 `target_1`、`target_2`、`target_3` 三个键。每个目标包含：

- `detected`：是否检测到；
- `center`：最终提供给控制侧的像素坐标；
- `bbox_center`：检测框中心；
- `bbox`：检测框坐标；
- `confidence`：置信度；
- `center_source`：`bbox` 或 `geometry_refined`。

靶环不要求完整出现在画面中。YOLO 先给出编号和粗框，部署端优先在框内拟合环边界；拟合失败或靶环不完整时自动回退到检测框中心，连续 2 帧确认后即可输出。

## 控制接口

```python
from target_ring import TargetRingRecognizer, VisionController

recognizer = TargetRingRecognizer("path/to/best.onnx")
controller = VisionController(target_ring=recognizer)

controller.handle_command({"command": "SET_MODE", "mode": "TARGET_RING"})
result = controller.process_frame(frame)
```

当前支持的控制命令：

- `SET_MODE`：`TARGET_RING` 或 `MATERIAL`；
- `GET_MODE`：读取当前模式；
- `DETECT`：由上层在拿到图像后调用 `process_frame` 执行识别。

`MATERIAL` 模式只保留接口，尚未实现物料模型，不会误调用靶环模型。串口只需要在上层把收到的命令传给 `handle_command`，再把 `process_frame` 返回的数据编码后发送；本目录不打开串口、不绑定设备。
### 靶环中心

YOLO 只负责靶环编号和粗定位。`target_ring/center.py` 在检测框扩展 ROI 内提取黑白边缘，对多个闭合环边界分别进行椭圆拟合，并按共同中心和不同尺度进行鲁棒融合。数字笔画不会作为中心证据。

至少两个不同尺度的椭圆通过残差、覆盖率、轴比和中心一致性检查时，结果标记为 `multi_ellipse`；否则返回检测框中心。部署端不要求几何拟合成功，连续 2 帧同编号且中心跳变在允许范围内即可输出；目标不完整时也可发送粗中心。

离线检查实拍图：

```powershell
python evaluate_photos.py --model ..\bullseyeV2.v1i.yolov11\best2.onnx --images C:\path\to\photos --save-dir .\photo_eval
```
