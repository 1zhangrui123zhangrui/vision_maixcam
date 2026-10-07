# MaixCAM Pro 视觉联调说明

## 单片机命令

单片机发送一行 ASCII 数字并以 LF (`0x0A`) 结束：

- `1`: 识别物料，输出颜色和顶部圆面中心。原料区和靶环上的物料共用此命令。
- `2`: 识别靶环，输出 `target_1/2/3` 中心。

也支持带序号 JSON，供正式协议使用：

```json
{"seq":1,"command":"SET_MODE","mode":"1"}
{"seq":2,"command":"SET_MODE","mode":"2"}
{"seq":3,"command":"SET_REFERENCE","x":346,"y":169}
```

数字命令没有命令序号，适合当前联调；JSON 命令的 `seq` 为 uint32，重复/乱序命令会被丢弃。视觉端不返回命令 ACK。

## UART0

- 设备：`/dev/ttyS0`
- MaixCAM Pro TX：A16，连接单片机 RX
- MaixCAM Pro RX：A17，连接单片机 TX
- 115200 baud，8N1，无流控，3.3 V TTL，共地
- UART0 默认映射 A16/A17，无需额外 pinmap。开机会输出日志，单片机应丢弃启动日志及不符合 JSON 协议的行，只接收有效 `VISION_RESULT` 包；不要将 `serial ready` 当作视觉程序已就绪。
+- 若启用了串口登录终端或其他 maix protocol 应用，需确认其未与本程序同时读写 UART0。打开端口成功不代表没有占用冲突；没有实机信息时不要直接修改 `/boot/uEnv.txt`。
+- A16 同时参与启动模式检测，上电时不能被外部电路拉低，否则可能无法启动。
+- 官方说明：https://wiki.sipeed.com/maixpy/doc/zh/peripheral/uart.html

## 视觉结果

视觉单向发送换行 JSON，结果包自带递增 `seq`，不等待 ACK：

每行一个 JSON 包，以换行结束；视觉端只发送，不等待应答。`seq` 是视觉端递增包序号，单片机应按它识别重复/丢包。

物料模式 (`1`) 示例：

```json
{"v":1,"seq":42,"type":"VISION_RESULT","mode":"1","model":"MATERIAL","status":"OK","frame_id":103,"size":[640,480],"ref":[346,169],"materials":[{"id":1,"confidence":0.92,"x":318,"y":241,"dx":-28,"dy":72,"flags":17}],"objects":[[1,0.92,318,241,-28,72,17]]}
```

靶环模式 (`2`) 示例：

```json
{"v":1,"seq":43,"type":"VISION_RESULT","mode":"2","model":"TARGET_RING","status":"OK","frame_id":104,"size":[640,480],"ref":[346,169],"targets":[{"id":1,"confidence":0.91,"x":300,"y":180,"dx":-46,"dy":11,"flags":17},{"id":2,"detected":false,"confidence":0,"x":0,"y":0,"dx":0,"dy":0,"flags":0},{"id":3,"detected":false,"confidence":0,"x":0,"y":0,"dx":0,"dy":0,"flags":0}],"objects":[[1,0.91,300,180,-46,11,17]]}
```

单片机优先读取 `materials` 或 `targets`；`objects` 是兼容旧程序的紧凑列表。

对象格式为 `[id, confidence, x, y, dx, dy, flags]`：

- 物料 `id`：红 1、黄 2、蓝 3、绿 4、黑 5、浅蓝 6。
- 靶环 `id`：1、2、3。
- `x/y`：检测框中心，单位为 640x480 图像像素。
- `dx/dy`：检测点减机械臂已知抓取点，计算为 `检测点 - 参考点`；`+x` 向右，`+y` 向下。单片机用这个向量控制机械臂移动。
- `materials[].id` 是颜色编号：红 1、黄 2、蓝 3、绿 4、黑 5、浅蓝 6。
- `targets[].id` 是靶环编号：1、2、3；靶环列表始终按 1、2、3 顺序发送，未检测到时 `detected=false`。
- flags bit0=连续帧确认可用，bit1=贴边，bit2=歧义，bit3=框过小，bit4=确认完成。
- 只有 bit0=1 的对象才允许单片机用于动作。

状态包括 `OK`、`UNCONFIRMED`、`NO_TARGET`、`ERROR`、`OVERFLOW`、`STALE`、`IDLE`。目标消失后不会沿用上一帧坐标。

## 固定点标定

当前程序默认参考点为 `(320,240)`，只是 640x480 画面中心，不代表已经完成机械标定。上机验证时先观察洋红色十字和靶环中心的相对位置，再按机械臂实际动作位置修改参考点。

1. 保持相机安装位置、焦距和 640x480 分辨率不变。
2. 把物料或靶环放到机械臂实际动作参考位置。
3. 让画面稳定，读取多帧 `x/y`，把稳定中心写入 `vision_target_ring_maixcam/config.py` 的 `REFERENCE_X/REFERENCE_Y`，或发送 `SET_REFERENCE`。
4. 检查该位置的 `dx/dy` 接近 0；目标向右/向下移动时偏差应分别增大。
5. 物料顶面和靶环平面高度不同时，应分别验证实际动作误差；本程序输出的是像素偏差，不是毫米坐标。

当前相机直观看效果的配置为 `START_MODE="2"`、`UART_ENABLED=False`。上传后直接启动靶环识别，结果只显示在屏幕上，不会发送到单片机。恢复串口联调时再改为 `UART_ENABLED=True`。

## PC 验证

```powershell
cd E:\daywork\工创赛\vision_target_ring_pc
python -B test_device_runtime.py
```

当前测试覆盖模型路由、两条数字命令、参考点偏差、连续帧确认、目标消失、串口半包/短写和旧结果丢弃。MaixCAM Pro 实测仍需确认模型加载、实际帧率、串口接线和机械标定误差。
