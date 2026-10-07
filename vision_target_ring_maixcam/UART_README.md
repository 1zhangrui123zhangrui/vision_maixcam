# UART 视觉数据包说明

## 串口

- 设备：UART0 `/dev/ttyS0`
- 115200 baud，8N1，无流控，3.3 V TTL，共地
- MaixCAM Pro TX A16 接单片机 RX；RX A17 接单片机 TX
- UART0 默认映射 A16/A17，无需额外 pinmap。开机会输出日志，单片机应丢弃启动日志及不符合 JSON 协议的行，只接收有效 `VISION_RESULT` 包；不要将 `serial ready` 当作视觉程序已就绪。
+- 若启用了串口登录终端或其他 maix protocol 应用，需确认其未与本程序同时读写 UART0。打开端口成功不代表没有占用冲突；没有实机信息时不要直接修改 `/boot/uEnv.txt`。
+- A16 同时参与启动模式检测，上电时不能被外部电路拉低，否则可能无法启动。
+- 官方说明：https://wiki.sipeed.com/maixpy/doc/zh/peripheral/uart.html
- 单片机命令为 `1\n`、`2\n`；视觉结果为单向发送，不要求单片机应答

## 坐标约定

- 图像为 640x480 像素，左上角为原点，x 向右、y 向下。
- `ref` 是机械臂已知抓取中心，不是图像中心。
- `dx = 检测中心 x - ref x`，`dy = 检测中心 y - ref y`。正值表示检测点位于已知点右方/下方。
- 向量单位为像素；未完成平面/尺度标定前，不代表毫米。

## 包格式

每个包是 ASCII JSON，末尾为换行符 `\n`。`seq` 是视觉端 uint32 递增序号，回绕后从 0 开始。单片机可检查序号是否连续。

### 物料模式：命令 1

```json
{"v":1,"seq":42,"type":"VISION_RESULT","mode":"1","model":"MATERIAL","status":"OK","frame_id":103,"t_ms":123456,"age_ms":2,"size":[640,480],"ref":[346,169],"materials":[{"id":1,"confidence":0.92,"x":318,"y":241,"dx":-28,"dy":72,"flags":17}],"objects":[[1,0.92,318,241,-28,72,17]]}
```

`mode` 回显当前命令，`id` 为颜色编号：红 1、黄 2、蓝 3、绿 4、黑 5、浅蓝 6。多件物料时 `materials` 含多个元素。

### 靶环模式：命令 2

```json
{"v":1,"seq":43,"type":"VISION_RESULT","mode":"2","model":"TARGET_RING","status":"OK","frame_id":104,"t_ms":123556,"age_ms":2,"size":[640,480],"ref":[346,169],"targets":[{"id":1,"detected":true,"confidence":0.91,"x":300,"y":180,"dx":-46,"dy":11,"flags":17},{"id":2,"detected":false,"confidence":0,"x":0,"y":0,"dx":0,"dy":0,"flags":0},{"id":3,"detected":false,"confidence":0,"x":0,"y":0,"dx":0,"dy":0,"flags":0}],"objects":[[1,0.91,300,180,-46,11,17]]}
```

`targets` 始终固定按 1、2、3 排列。`detected=false` 表示该靶环当前不可用，坐标清零。

## 状态和动作安全

- `status`：`OK`、`UNCONFIRMED`、`NO_TARGET`、`ERROR`、`OVERFLOW`、`STALE`、`IDLE`。
- `flags` 是位标志：bit0 连续帧确认、可供动作使用；bit1 贴图像边缘；bit2 检测歧义；bit3 框过小；bit4 确认完成。
- 仅在 `status="OK"` 且对象 `flags & 1 != 0` 时使用对应检测点和偏差。
- 没有目标、识别错误、过期结果时不发送可用对象坐标；不要沿用上一包的坐标。
- `objects` 是兼容旧固件的紧凑格式 `[id,confidence,x,y,dx,dy,flags]`，新程序应读取清晰命名的 `materials` / `targets`。
