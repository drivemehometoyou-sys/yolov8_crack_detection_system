# 训练日志

---

## exp5 — 2026-07

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | yolov8m (25.86M params) |
| 数据集 | my_real_cracks (3717 train / 200 val) |
| epochs | 200 |
| batch | 16 |
| 优化器 | AdamW, lr0=0.001 |
| workers | 4 |
| device | NVIDIA RTX 4060 Laptop (8G) |
| 增强 | 默认 (mosaic=1.0, scale=0.5, translate=0.1, fliplr=0.5, hsv) |
| 早停 | 无 (patience=100) |
| 训练时长 | ~7 小时 |
| 备注 | 从 ~170 轮 resume 续训完成 |

### 结果指标

| 指标 | 值 |
|------|-----|
| best mAP50 | 0.8355 (epoch 95) |
| best mAP50-95 | 0.6457 (epoch 95) |
| best P | 0.842 |
| best R | 0.767 |
| 最终 mAP50 (epoch 200) | 0.7606 |
| final box_loss | 0.554 |
| final cls_loss | 0.407 |

### 问题分析

- **过拟合**：95 轮后 mAP 持续下降至 0.761，box_loss 却从 0.84 降至 0.55，107 轮白练
- **漏检**：P=0.85 > R=0.73，约 27% 裂缝未被检测
- **定位粗糙**：mAP50-95 与 mAP50 差距 0.19，IoU 严格时框不准

---

