# 贡献与安全

## 报告的问题

每份报告都带 `path:line` 引用。如果你发现某处引用对不上（行号漂移、结论与源码不符、数字无法从 `artifacts/` 复算），那是一个真问题——请开 issue 并写明：

1. 哪份报告、哪一节；
2. 引用的 `path:line` 现在实际是什么；
3. 你复算的步骤与结果。

## 已标注的 UNKNOWN

以下项在源码阅读与实测中确认**未定位**，不算 bug，但欢迎补：

- 真 `headroom=0` 的引擎容量从未实测（`reports/18`）。
- int8 页数量级偏 2.3% 的方向未定位；`minres` 差 98 MiB 的成因未定位（`reports/21 §6`）。
- ROCm/HIP 侧是否存在对应 WDDM 的 pinned host 约束（`reports/06`）。

## 安全与脱敏

- 仓库**不含**任何凭据、令牌、私钥。
- 云主机地址已脱敏为 `[T4_HOST]` / `[ECS_US_HOST]` / `[ECS_HZ_HOST]` / `[SEETACLOUD_HOST]`；个人绝对路径已移除。
- `artifacts/` 是 Kaggle 内核的公开输出，不含账号信息。

## 许可

本仓库的报告、笔记与脚本以 MIT 发布，见 [LICENSE](LICENSE)。被引用的上游项目（NInfer / KVMem / llama.cpp）保留其原始许可，本仓库不重分发其源码。
