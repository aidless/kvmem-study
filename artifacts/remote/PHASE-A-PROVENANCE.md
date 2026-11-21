# Phase A 模型制备溯源（2026-11-10 完成）

## 产物
- `kvmem-study/models/qwen3_5_4b_mtp.ninfer`
  - size = 2,827,176,704 bytes（与 docs/qwen35-dense-family.md 参考值 2,827,176,704 逐字节一致）
  - sha256 = c81dfc4a7de9c843a297d49cd4f12ee5057a7822c41ad1c986bc76d70bbbf16a（本地 sha256sum 与内核自算一致）
- `kvmem-study/remote/qwen3_5_4b_mtp.ninfer.conversion.json`（转换器生成的逐张量溯源清单）

## 执行环境
- Kaggle CPU 内核 <kaggle-user>/kvmem-convert-qwen35-4b，version 10，2026-11-10T12:24:25Z→12:26:48Z（全程 2m23s）
- 脚本：`kvmem-study/kaggle/convert_kernel.py`；数据集：<kaggle-user>/kvmem-src（kvmem master tarball, VERSION=0.19.0）

## 四阶段（result.json 全绿）
1. extract_kvmem：kvmem-master 目录树 → /tmp/work/kvmem
2. prepare_ninfer：git init kvmem 根（绕开 SOURCE-MANIFEST.json 导出包分支）+ backends/ninfer 钉 iamwavecut/ninfer-all@6cc1f9fa438b7094685ae417dfa40e1ca324e617 → prepare-backends.py 补丁链逐个 verify 全过
3. download_model：unsloth/Qwen3.5-4B-MTP-GGUF@86835bf9949e4d14d6860f7910b1340ad4f271a9 的 Qwen3.5-4B-Q4_K_M.gguf，sha256=3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630（权威值核对一致）；config/tokenizer 取 Qwen/Qwen3.5-4B
4. convert：`python3 -m tools.convert --model <config> --recipe qwen3_5_dense_gguf --components text,mtp --source gguf=<...> --device cpu`

## generation_config.json 来源记录（weight-conversion.md 指引）
官方 Qwen/Qwen3.5-4B 仓库无 generation_config.json（docs/qwen35-dense-family.md:25 亦确认）。
内核按官方 config.json 的 eos_token_id=248044（ModelScope Qwen/Qwen3.5-4B-Base config.json 实测核对）生成最小资源 {"eos_token_id": 248044}，未借用其他模型 generation 设置。

## 遗留
- Phase B（构建+扫描）仍缺 Linux GPU 主机：T4/ecs/seetacloud 2026-11-10 均 DEAD；Kaggle GPU P100 sm_60 不满足 NInfer 最低 sm_70。
- Kaggle 端输出仍保留同一 .ninfer（可在新 GPU 主机上直接 kaggle kernels output 拉取，避免再转换）。
