#!/usr/bin/env python3
"""capacity.py — 从 .ninfer 产物算显存容量，不需要 GPU、不需要跑引擎。

回答的问题是 ``stage_fits`` 的问题（src/core/stage_plan.h:202-213）：

    used = fixed_bytes + Σ权重 + ΣGDN state + (Σ该段 KV/页组 + 固定KV/页组) × page_groups
    fits ⟺ used ≤ available_bytes

所有输入都来自模型文件本身：.ninfer 头 32 字节后是一份 JSON manifest，
``objects[]`` 给了每个张量的物理字节，``bindings`` 给了逻辑名到物理对象的映射。

用法：
    python3 tools/capacity.py MODEL.ninfer                      # 默认 6144 MiB / 扣 400 / bf16+int8
    python3 tools/capacity.py MODEL.ninfer --vram-mib 12288
    python3 tools/capacity.py MODEL.ninfer --kv int8_g64 --json
    python3 tools/capacity.py MODEL.ninfer --kv nvfp4_g16   # 九档：bf16/int8_g64/fp8_row256/
                                                            # k8v4/nvfp4_g16/rk8v4/rk4v4/rk4v4_e8/rk2v4_e8
    python3 tools/capacity.py --selftest                       # 用 16 号报告的实测值反证脚本

口径（两条，报数必须带）：
    budget   —— 引擎自己的 fixed_bytes（1536 MiB + 1024 MiB + head_bytes）
    measured —— 只留 --workspace-mib（默认 135 MiB，来自上游实测峰值 2868 MiB − 全常驻）

两个口径在 6 GB 卡上差 6.4 倍容量，混用等于报错。
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
import sys

MiB = 1 << 20

# src/models/qwen3_5/load.cpp:189-191
K_STAGE_FIXED_BYTES = 1536 * MiB
K_HEAD_STAGE_FIXED_BYTES = 1024 * MiB
# src/core/paged_kv_cache.h:20
K_PAGED_KV_PAGE_SIZE = 64

# capacity.py 的容量公式是「从标称显存扣掉驱动+ECC+全部权重+GDN state」，与引擎
# 「从 available_after_weights 扣 headroom+minres」是两套口径；本加数是本工具口径的
# 标定常数（2），不是引擎 curve.minimum_main_page_groups。
# 引擎的 m 由 startup.cpp:1803-1807 = max(page_count(max_context), max_concurrency)
# + resident_main_pages = max(128,1)+0 = 128（报告 16 §12.1 钉死，v5 explicit 行佐证）。
# 两者同名不同物：把本常数改成 128 会让全部容量口径 selftest 偏 +126 页（已实测）。
MIN_MAIN_PAGE_GROUPS = 2
# 上游实测：docs/qwen35-dense-family.md:27（RTX 5050 Laptop 峰值 2868 MiB）
MEASURED_PEAK_MIB = 2868
MEASURED_WORKSPACE_MIB = 135

MAGIC = b"NINFER\x00"


class ManifestError(RuntimeError):
    pass


def read_manifest(path):
    """返回 (manifest_dict, digest_hex, header_bytes)。"""
    with open(path, "rb") as handle:
        head = handle.read(32)
        if not head.startswith(MAGIC):
            raise ManifestError(f"{path}: 不是 .ninfer（magic={head[:8]!r}）")
        version = head[7]
        (length,) = struct.unpack("<I", head[8:12])
        digest = head[16:32].hex()
        blob = handle.read(length)
        if len(blob) != length:
            raise ManifestError(f"{path}: manifest 截断，声明 {length} 实读 {len(blob)}")
        try:
            manifest = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}: manifest 不是 JSON（{exc}）") from exc
    return manifest, digest, {"version": version, "manifest_bytes": length, "header_bytes": 32}


def attribute_objects(manifest):
    """把每个物理 object 归属到 layer / head / mtp。

    返回 (owner, objects)。owner[object_id] 是 ('layer', i) / ('mtp',) / ('head',) 的集合。

    为什么不按 parts[].range 累加：**range 是元素数不是字节数**。实测
    ``weight/000061`` 物理 bytes=7,372,800，而 q 的 range 声称 [0, 10,485,760)
    （10,485,760 = 4096×2560 个元素）。按 range 差值当字节会让 32 层之和
    超过整个文件 308 MiB。物理字节只有一个权威来源：objects[].bytes。
    """
    objects = {obj["id"]: obj for obj in manifest.get("objects", [])}
    owner = collections.defaultdict(set)
    for name, binding in manifest.get("bindings", {}).items():
        if "object" in binding:
            ids = [binding["object"]]
        else:
            ids = [part["object"] for part in binding.get("parts", [])]
        if name.startswith("text/layers/"):
            where = ("layer", int(name.split("/")[2]))
        elif name.startswith("mtp/"):
            where = ("mtp",)
        else:
            where = ("head",)
        for oid in ids:
            owner[oid].add(where)
    return owner, objects


def weight_breakdown(manifest):
    """按 owner 把物理字节均摊去重，返回 dict。"""
    owner, objects = attribute_objects(manifest)
    cfg = manifest["components"]["text"]["config"]
    layer_count = cfg["num_hidden_layers"]

    per_layer = collections.Counter()
    other = collections.Counter()
    for oid, where in owner.items():
        obj = objects.get(oid)
        if obj is None:
            continue
        if obj.get("kind") == "resource":      # tokenizer 等，不进显存
            continue
        share = obj["bytes"] / len(where)       # 共享张量只算一次（如 tied embedding/head）
        for tag in where:
            if tag[0] == "layer":
                per_layer[tag[1]] += share
            else:
                other[tag[0]] += share
    return {
        "layers": [int(per_layer[i]) for i in range(layer_count)],
        "head": int(other["head"]),
        "mtp": int(other["mtp"]),
    }


def conv_channels(manifest):
    """从第一个 linear_attention 层的 gdn/convolution 形状取 [kernel, channels]。"""
    objects = {obj["id"]: obj for obj in manifest.get("objects", [])}
    cfg = manifest["components"]["text"]["config"]
    for index, kind in enumerate(cfg["layer_types"]):
        if kind != "linear_attention":
            continue
        binding = manifest["bindings"].get(f"text/layers/{index}/gdn/convolution")
        if not binding:
            continue
        oid = binding.get("object") or binding["parts"][0]["object"]
        shape = objects[oid].get("shape") or []
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
    return None


# src/core/dtype.h:7-27
DTYPE_SIZE = {"BF16": 2, "FP32": 4, "I32": 4, "U8": 1, "I64": 8,
              "I8": 1, "FP16": 2, "FP8_E4M3FN": 1}

# include/ninfer/kv_cache_storage.h:10-34 的九个值，按枚举顺序
KV_DTYPES = ("bf16", "int8_g64", "fp8_row256", "k8v4", "nvfp4_g16",
             "rk8v4", "rk4v4", "rk4v4_e8", "rk2v4_e8")


def _vector_bytes(data_dtype, data_extent, scale_dtype, scale_extent):
    """src/core/paged_kv_storage.h:38-44  PagedKVVectorLayout::physical_bytes()."""
    return (data_extent * DTYPE_SIZE[data_dtype]
            + scale_extent * DTYPE_SIZE[scale_dtype])


def kv_bytes_per_token_head(kv_dtype, head_dim=256):
    """复刻 src/core/paged_kv_storage.h:60-125 的 paged_kv_storage_layout() 并求和 K+V。

    每个 dtype 的布局直接照抄 switch 分支。九档全部按引擎代码算过一遍，
    权威对照是 /tmp/kvtable（编译 ninfer-src/src/core/dtype.cpp 跑出来的表）：
        1024 / 528 / 516 / 402 / 288 / 408 / 280 / 280 / 216
    顺序为 bf16, int8_g64, fp8_row256, k8v4, nvfp4_g16, rk8v4, rk4v4, rk4v4_e8, rk2v4_e8。
    """
    hd = head_dim
    symmetric = {
        # name          data_dtype      extent  scale_dtype  scale_extent
        "bf16":        ("BF16",         hd,     "U8",        0),
        "int8_g64":    ("I8",           256,    "FP16",      4),
        "fp8_row256":  ("FP8_E4M3FN",   256,    "FP16",      1),
        "nvfp4_g16":   ("U8",           128,    "U8",        16),
    }
    if kv_dtype in symmetric:
        if kv_dtype != "bf16" and hd != 256:
            raise ManifestError(
                f"{kv_dtype} 只在 head_dim 256 有布局（src/core/paged_kv_storage.h:60-125 "
                f"的 break→throw），本模型 head_dim {hd}")
        v = _vector_bytes(*symmetric[kv_dtype])
        return 2 * v                                    # K and V are the same layout
    asymmetric = {
        # name        key (dtype, extent, scale_dtype, scale_extent)      value (...)
        "k8v4":      (("FP8_E4M3FN", 256, "FP16", 1),                   ("U8", 128, "U8", 16)),
        "rk8v4":     (("I8", hd, "FP16", 4),                            ("U8", hd // 2, "FP16", 8)),
        "rk4v4":     (("U8", hd // 2, "FP16", 4),                       ("U8", hd // 2, "FP16", 8)),
        "rk4v4_e8":  (("U8", hd // 2, "FP16", 4),                       ("U8", hd // 2, "FP16", 8)),
        "rk2v4_e8":  (("U8", hd // 4, "FP16", 4),                       ("U8", hd // 2, "FP16", 8)),
    }
    if kv_dtype in asymmetric:
        key, value = asymmetric[kv_dtype]
        return _vector_bytes(*key) + _vector_bytes(*value)
    raise ManifestError(
        f"未知 kv dtype: {kv_dtype}（可用：{'/'.join(list(symmetric) + list(asymmetric))}）")


def planes_per_layer(kv_dtype):
    """src/core/paged_kv_storage.h: 只有 BFloat16 是 2 个 plane，其余量化档是 4 个。

    这决定 plane_page_range() 的可借段要对齐几个 plane（报告 06 第 4 点）——
    换 dtype 同时改变容量与可借性，不只是改变容量。
    """
    return 2 if kv_dtype == "bf16" else 4


def layer_costs(manifest, kv_dtype="bf16", gdn_state_fp16=False):
    """复刻 src/models/qwen3_5/load.cpp:304-325 的 LayerCost。

    返回 dict：kv_per_layer（每 full-attention 层每页组）、state_per_layer、
    full_attention 层数、linear 层数、state 合计。
    """
    cfg = manifest["components"]["text"]["config"]
    kinds = cfg["layer_types"]
    layer_count = cfg["num_hidden_layers"]
    if len(kinds) != layer_count:
        raise ManifestError(f"layer_types {len(kinds)} 项 ≠ num_hidden_layers {layer_count}")

    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    # src/core/paged_kv_storage.h:38-56,51,60-125 — physical_bytes_per_token_head() is
    # key.physical_bytes() + value.physical_bytes(), i.e. BOTH the K and the V vector.
    # First version of this file used one vector and undercounted every KV number by exactly 2×.
    # The judge was the spec's own 27B row: 64 tok × 4 KV heads × 16 FA layers × 1024 B = 4.00 MiB
    # (docs/maintainer/hybrid-prefix-cache-spec.md §3.1); one vector gives 2.00 MiB.
    bytes_per_token_head = kv_bytes_per_token_head(kv_dtype, head_dim)
    kv_per_layer = K_PAGED_KV_PAGE_SIZE * n_kv * bytes_per_token_head

    conv = conv_channels(manifest)
    linear_count = sum(1 for k in kinds if k != "full_attention")
    full_count = layer_count - linear_count
    state_total = 0
    state_per_layer = 0
    if linear_count and conv:
        kernel, channels = conv
        state_dtype = 2 if gdn_state_fp16 else 4
        conv_bytes = channels * (kernel - 1) * 2                    # bf16
        recurrent = (cfg["linear_key_head_dim"] * cfg["linear_value_head_dim"]
                     * cfg["linear_num_value_heads"] * state_dtype)
        state_per_layer = conv_bytes + recurrent
        state_total = linear_count * state_per_layer

    return {
        "kv_dtype": kv_dtype,
        "kv_per_layer": kv_per_layer,
        "full_attention_layers": full_count,
        "linear_layers": linear_count,
        "state_per_layer": state_per_layer,
        "state_total": state_total,
        "bytes_per_token_head": bytes_per_token_head,
    }


def plan(manifest, vram_mib, kv_dtype="bf16", driver_mib=400, workspace_mib=None,
         gdn_state_fp16=False, ecc=False, headroom_mib=1024):
    """单卡单 stage 的容量算术。

    两种口径（2026-11-19 由报告 18 的 T4 实测重新定义，见报告 16 第十二节）：

    - ``workspace_mib=None`` → **stage 预算口径**：引擎 `StageBudget.fixed_bytes`
      = 1536 + 1024 + head_bytes（`load.cpp:189-191,215-241`）。这回答的是
      「层切分方案装不装得下」，喂给 `stage_fits`（`stage_plan.h:202-213`）。
      单价含 MTP（`LayerCost::kv_bytes_per_page_group`，`load.cpp:317`）。
    - ``workspace_mib=<N>`` → **容量口径**：引擎 auto 容量
      `pages = min_pages + (available − headroom − minres) / 斜率`
      （`kv_capacity.cpp:115-133`、`startup.cpp:1832-1837`）。斜率是**实测差**，
      不含 MTP（MTP 的 KV 是 backend pool，`startup.cpp:1465-1471`）。

    两套口径的单价不同，混用会得出错的页数——报告 16 的第 12.4 节记的就是这个错。
    """
    weights = weight_breakdown(manifest)
    costs = layer_costs(manifest, kv_dtype, gdn_state_fp16)
    w_main = sum(weights["layers"])
    # head_stage_bytes（load.cpp:215-241）= token_embedding + output_head(tied，去重后 0)
    #                                 + final_norm + MTP 自己的全部参数
    head_bytes = weights["head"] + weights["mtp"]

    # 数据中心卡（T4/A100/H100/V100）默认开 ECC，标称显存的 ~6.25% 被 ECC 吃掉。
    # 报告 18 的 T4 日志原文：costing ~6.25% of capacity (23,028 vs 24,564 MiB on a 24 GB card)。
    ecc_factor = 0.9375 if ecc else 1.0

    # 第一层 stage 额外背 MTP 那一层的 KV（stage 预算口径）
    kv_per_layer = costs["kv_per_layer"]
    if workspace_mib is None:
        # stage 预算口径：单价含 MTP（load.cpp:317 的 LayerCost）
        kv_per_group = costs["full_attention_layers"] * kv_per_layer + kv_per_layer
        fixed = K_STAGE_FIXED_BYTES + K_HEAD_STAGE_FIXED_BYTES + head_bytes
        basis = "budget"
    else:
        # 容量口径：斜率不含 MTP（MTP KV 独立成 backend pool）
        kv_per_group = costs["full_attention_layers"] * kv_per_layer
        fixed = workspace_mib * MiB + head_bytes
        basis = "measured"

    available = (vram_mib - driver_mib) * MiB * ecc_factor
    if workspace_mib is None:
        residual = available - fixed - w_main - costs["state_total"]
    else:
        # 容量口径：引擎先扣 headroom，再扣 minimum_device_reservation_bytes
        residual = available - headroom_mib * MiB - fixed - w_main - costs["state_total"]
    page_groups = int(residual // kv_per_group) if residual > 0 else 0
    if workspace_mib is not None:
        # 引擎 pages = minimum_main_page_groups + floor(residual / 斜率)（kv_capacity.cpp:138-145）。
        # minimum_main_page_groups = max(logical_pages, max_concurrency) + resident_main_pages
        # （startup.cpp:1805-1807）；本模型无注入 graft → resident_main_pages = 0，
        # 单并发下 = max(1, 1) = 1... 实测标定值为 2，故取 MAX_MIN_PAGES。
        page_groups += MIN_MAIN_PAGE_GROUPS
    return {
        "basis": basis,
        "vram_mib": vram_mib,
        "driver_mib": driver_mib,
        "available_mib": available / MiB,
        "fixed_mib": fixed / MiB,
        "weights_mib": w_main / MiB,
        "head_mib": head_bytes / MiB,
        "state_mib": costs["state_total"] / MiB,
        "resident_mib": (w_main + head_bytes + costs["state_total"]) / MiB,
        "kv_per_group_bytes": kv_per_group,
        "kv_per_group_mib": kv_per_group / MiB,
        "residual_mib": residual / MiB,
        "page_groups": page_groups,
        "tokens": page_groups * K_PAGED_KV_PAGE_SIZE,
        "kv_dtype": kv_dtype,
        "layer_weights": weights["layers"],
        "full_attention_layers": costs["full_attention_layers"],
        "linear_layers": costs["linear_layers"],
        "state_per_layer": costs["state_per_layer"],
    }


def _fmt_row(name, vram):
    return f"{name:<20}{vram:>9}"


DEVICES = [
    ("3060 Laptop", 6144),
    ("4060 Laptop", 8188),
    ("3060 12G", 12288),
    ("4080 16G", 16380),
    ("4090 24G", 24564),
    ("5090 32G", 32607),
]


def render(manifest, args):
    out = []
    head = plan(manifest, args.vram_mib, "bf16", args.driver_mib, None)
    out.append(f"模型常驻（层 + head + MTP + GDN state）= {head['resident_mib']:.2f} MiB")
    out.append(f"  full_attention 层 {head['full_attention_layers']} / linear 层 {head['linear_layers']}"
               f"；GDN 每层 state {head['state_per_layer']:,} B")
    out.append("")
    for kv in KV_DTYPES:
        base = plan(manifest, args.vram_mib, kv, 0)
        out.append(f"=== KV={kv}  Σkv/页组 = {base['kv_per_group_bytes']:,} B "
                   f"({base['kv_per_group_mib']:.4f} MiB)  "
                   f"每 64K 上下文 {1024 * base['kv_per_group_mib']:.0f} MiB ===")
        for basis, ws in (("budget", None), (f"measured({int(args.workspace_mib)}MiB)", args.workspace_mib)):
            out.append(f"  -- {basis} 口径，扣 {args.driver_mib} MiB 驱动 --")
            out.append(f"  {'设备':<20}{'MiB':>9}{'B(页组)':>10}{'上下文 token':>14}")
            for name, vram in DEVICES:
                r = plan(manifest, vram, kv, args.driver_mib, ws)
                out.append(f"  {name:<20}{vram:>9}{r['page_groups']:>10}{r['tokens']:>14,}")
        out.append("")
    out.append("注：容量算术不知道模型的上下文上限，与 max_position_embeddings 取 min 才是可用值。")
    return "\n".join(out)


def selftest(model_path, perturb=False):
    """用 16 号报告的实测值反证本脚本（精确值逐字复制）。

    perturb=True 是 RED 证明：把一个期望值故意改 1，脚本必须报 FAIL。
    没有这道证明，selftest 通过只说明它没崩，不说明它会红。
    """
    manifest, digest, meta = read_manifest(model_path)
    checks = []

    def expect(label, got, want):
        checks.append((label, got, want, got == want))

    expect("manifest_bytes", meta["manifest_bytes"], 126_944)
    expect("digest", digest, "d9510404e5b946769c4c79dda6d0d2c9")

    weights = weight_breakdown(manifest)
    expect("32 层权重字节", sum(weights["layers"]), 2_211_549_184)
    expect("head 字节", weights["head"], 521_477_120)
    expect("MTP 字节", weights["mtp"], 81_168_384)
    expect("层数", len(weights["layers"]), 32)

    costs_bf16 = layer_costs(manifest, "bf16")
    expect("bf16 每层每页 KV", costs_bf16["kv_per_layer"], 262_144)
    expect("full_attention 层数", costs_bf16["full_attention_layers"], 8)
    expect("GDN state 合计", costs_bf16["state_total"], 51_511_296)
    costs_i8 = layer_costs(manifest, "int8_g64")
    expect("int8 每层每页 KV", costs_i8["kv_per_layer"], 135_168)

    # 九档 B/token/head —— 权威值来自编译引擎自己的代码（/tmp/kvtable，链接
    # ninfer-src/src/core/dtype.cpp，调 paged_kv_storage_layout(...).physical_bytes_per_token_head()）。
    # 这是「不与引擎对表就不知道自己错了」那一条的机器化：任何一档改了布局，这里必红。
    engine_table = {"bf16": 1024, "int8_g64": 528, "fp8_row256": 516, "k8v4": 402,
                    "nvfp4_g16": 288, "rk8v4": 408, "rk4v4": 280, "rk4v4_e8": 280,
                    "rk2v4_e8": 216}
    for name, want in engine_table.items():
        expect(f"{name} B/token/head", kv_bytes_per_token_head(name, 256), want)
    expect("planes_per_layer bf16", planes_per_layer("bf16"), 2)
    expect("planes_per_layer 量化档", planes_per_layer("int8_g64"), 4)
    # 27B 的外部独立裁判：64 tok × 4 KV × 16 FA × 1024 B = 4.00 MiB（spec §3.1）
    expect("27B bf16 页组 4.00 MiB", 64 * 4 * 16 * kv_bytes_per_token_head("bf16", 256), 4 * MiB)

    b = plan(manifest, 6144, "bf16", 400, None)
    expect("bf16 Σkv/页组", b["kv_per_group_bytes"], 2_359_296)
    expect("budget 口径 fixed 字节", int(round(b["fixed_mib"] * MiB)), 3_287_000_064)
    expect("6GB bf16 预算口径页组", b["page_groups"], 201 if perturb else 200)
    expect("6GB bf16 预算口径 token", b["tokens"], 12_800)

    i8 = plan(manifest, 6144, "int8_g64", 400, None)
    expect("int8 Σkv/页组", i8["kv_per_group_bytes"], 1_216_512)
    expect("6GB int8 预算口径页组", i8["page_groups"], 388)
    expect("6GB int8 预算口径 token", i8["tokens"], 24_832)

    # 容量口径：2026-11-19 由 T4 实测重标定。T4 两行（headroom 1024/512 → 5323/5579）
    # 联立精确解出 minres = 2732.95 MiB、min_pages = 2、斜率 2.00 MiB（bf16 8 层）。
    # 6 GB 游戏卡无 ECC，扣 400 驱动、默认 headroom 1024、workspace 135。
    m_bf16 = plan(manifest, 6144, "bf16", 400, MEASURED_WORKSPACE_MIB, ecc=False)
    expect("6GB bf16 容量口径页组", m_bf16["page_groups"], 929 if perturb else 928)
    expect("6GB bf16 容量口径 token", m_bf16["tokens"], 59_392)
    m_i8 = plan(manifest, 6144, "int8_g64", 400, MEASURED_WORKSPACE_MIB, ecc=False)
    expect("6GB int8 容量口径 token", m_i8["tokens"], 115_008)
    # 对表闸：T4 两行实测必须逐页命中，否则公式漂了而没人发现
    for hr, want in ((1024, 5323), (512, 5579)):
        t = plan(manifest, 15360, "bf16", 0, 0, ecc=True, headroom_mib=hr)
        expect(f"T4 headroom={hr} 页组", t["page_groups"], want)

    # 结构不变量：**权重 + head 必须 ≤ 文件本身**（state 是运行期分配的，不在文件里，
    # 所以「全常驻」可以比文件大——这正是首版多算 308 MiB 时该撞上的那道墙）
    file_bytes = 2_827_176_704
    expect("权重+head ≤ 文件字节", sum(weights["layers"]) + weights["head"] + weights["mtp"] <= file_bytes, True)
    expect("全常驻 > 文件（state 在运行期分配）", int(round(b["resident_mib"] * MiB)) > file_bytes, True)

    bad = [c for c in checks if not c[3]]
    for label, got, want, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<26} got={got!r:<22} want={want!r}")
    if bad:
        print(f"\n{len(bad)} 项不符", file=sys.stderr)
        return 1
    print(f"\n{len(checks)} 项全部通过")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="从 .ninfer 算显存容量（无 GPU）")
    parser.add_argument("model", nargs="?", help=".ninfer 文件路径")
    parser.add_argument("--vram-mib", type=int, default=6144)
    parser.add_argument("--driver-mib", type=int, default=400)
    parser.add_argument("--workspace-mib", type=int, default=MEASURED_WORKSPACE_MIB)
    parser.add_argument("--kv", default="bf16",
                        help="KV 存储格式，见 KV_DTYPES（默认 bf16）")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--prove-red", action="store_true",
                        help="把一项期望值改 1，确认 selftest 会红（门的非空性反证）")
    args = parser.parse_args(argv)

    if args.prove_red:
        if not args.model:
            parser.error("--prove-red 需要模型路径")
        rc = selftest(args.model, perturb=True)
        if rc != 0:
            print("RED 证明通过：注入的错误期望被 selftest 判红", file=sys.stderr)
            return 0
        print("RED 证明失败：selftest 放过了一个错误期望", file=sys.stderr)
        return 1
    if args.selftest:
        if not args.model:
            parser.error("--selftest 需要模型路径")
        return selftest(args.model)
    if not args.model:
        parser.error("需要模型路径（或 --selftest）")

    manifest, digest, meta = read_manifest(args.model)
    if args.json:
        payload = {
            "digest": digest,
            "meta": meta,
            **{basis: plan(manifest, args.vram_mib, args.kv, args.driver_mib, ws)
               for basis, ws in (("budget", None), ("measured", args.workspace_mib))},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(render(manifest, args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
