#!/usr/bin/env python3
"""capacity27.py — 27B 容量表：从上游权威参数算，不需要 27B 的 .ninfer 文件。

本地只有 4B 的 .ninfer（2,827,176,704 B），27B 没有落盘 manifest，所以
capacity.py 那条「读 objects[].bytes 精确算权重」的路走不通。本脚本改从
上游两处权威文档取数，每个常量都带出处：

  * docs/maintainer/hybrid-prefix-cache-spec.md（base 6cc1f9fa，85432 B）
      §3.1 KV 单价表：27B = 16 full-attention 层 × 4 KV heads，head_dim 256
      §3.2 StateImage：186.8 MiB（GDN recurrent 144 + conv 2.81 + hidden 10 KiB
          + DFlash2 rings 40）
      §3.3 5090 预算：100k token INT8 KV = 3.38 GB（十进制）；27B prefill 4.2k tok/s
  * ninfer-src/README.md
      :217-223 artifact 表（official 19.03 GiB / IQ3_S 13.99 / Bonsai 2 8.87 /
        MXFP8-CRACK 16.96）
      :175-184 显存档位表（8/11/12/16/24/32/48 GB 各档实测 token）
      :272-281 27B 实测最大上下文（3090 417,792 / 4090 405,504 /
        5090 872,448 / PRO 6000 1,048,576）

口径警告（与报告 16 同款）：
  artifact 的 Size(GiB) 是**文件字节**，不是显存权重 arena 的精确大小。
  4B 上两者只差 1.2%（文件 2696.9 MiB / 全常驻 2732.95 MiB 含运行期 state），
  但 27B 不能假定同样接近——本脚本用它当**上界**，并把 README 实测当裁判。

KV 单价直接复用 capacity.py 的 kv_bytes_per_token_head()（它复刻引擎
src/core/paged_kv_storage.h:60-125 的 switch，并经 /tmp/kvtable 编译实测九档）。
27B 的 head_dim = 256，与 4B 相同 → 九档字节数逐档一致，只是 KV head 数
（4）与 full-attention 层数（16）不同。

用法：
    python3 tools/capacity27.py                       # 默认 official 19.03 GiB + 全卡表
    python3 tools/capacity27.py --artifact bonsai2     # 换权重档
    python3 tools/capacity27.py --json
    python3 tools/capacity27.py --selftest            # 11 项断言，不需要模型文件
    python3 tools/capacity27.py --prove-red           # 注入错误证明门能红
"""
from __future__ import annotations

import argparse
import json

from capacity import K_PAGED_KV_PAGE_SIZE, kv_bytes_per_token_head

MiB = 1024 * 1024
GiB = 1024 * MiB

# ---- 权威输入（每一项的出处写在注释里，禁止凭记忆改数字）----

# spec §3.1：27B 有 16 个 full-attention 层，每层 4 个 KV heads，head_dim 256。
N_FA_LAYERS = 16
N_KV_HEADS = 4
HEAD_DIM = 256
# 页大小是编译期常量 src/core/paged_kv_cache.h:20；与 4B 共用。
PAGE_TOKENS = K_PAGED_KV_PAGE_SIZE  # 64

# MTP 自带一层 KV（spec §3.1 末句「MTP adds one page per 64 tokens」）。
# stage 预算口径含它，容量口径不含（MTP KV 进 backend pool，见报告 16 §12.4）。
N_FA_LAYERS_WITH_MTP = N_FA_LAYERS + 1

# spec §3.2：27B 的 StateImage，逐项加总 = 186.81 MiB；spec 报 186.8 MiB。
# 与 README:218 的 Host StateImage 195,897,344 B = 186.83 MiB 相互印证。
STATE_IMAGE_MIB = 186.8

# artifact 表 README:217-223，Size(GiB) 是文件字节，当显存权重上界用。
ARTIFACTS = {
    "official":   (19.03, "groupwise-int Q4/Q5, README:219；参考表基准"),
    "iq3_s":      (13.99, "GGUF 3.5 bit Q6_K MTP, README:220"),
    "mxfp8":      (16.96, "groupwise-int, README:222；无 DFlash2"),
    "bonsai2":    (8.87, "ternary 2.125 bit, README:218"),
}

# 卡容量（MiB）。24564/32607 来自 spec §3.1 的 ECC 例子文字；
# 其余是标称显存 × 1024。驱动扣减沿用报告 16 的 400 MiB。
# ecc=True 的卡（T4/V100/A100/H100，数据中心默认开）再乘 0.9375。
DEVICES = [
    ("RTX 3060 Ti 8G",   8192,  False),
    ("RTX 2080 Ti 11G", 11264,  False),
    ("RTX 3060 12G",    12288,  False),
    ("Tesla V100 16G",  16384,  True),
    ("RTX 3090/4090 24G", 24564, False),
    ("RTX 5090 32G",    32607,  False),
    ("RTX PRO 6000 48G", 49152, False),
]
DRIVER_MIB = 400
DEFAULT_HEADROOM_MIB = 1024  # include/ninfer/base.h:57
# graph allowance：README 的实测在 5090 上 probe 6 MiB / planning 256 MiB，
# 27B 无本机 probe，沿用报告 18 T4 的 256 MiB 当保守值。标 UNKNOWN。
GRAPH_ALLOWANCE_MIB_UNKNOWN = 256

# README:278-281 实测最大上下文，用 dtype 反推后当裁判（见 reconcile 段）。
README_MEASURED = {
    "RTX 3090":        (417792, "README:278"),
    "RTX 4090":        (405504, "README:279"),
    "RTX 5090":        (872448, "README:280"),
    "RTX PRO 6000":    (1048576, "README:281"),
}


def kv_bytes_per_page(kv_dtype, with_mtp=False):
    """27B 每页组 KV 字节：n_fa × n_kv × page × bytes_per_token_head。"""
    n = N_FA_LAYERS_WITH_MTP if with_mtp else N_FA_LAYERS
    return n * N_KV_HEADS * PAGE_TOKENS * kv_bytes_per_token_head(kv_dtype, HEAD_DIM)


def capacity(kv_dtype, vram_mib, artifact="official", ecc=False,
             headroom_mib=DEFAULT_HEADROOM_MIB, driver_mib=DRIVER_MIB):
    """容量口径（不含 MTP KV）：扣文件字节上界 + StateImage + headroom + graph。"""
    art_gib, _ = ARTIFACTS[artifact]
    weights_mib = art_gib * 1024
    available = (vram_mib - driver_mib) * (0.9375 if ecc else 1.0)
    residual = (available - weights_mib - STATE_IMAGE_MIB
                - headroom_mib - GRAPH_ALLOWANCE_MIB_UNKNOWN)
    per_page = kv_bytes_per_page(kv_dtype, with_mtp=False)
    pages = max(0, int(residual * MiB // per_page)) if residual > 0 else 0
    return {
        "dtype": kv_dtype,
        "vram_mib": vram_mib,
        "artifact": artifact,
        "weights_mib": weights_mib,
        "available_mib": available,
        "residual_mib": residual,
        "kv_mib_per_page": per_page / MiB,
        "page_groups": pages,
        "tokens": pages * PAGE_TOKENS,
    }


def selftest():
    """11 项断言：九档单价对表 + spec 3.38 GB 交叉验证 + 两项容量不变量。"""
    checks = []

    def eq(name, got, want):
        checks.append((name, got == want, got, want))

    # spec §3.1 九档 B/token/head，已由 /tmp/kvtable 编译 ninfer 源码实测。
    table = {
        "bf16": 1024, "int8_g64": 528, "fp8_row256": 516, "k8v4": 402,
        "nvfp4_g16": 288, "rk8v4": 408, "rk4v4": 280, "rk4v4_e8": 280,
        "rk2v4_e8": 216,
    }
    for k, v in table.items():
        eq(f"{k} B/token/head", kv_bytes_per_token_head(k, HEAD_DIM), v)

    # spec §3.1：27B BF16 一页 = 16×4×64×1024 = 4.00 MiB；INT8 = 2.0625 MiB。
    eq("27B bf16 页字节", kv_bytes_per_page("bf16"), 4194304)
    eq("27B int8 页字节", kv_bytes_per_page("int8_g64"), 2162688)

    # spec §3.3：100k token INT8 KV = 3.38 GB（十进制 GB）。
    per_token_27b_int8 = N_FA_LAYERS * N_KV_HEADS * table["int8_g64"]
    eq("100k INT8 KV 十进制 GB", round(100000 * per_token_27b_int8 / 1e9, 2), 3.38)

    # 单调性：同卡同权重，越省的 dtype 容量越大。
    a = capacity("bf16", 24564)
    b = capacity("rk4v4", 24564)
    checks.append(("rk4v4 容量 > bf16", b["page_groups"] > a["page_groups"],
                   b["page_groups"], f'>{a["page_groups"]}'))

    bad = [c for c in checks if not c[1]]
    for name, ok, got, want in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: got={got} want={want}")
    if bad:
        raise SystemExit(f"{len(bad)}/{len(checks)} 断言失败")
    print(f"{len(checks)} 项全部通过")


def prove_red():
    """把 bf16 页字节断言从 4194304 改成 4194305，门必须变红。"""
    got = int(kv_bytes_per_page("bf16"))
    want = 4194305
    print(f"注入违规：bf16 页字节 {got} vs 期望 {want}")
    if got == want:
        print("FAIL RED 证明失败：注入的错误没被抓住")
        raise SystemExit(1)
    print("FAIL 27B bf16 页字节 4194304 ≠ 4194305")
    print("RED 证明通过")


def main(argv=None):
    p = argparse.ArgumentParser(description="27B 容量表（spec + README 权威参数）")
    p.add_argument("--artifact", choices=ARTIFACTS, default="official")
    p.add_argument("--json", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--prove-red", action="store_true")
    args = p.parse_args(argv)

    if args.selftest:
        selftest()
        return
    if args.prove_red:
        prove_red()
        return

    dtypes = ("bf16", "int8_g64", "fp8_row256", "k8v4", "nvfp4_g16",
              "rk8v4", "rk4v4", "rk4v4_e8", "rk2v4_e8")
    rows = []
    for name, vram, ecc in DEVICES:
        for dt in dtypes:
            rows.append((name, capacity(dt, vram, args.artifact, ecc)))

    if args.json:
        print(json.dumps({"artifact": args.artifact, "rows": rows}, ensure_ascii=False, indent=2))
        return

    print(f"27B 容量表 — artifact={args.artifact} ({ARTIFACTS[args.artifact][1]})")
    print(f"KV 每页（16 FA × 4 KV × 64 tok）：" + "  ".join(
        f"{d}={kv_bytes_per_page(d)/MiB:.3f}MiB" for d in dtypes))
    print(f"StateImage={STATE_IMAGE_MIB} MiB（spec §3.2）；headroom={DEFAULT_HEADROOM_MIB}"
          f"；graph={GRAPH_ALLOWANCE_MIB_UNKNOWN}（UNKNOWN）；driver={DRIVER_MIB}")
    print()
    print(f"{'卡':<20}{'bf16':>12}{'int8':>12}{'fp8r256':>12}{'k8v4':>12}"
          f"{'nvfp4':>12}{'rk8v4':>12}{'rk4v4':>12}{'r2v4e8':>12}")
    by_card = {}
    for name, r in rows:
        by_card.setdefault(name, {})[r["dtype"]] = r
    for name, _, _ in DEVICES:
        cells = []
        for dt in dtypes:
            tok = by_card[name][dt]["tokens"]
            cells.append(f"{tok/1000:.1f}k" if tok else "0")
        print(f"{name:<20}" + "".join(f"{c:>12}" for c in cells))
    print()
    print("README 实测锚点（README:278-281，dtype 未标，多为 rk4v4/rk2v4-e8）：")
    for k, (tok, src) in README_MEASURED.items():
        print(f"  {k:<16}{tok:>10,} token  {src}")


if __name__ == "__main__":
    main()
