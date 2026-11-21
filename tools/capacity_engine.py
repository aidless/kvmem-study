#!/usr/bin/env python3
"""capacity_engine.py — 引擎口径容量计算器：复现 kv_capacity.cpp 的 pages 公式。

与 capacity.py 的区别（两套口径不可混用，见报告 16 §12.1.1）：
  capacity.py        : pages = MIN_MAIN_PAGE_GROUPS(简化加数 2) + floor(residual/斜率)
                       residual = 标称 − driver − headroom − fixed − 权重 − state
                       输入是 .ninfer manifest 的逐张量字节，回答「这张卡装得下多少」。
  capacity_engine.py : pages = m + floor((available − headroom − minres)/P)
                       这是引擎 resolve_kv_capacity 的原式（kv_capacity.cpp:115-133），
                       m / minres / P 由 T4 实测反解钉死，回答「引擎实际会报到多少页」。

权威出处：
  m = 128        = page_count(max_context=8192) + resident_main_pages(0)
                   （startup.cpp:1803-1807, serve_options.h:46/:49, startup.cpp:79-82；
                    报告 16 §12.1.1 用 v5 explicit 行 32768→512 页证 resident=0）
  minres = 2986  MiB，T4 两行 headroom 反解，差 0.0000（报告 18 §3.1、报告 16 §12.1.1）
  P          = 斜率 = bytes_per_additional_main_page_group（startup.cpp:1832-1837）
               T4 bf16 实测 2.00 MiB/页（8 层，不含 MTP，MTP 是 backend pool）

用法：
    python3 tools/capacity_engine.py --selftest
    python3 tools/capacity_engine.py --vram-mib 6144 --kv bf16        # 6 GB 引擎口径
    python3 tools/capacity_engine.py --vram-mib 15360 --ecc --kv int8 # T4
    python3 tools/capacity_engine.py --json
    python3 tools/capacity_engine.py --prove-red                      # 门的非空性反证
"""
from __future__ import annotations

import argparse
import json
import sys

MiB = 1 << 20

# 引擎结构常量（T4 反解钉死，见模块 docstring 出处）
M_MAIN_PAGE_GROUPS = 128          # startup.cpp:1803-1807
PAGE_TOKENS = 64                  # src/core/paged_kv_cache.h:20
ECC_FACTOR = 0.9375               # 数据中心卡默认 ECC，报告 18 日志 warning
DEFAULT_HEADROOM_MIB = 1024       # include/ninfer/base.h:57

# minres = persistent + workspace + graph（含 CUDA graph planning allowance 256 MiB），
# **不含权重**，且 **随 KV dtype 变**——2026-11-20 的 v3 内核实测发现：
#   bf16 2986.00（T4 两行反解，差 0.0000）
#   int8 ≈ 2889   （v3 两行反解 2889.22 / 2888.69，取整数 2889 同时命中两行）
# 差的 97 MiB 成因未定位（`mtp_extra_pages` 随 dtype 缩放是候选，startup.cpp:146），
# 按 AGENTS.local 第 13 条不再开第三个反解实验。其余 dtype 无实测，按 bf16 近似并标 UNKNOWN。
MINRES_BY_DTYPE = {               # 键 = 有实测的 dtype
    "bf16": 2986.0,
    # int8：斜率 P 是精确值（8 层 × 4 heads × 64 token × 528 B = 1,081,344 B = 1.03125 MiB，
    # 所以两行只剩 minres 一个未知数）。两行各给一个半开区间，交集是 (2888.19, 2888.69]，
    # 宽 0.50 MiB，取中点 2888.44。区间内的任何值都同时精确命中 10297 与 10794。
    "int8": 2888.44,
}
MINRES_MIB = 2986.0               # bf16 基准；未实测的 dtype 沿用它并在结果里标 minres_measured=False

# 斜率 P：T4 实测 2.00 MiB/页组（bf16）。其他档按报告 16 第 5 节的 8 层容量口径单价换算：
#   P(dtype) = 8 层 × 4 KV heads × 64 token × B/token/head ÷ MiB
#   bf16 是反解基准 2.00；其余用与 bf16 的字节比缩放（capacity.py:kv_bytes_per_token_head 实测九档）。
B_PER_TOKEN_HEAD = {
    "bf16": 1024, "int8": 528, "fp8": 516, "k8v4": 402, "nvfp4": 288,
    "rk8v4": 408, "rk4v4": 280, "rk4v4_e8": 280, "rk2v4_e8": 216,
}
BF16_B = B_PER_TOKEN_HEAD["bf16"]
P_BF16_MIB = 2.00                 # T4 反解基准，不用公式重算（避免又引入一个口径）


def slope_mib(kv_dtype):
    """P(dtype) = P_bf16 × B(dtype)/B(bf16)。bf16 走精确反解值 2.00。"""
    if kv_dtype not in B_PER_TOKEN_HEAD:
        raise ValueError(f"未知 kv dtype {kv_dtype}，可用 {'/'.join(B_PER_TOKEN_HEAD)}")
    if kv_dtype == "bf16":
        return P_BF16_MIB
    # 容量口径 8 层：字节比缩放，基准也用 2.00 而非 bf16 原始字节（保证 bf16 自洽）
    return round(P_BF16_MIB * B_PER_TOKEN_HEAD[kv_dtype] / BF16_B, 6)


def default_minres(kv_dtype):
    """minres 随 dtype 变：有实测的用实测值，没有的沿用 bf16 并标未测量。"""
    return MINRES_BY_DTYPE.get(kv_dtype, MINRES_MIB)


def minres_is_measured(kv_dtype):
    return kv_dtype in MINRES_BY_DTYPE


def pages(vram_mib, headroom_mib=DEFAULT_HEADROOM_MIB, kv_dtype="bf16",
          ecc=False, m=M_MAIN_PAGE_GROUPS, minres_override=None):
    """引擎 resolve_kv_capacity 原式。available 已含权重加载后的余额，不再单扣权重。

    minres_override 省略时按 dtype 取实测值；显式给出则覆盖（反解/敏感性实验用）。
    """
    if kv_dtype not in B_PER_TOKEN_HEAD:
        raise ValueError(f"未知 kv dtype {kv_dtype}")
    minres = default_minres(kv_dtype) if minres_override is None else minres_override
    available = vram_mib * (ECC_FACTOR if ecc else 1.0)
    P = slope_mib(kv_dtype)
    residual = available - headroom_mib - minres
    n = m + int(residual // P) if residual > 0 else m
    return {
        "vram_mib": vram_mib, "ecc": ecc, "headroom_mib": headroom_mib,
        "kv_dtype": kv_dtype, "available_mib": available, "minres_mib": minres,
        "minres_measured": minres_is_measured(kv_dtype),
        "m": m, "slope_mib": P, "residual_mib": residual,
        "page_groups": n, "tokens": n * PAGE_TOKENS,
    }


def selftest(perturb=False):
    """判据全部来自 T4 实测（报告 18）与源码字面值，不来自本脚本自己的输出。"""
    checks = []
    def expect(label, got, want):
        checks.append((label, got, want, got == want))

    # T4 两行：vram 15360 + ECC，bf16。m=128 / minres=2986 / P=2.00 反解钉死。
    a = pages(15360, 1024, "bf16", ecc=True)
    b = pages(15360, 512, "bf16", ecc=True)
    delta = 1 if perturb else 0
    expect("T4 hr1024 页组", a["page_groups"], 5323 + delta)
    expect("T4 hr512 页组", b["page_groups"], 5579)
    expect("T4 两档差（差判据）", b["page_groups"] - a["page_groups"], 256)
    expect("T4 hr1024 token", a["tokens"], (5323 + delta) * 64)
    expect("斜率 P bf16", slope_mib("bf16"), 2.00)
    expect("斜率 P int8", slope_mib("int8"), 1.03125)      # 528/1024×2 = 1.03125
    expect("斜率 P nvfp4", slope_mib("nvfp4"), 0.5625)     # 288/1024×2 = 0.5625
    expect("m 字面 = page_count(8192)", M_MAIN_PAGE_GROUPS, 128)
    expect("页大小常量", PAGE_TOKENS, 64)

    # int8 单调：同 headroom 下页组必须 > bf16（斜率更小）
    i8 = pages(15360, 1024, "int8", ecc=True)
    expect("int8 > bf16（单调）", i8["page_groups"] > a["page_groups"], True)
    expect("int8 hr512 > hr1024", pages(15360, 512, "int8", ecc=True)["page_groups"] > i8["page_groups"], True)

    # int8 实测（v3 内核，2026-11-20）：判据数字逐字复制自 result.json 的启动日志，
    # 不是本脚本的输出。斜率用理论 1.03125，minres 用区间中点 2888.44。
    expect("T4 int8 hr1024 页组（实测）", i8["page_groups"], 10297 + delta)
    expect("T4 int8 hr512 页组（实测）", pages(15360, 512, "int8", ecc=True)["page_groups"], 10794)
    expect("T4 int8 hr1024 token（实测）", i8["tokens"], (10297 + delta) * 64)
    expect("int8 minres 与 bf16 差 ≈ 98 MiB", round(2986 - default_minres("int8")), 98)
    expect("int8 minres 已实测标记", i8["minres_measured"], True)
    expect("nvfp4 minres 未实测标记", pages(15360, 1024, "nvfp4", ecc=True)["minres_measured"], False)

    # 6 GB：没有引擎实测（本机 3060 Laptop 未跑引擎），所以判据从公式字面值独立手算：
    #   residual = 6144 − 2986 − 1024 = 2134；2134 // 2.00 = 1067；+ m 128 = 1195 页
    # 这验的是「脚本正确实现公式」，不是「脚本复现引擎实测」——后者只有 T4 两行有。
    g = pages(6144, 1024, "bf16", ecc=False)
    expect("6GB 公式手算页组", g["page_groups"], 1195 + delta)
    expect("6GB 公式手算 token", g["tokens"], (1195 + delta) * 64)

    bad = [c for c in checks if not c[3]]
    for label, got, want, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label:<28} got={got!r:<12} want={want!r}")
    if bad:
        print(f"\n{len(bad)} 项不符", file=sys.stderr)
        return 1
    print(f"\n{len(checks)} 项全部通过")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="引擎口径容量计算器（复现 kv_capacity.cpp）")
    p.add_argument("--vram-mib", type=int, default=6144)
    p.add_argument("--headroom-mib", type=int, default=DEFAULT_HEADROOM_MIB)
    p.add_argument("--kv", default="bf16")
    p.add_argument("--ecc", action="store_true", help="数据中心卡 ECC（×0.9375）")
    p.add_argument("--json", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--prove-red", action="store_true")
    a = p.parse_args()
    if a.selftest or a.prove_red:
        rc = selftest(perturb=a.prove_red)
        if a.prove_red:
            if rc != 0:
                print("RED 证明通过：注入页组 +1 被 selftest 判红", file=sys.stderr)
                return 0
            print("RED 证明失败：门放过了错误期望", file=sys.stderr)
            return 1
        return rc
    try:
        r = pages(a.vram_mib, a.headroom_mib, a.kv, a.ecc)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(f"{r['vram_mib']} MiB" + (" (ECC)" if r["ecc"] else "") +
              f" / {r['kv_dtype']} / headroom {r['headroom_mib']} MiB → "
              f"{r['page_groups']} 页组 = {r['tokens']:,} token")
        print(f"  available {r['available_mib']:.2f} − minres {r['minres_mib']} − hr {r['headroom_mib']}"
              f" = residual {r['residual_mib']:.2f} ÷ P {r['slope_mib']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
