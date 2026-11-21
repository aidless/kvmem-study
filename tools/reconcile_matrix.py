#!/usr/bin/env python3
"""对账 kvmem-matrix-sm75 的 result.json 与 remote/matrix-sm75/PREDICTIONS.md。

跑前写死的判据在这里变成断言：D1 合法格全绿、D2 非法格不启进程、D5 host=0 时写计数全 0。
D3/D4/D6 是「观测项」——打印实测值，不给 PASS/FAIL，因为它们没有跑前定下的真值。

用法：
  python3 tools/reconcile_matrix.py remote/matrix-sm75/result.json
  python3 tools/reconcile_matrix.py remote/matrix-sm75/result.json --selftest
"""
import argparse
import json
import sys

# PREDICTIONS.md §三 的判据。断言里的数字逐字复制自预注册，不许从 result 里反推。
LEGAL_EXPECTED = 13          # 主扫描 9 + H 扫描 4
ILLEGAL_EXPECTED = 6         # ctx=8192 两格 + ctx=32768 两格 + ctx=131072 四格（B<ctx）
HOST_ZERO_WRITE_KEYS = ("host_image_writes", "host_block_writes", "host_restore_bytes")

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not ok:
        failures.append(name)
    return ok


def hybrid_of(run):
    stats = (run or {}).get("stats") or {}
    return (stats.get("hybrid") or {}) if stats.get("ok") else {}


def rows_of(doc):
    m = doc.get("matrix") or {}
    return list(m.get("main") or []) + list(m.get("hsweep") or [])


def reconcile(doc):
    rows = rows_of(doc)
    legal = [r for r in rows if "run" in r]
    illegal = [r for r in rows if r.get("status") == "STRUCTURAL_NO"]

    check("D2 非法格数量 == 6", len(illegal) == ILLEGAL_EXPECTED, f"got={len(illegal)}")
    check("D1 合法格数量 == 13", len(legal) == LEGAL_EXPECTED, f"got={len(legal)}")

    # D2 的实质：非法格不许有 run 键（有 run 说明真的起了进程）
    started = [r["tag"] for r in illegal if "run" in r]
    check("D2 非法格未启动进程", not started, f"started={started}")

    # D1 的实质：进程启动过、就绪、且以正常方式结束。
    # ninfer-serve 收到 terminate 后走优雅关闭 (log 有 `server stopped`) 返回 0，
    # 不是 SIGTERM 的 -15——预注册写"returncode 为正常 terminate"时没料到这一点。
    # 所以接受 rc ∈ {0, -15}；关键判据是 alive_after_probe + ready_after_seconds。
    bad = []
    for r in legal:
        run = r["run"]
        ready_s = (run.get("ready") or {}).get("ready_after_seconds")
        if run.get("returncode") not in (0, -15) or not run.get("alive_after_probe") or ready_s is None:
            bad.append({"tag": r["tag"], "rc": run.get("returncode"),
                        "alive": run.get("alive_after_probe"), "ready": ready_s})
    check("D1 合法格全部启动并就绪", not bad, str(bad[:3]))

    # D5：host_mib == 0 时三项写计数必须全 0
    zero_rows = [r for r in legal if r.get("host_mib") == 0]
    if zero_rows:
        offenders = []
        for r in zero_rows:
            h = hybrid_of(r["run"])
            for k in HOST_ZERO_WRITE_KEYS:
                if (h or {}).get(k):
                    offenders.append({"tag": r["tag"], "key": k, "value": h[k]})
        check("D5 host=0 时写计数全 0", not offenders, str(offenders[:3]))
    else:
        print("SKIP  D5 本次矩阵没有 host=0 的格")

    # 观测项：不判定，只打印。D3/D4/D6 在预注册里明确「不预测」。
    print("\n观测项（无跑前真值，只记录）")
    for r in legal:
        h = hybrid_of(r["run"])
        print(f"  {r['tag']:<22} rc={r['run'].get('returncode'):>4} "
              f"ready={(r['run'].get('ready') or {}).get('ready_after_seconds')}s "
              f"hits={h.get('snapshot_hits')} reused={h.get('reused_tokens')} "
              f"host_writes={h.get('host_block_writes')} "
              f"unbacked={h.get('unbacked_node_losses')}")

    host_rows = [r for r in legal if "h-" in r["tag"]]
    if len(host_rows) >= 2:
        vals = [(r["host_mib"], (hybrid_of(r["run"]) or {}).get("reused_tokens")) for r in host_rows]
        monotone = all(a[1] is not None and b[1] is not None and a[1] <= b[1]
                       for a, b in zip(vals, vals[1:]))
        print(f"\nD4 H 扫描复用是否单调递增: {'是' if monotone else '否'}  {vals}")
        print("   预注册说的是「不保证单调」，所以两种结果都算记录成功，不判 FAIL。")

    return not failures


SELFTEST_GOOD = {
    "matrix": {
        "main": [{"tag": "m-b8192-ctx8192", "host_mib": 2048,
                  "run": {"returncode": -15, "alive_after_probe": True, "ready": {"ready": True, "ready_after_seconds": 22.0},
                          "stats": {"ok": True, "hybrid": {"snapshot_hits": 1}}}},
                 {"tag": "m-b8192-ctx131072", "status": "STRUCTURAL_NO",
                  "reason": "kv_capacity must be at least max_context"}],
        "hsweep": [{"tag": "h-0", "host_mib": 0,
                    "run": {"returncode": -15, "alive_after_probe": True, "ready": {"ready": True, "ready_after_seconds": 22.0},
                            "stats": {"ok": True, "hybrid": {k: 0 for k in HOST_ZERO_WRITE_KEYS}}}}],
    }
}


def selftest():
    """两段：先证明会红，再证明放行该放行的。数字逐字复制自预注册。"""
    global failures
    print("=== RED 段：注入违规，必须 FAIL")
    doc = json.loads(json.dumps(SELFTEST_GOOD))
    doc["matrix"]["main"].append({"tag": "m-b16384-ctx131072", "status": "STRUCTURAL_NO",
                                  "run": {"returncode": -15, "alive_after_probe": True}})
    doc["matrix"]["hsweep"][0]["run"]["stats"]["hybrid"]["host_block_writes"] = 7
    failures = []
    reconcile(doc)
    red_ok = {"D2 非法格未启动进程", "D5 host=0 时写计数全 0"} & set(failures)
    print(f"\nRED 证明{'通过' if red_ok else '失败'}：抓到 {sorted(red_ok)}")
    if not red_ok:
        return False

    print("\n=== GREEN 段：注入合规，必须放行（除数量断言，因为 fixture 只有 3 行）")
    doc = json.loads(json.dumps(SELFTEST_GOOD))
    failures = []
    reconcile(doc)
    only_counts = set(failures) <= {"D2 非法格数量 == 6", "D1 合法格数量 == 13"}
    print(f"\nGREEN 证明{'通过' if only_counts else '失败'}：仅数量断言红（fixture 行数不足），"
          f"其余 {sorted(set(failures))}")
    return only_counts


def page_groups_of(run):
    """从启动日志的 `capacity | ... | pages N,M/...` 行解析 KV 页组数。

    格式逐字来自 v5 real: `capacity | KV 340,672 tokens, bf16, auto | pages 5,323/33,554,431 | ...`
    只在 int8 段的无流量行出现；bf16 主扫描走 /stats，不从这里读。
    """
    import re
    log = (run or {}).get("log") or ""
    m = re.search(r"pages\s+([\d,]+)\s*/\s*[\d,]+", log)
    return int(m.group(1).replace(",", "")) if m else None


def reconcile2(doc):
    """第二轮 matrix2 的 D1–D6。判据数字逐字复制自 remote/matrix2-sm75/PREDICTIONS.md。"""
    m = doc.get("matrix") or {}
    main, hsweep, int8 = list(m.get("main") or []), list(m.get("hsweep") or []), list(m.get("int8") or [])

    # D5：bf16 两段行数与第一轮一致（9 合法 + 6 结构 NO + 4 hsweep = 19 行）
    legal_main = [r for r in main if "run" in r]
    illegal_main = [r for r in main if r.get("status") == "STRUCTURAL_NO"]
    bf16_rows = legal_main + hsweep
    check("D5 bf16 合法+hsweep 行数 == 13", len(bf16_rows) == 13,
          f"legal={len(legal_main)} hsweep={len(hsweep)}")
    check("D5 结构 NO 行数 == 6", len(illegal_main) == 6, f"got={len(illegal_main)}")

    # D1：int8 两行活着 + 有 engine_capacity 行（日志里的 `capacity | ... pages`）
    live = [r for r in int8 if r.get("run", {}).get("alive_after_probe")]
    check("D1 int8 两行启动", len(live) == 2 and len(int8) == 2, f"live={len(live)}/{len(int8)}")
    pages = {}
    for r in int8:
        p = page_groups_of(r.get("run"))
        pages[r.get("headroom_mib")] = p
        check(f"D1 {r.get('tag')} 有容量行", p is not None, f"pages={p}")

    # 第一轮 bf16 实测（PREDICTIONS.md 逐字），用于 D2 的「严格大于」判据
    BF16_PAGES = {1024: 5323, 512: 5579}
    for hr, bf16p in BF16_PAGES.items():
        p = pages.get(hr)
        check(f"D2 int8 hr{hr} 页数 > bf16 {bf16p}", p is not None and p > bf16p,
              f"int8={p} bf16={bf16p}")

    # D3：headroom 512 < 1024 预留 → 页数更多
    p1, p5 = pages.get(1024), pages.get(512)
    check("D3 int8 hr512 > hr1024", p1 is not None and p5 is not None and p5 > p1,
          f"hr1024={p1} hr512={p5}")

    # D4：每行 rss 采样器工作（采样点数 >=1 且峰值 > 0）
    bad_rss = []
    for r in bf16_rows + int8:
        rr = (r.get("run") or {}).get("rss") or {}
        if not (rr.get("n", 0) >= 1 and (rr.get("peak_kib") or 0) > 0):
            bad_rss.append({"tag": r.get("tag"), "rss": rr})
    check("D4 每行 RSS 采样有效", not bad_rss, str(bad_rss[:3]))

    # 观测项：量级 sanity（不当判据）+ RSS 单调性（D6 不预测）+ D1 int8 详细
    print("\n观测项（无跑前真值，只记录）")
    for hr in (1024, 512):
        print(f"  int8 hr{hr}: pages={pages.get(hr)} （预注册量级 ~{10544 if hr==1024 else 11040}，±2% 不硬判）")
    host_rss = [(r.get("host_mib"), (r.get("run") or {}).get("rss", {}).get("peak_mib"))
                for r in hsweep]
    print(f"  D6 RSS 跨 host_mib: {host_rss} （预注册不预测单调性）")
    return not failures


def _rss(n=3, peak=4000000):
    return {"n": n, "peak_kib": peak, "peak_mib": round(peak / 1024, 1)}


def _fixture2():
    """合规 fixture：9 合法 + 6 结构 NO + 4 hsweep + 2 int8，页数用预注册量级值。"""
    doc = {"matrix": {"main": [], "hsweep": [], "int8": []}}
    for i in range(9):
        doc["matrix"]["main"].append({"tag": f"m{i}", "host_mib": 2048,
                                      "run": {"alive_after_probe": True, "rss": _rss()}})
    for i in range(6):
        doc["matrix"]["main"].append({"tag": f"x{i}", "status": "STRUCTURAL_NO"})
    for i in range(4):
        doc["matrix"]["hsweep"].append({"tag": f"h{i}", "host_mib": 64 * (i + 1),
                                        "run": {"alive_after_probe": True, "rss": _rss()}})
    for hr, p in ((1024, 10544), (512, 11040)):
        doc["matrix"]["int8"].append({"tag": f"int8-hr{hr}", "headroom_mib": hr,
                                      "run": {"alive_after_probe": True, "rss": _rss(),
                                              "log": f"INFO  capacity | KV 675,000 tokens, int8_g64, auto | pages {p:,}/33,554,431 |"}})
    return doc


def selftest2():
    """round2 判据 RED→GREEN：注入页数不够/无容量行/RSS 失效必须红，合规 fixture 放行。"""
    global failures
    print("=== RED 段：注入违规，round2 必须 FAIL")
    doc = _fixture2()
    doc["matrix"]["int8"][0]["run"]["log"] = "capacity | pages 5,000/33,554,431"   # < bf16 5323
    doc["matrix"]["int8"][1]["run"]["log"] = "no capacity line"                      # 无容量行
    doc["matrix"]["hsweep"][0]["run"]["rss"] = {"n": 0, "peak_kib": None}
    failures = []
    reconcile2(doc)
    red = {"D2 int8 hr1024 页数 > bf16 5323", "D1 int8-hr512 有容量行",
           "D4 每行 RSS 采样有效"} & set(failures)
    print(f"\nRED 证明{'通过' if len(red) == 3 else '失败'}：抓到 {sorted(red)}")
    if len(red) != 3:
        return False

    print("\n=== GREEN 段：合规 fixture，round2 必须全绿")
    failures = []
    ok = reconcile2(_fixture2())
    print(f"\nGREEN 证明{'通过' if ok else '失败'}：failures={failures}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--round2", action="store_true", help="用 matrix2 的 D1–D6 判据")
    a = ap.parse_args()
    if a.selftest:
        return 0 if (selftest2() if a.round2 else selftest()) else 1
    if not a.result:
        print("需要 result.json 路径，或 --selftest")
        return 2
    doc = json.load(open(a.result, encoding="utf-8"))
    ok = reconcile2(doc) if a.round2 else reconcile(doc)
    print(f"\n{'全部判据通过' if ok else '失败项: ' + str(failures)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
