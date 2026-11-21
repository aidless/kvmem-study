#!/usr/bin/env python3
"""md 体检：检查标题空格、行数、节数、path:line 引用格式、代码块配对。"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "kvmem-study/reports/22-core-chains.md"
with open(path, encoding="utf-8") as f:
    lines = f.read().split("\n")

errors = []
warnings = []
stats = {"total": len(lines), "h1": 0, "h2": 0, "h3": 0, "path_lines": 0, "code_blocks": 0}

in_code = False
for i, line in enumerate(lines, 1):
    # 代码块围栏配对
    if line.strip().startswith("```"):
        stats["code_blocks"] += 1
        in_code = not in_code
        continue
    if in_code:
        continue
    # 标题：取前导井号串，要求其后是空格或行尾（正则陷阱：不能用 ^#+\S）
    m = re.match(r'^(#+)(.*)$', line)
    if m:
        hashes, rest = m.group(1), m.group(2)
        if rest and not rest[0].isspace():
            errors.append(f"L{i}: 标题井号后缺空格: {line[:60]}")
        if len(hashes) == 1:
            stats["h1"] += 1
        elif len(hashes) == 2:
            stats["h2"] += 1
        elif len(hashes) == 3:
            stats["h3"] += 1
    # path:line 引用统计（path.ext:NN 形式）
    for _ in re.finditer(r'[a-zA-Z_][a-zA-Z0-9_./-]*\.(?:cpp|h|md|py|json|ts):\d+', line):
        stats["path_lines"] += 1

if in_code:
    errors.append("代码块围栏未配对（奇数个 ```）")

# 报告约束
if stats["total"] < 250 or stats["total"] > 350:
    warnings.append(f"行数 {stats['total']} 不在 250-350 区间")
if stats["h2"] < 8 or stats["h2"] > 14:
    warnings.append(f"二级节数 {stats['h2']} 不在 8-14 区间")
if stats["path_lines"] == 0:
    errors.append("没有找到任何 path:line 引用")

print(f"文件: {path}")
print(f"行数: {stats['total']}  H1={stats['h1']} H2={stats['h2']} H3={stats['h3']}")
print(f"path:line 引用: {stats['path_lines']} 处")
print(f"代码块围栏: {stats['code_blocks']} 个（配对{'正确' if not in_code else '错误'}）")
if warnings:
    print("WARN:")
    for w in warnings:
        print(f"  - {w}")
if errors:
    print("ERROR:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
print("PASS: 体检通过")