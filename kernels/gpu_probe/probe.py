import os, shutil, subprocess, json, sys

def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return (out.stdout or "") + (out.stderr or "")
    except Exception as exc:
        return f"<{exc}>"

info = {
    "nvidia_smi_L": run("nvidia-smi -L"),
    "nvidia_smi_query": run("nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv"),
    "nvcc": run("nvcc --version"),
    "cuda_home": run("ls -d /usr/local/cuda* 2>/dev/null"),
    "cpu": run("nproc"),
    "mem_gb": run("free -g | head -2"),
    "disk": run("df -h /kaggle/working | tail -1"),
}
for k, v in info.items():
    print(f"--- {k} ---\n{v.strip()}\n")

# 关键判据：vm 里能不能真的编译一个 sm_75/sm_80 的 CUDA 小样例
src = '/tmp/t.cu'
open(src, 'w').write('''
#include <cstdio>
__global__ void k(){ printf("gpu ok\\n"); }
int main(){ k<<<1,1>>>(); cudaDeviceSynchronize(); return 0; }
''')
for arch in ("sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_120a"):
    out = run(f"nvcc -arch={arch} -o /tmp/t_{arch} {src} && /tmp/t_{arch}")
    print(f"compile_run[{arch}]: {'OK' if 'gpu ok' in out else 'FAIL'}")
    if 'gpu ok' not in out:
        print("   ", out.strip().replace("\n", " ")[:300])

json.dump(info, open('/kaggle/working/gpu_probe.json', 'w'), indent=1)
print("DONE")
