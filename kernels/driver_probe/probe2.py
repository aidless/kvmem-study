"""Locate the CUDA driver library on a Kaggle GPU image, and prove whether a link against it works.

Why: `kvmem-scan-sm75` failed at stage `cuda_driver` — all nine probed directories were empty
(`/usr/local/cuda/lib64`, `.../lib64/stubs`, `/usr/lib/x86_64-linux-gnu`, `/usr/lib64`, `/usr/lib`, ...).
`nvidia-smi` works, so a driver library must exist somewhere; this probe finds where, without
spending a two-hour build on a guess.

Decisive tests, in order of strength:
  1. `ctypes.CDLL("libcuda.so.1")` actually loads  -> the loader can find it at runtime
  2. `/proc/self/maps` after the load              -> its real absolute path
  3. `nvcc -o t t.cu -lcuda` then run on the GPU   -> a link against it produces a working binary
Static `find` alone is only a hypothesis source; the run is the proof.
"""

import ctypes
import glob
import json
import os
import subprocess

OUT = "/kaggle/working"


def sh(cmd, timeout=120, env=None):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return {"cmd": cmd, "rc": p.returncode,
                "out": p.stdout.strip()[:4000], "err": p.stderr.strip()[:1500]}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "rc": "timeout", "out": "", "err": ""}


def find_driver():
    dirs = ["/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/local/nvidia",
            "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib",
            "/usr/local/cuda/lib64/stubs", "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
            "/opt/conda/lib", "/usr/local/lib"]
    listing = {}
    for d in dirs:
        listing[d] = sorted(os.path.basename(p) for p in glob.glob(d + "/libcuda*")
                            + glob.glob(d + "/libnvidia-ml*"))
    found = sh("find / -xdev -name 'libcuda*' -o -xdev -name 'libnvidia-ml*' 2>/dev/null | grep -v '^/proc' | head -40")
    ldconfig = sh("ldconfig -p 2>/dev/null | grep -i -E 'libcuda|libnvidia-ml' | head -20")
    return {"dirs": listing, "find": found, "ldconfig": ldconfig}


def ctypes_load():
    """The runtime test: can the loader open libcuda.so.1, and from where?"""
    result = {"loaded": False, "handle": None, "maps": []}
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            result.setdefault("errors", []).append(f"{name}: {exc}")
            continue
        result["loaded"] = True
        result["handle"] = name
        try:
            # cuInit(0) is the cheapest real driver call.
            rc = lib.cuInit(0)
            result["cuInit_rc"] = rc
            dev = ctypes.c_int(0)
            result["cuDeviceGet_rc"] = lib.cuDeviceGet(ctypes.byref(dev), 0)
            name_buf = ctypes.create_string_buffer(128)
            result["cuDeviceGetName_rc"] = lib.cuDeviceGetName(name_buf, 128, dev)
            result["device_name"] = name_buf.value.decode(errors="replace")
        except Exception as exc:  # noqa: BLE001 - probe reports whatever happens
            result.setdefault("call_errors", []).append(repr(exc))
        break
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
            result["maps"] = sorted({line.split()[-1] for line in handle
                                     if "libcuda" in line or "libnvidia-ml" in line})[:10]
    except OSError:
        pass
    return result


CU = r"""
#include <cstdio>
#include <cuda.h>
int main() {
    CUresult r = cuInit(0);
    if (r != CUDA_SUCCESS) { std::printf("cuInit failed %d\n", (int)r); return 1; }
    int n = 0;
    cuDeviceGetCount(&n);
    CUdevice d;
    cuDeviceGet(&d, 0);
    char name[128] = {0};
    cuDeviceGetName(name, sizeof(name), d);
    std::printf("driver ok: %d device(s), first=%s\n", n, name);
    return 0;
}
"""


def link_test(stub_dir):
    """Prove a link against -lcuda produces a binary that runs."""
    src = "/tmp/work/driver_probe.cu"
    os.makedirs("/tmp/work", exist_ok=True)
    with open(src, "w", encoding="utf-8") as handle:
        handle.write(CU)
    env = dict(os.environ)
    if stub_dir:
        env["LIBRARY_PATH"] = stub_dir + ":" + env.get("LIBRARY_PATH", "")
    build = sh(f"nvcc -arch=sm_75 -o /tmp/work/driver_probe {src} -lcuda 2>&1 | tail -20", env=env)
    run = sh("/tmp/work/driver_probe 2>&1 | tail -5", env=env)
    return {"stub_dir": stub_dir, "build": build, "run": run}


def main():
    out = {"driver": find_driver(), "ctypes": ctypes_load()}
    # If the loader found it, point the linker at that directory via a symlink farm.
    maps = out["ctypes"].get("maps") or []
    stub_dir = None
    if maps:
        real = maps[0]
        stub_dir = "/tmp/work/cuda-stub"
        os.makedirs(stub_dir, exist_ok=True)
        link = os.path.join(stub_dir, "libcuda.so")
        if not os.path.exists(link):
            os.symlink(real, link)
        out["symlink"] = {"from": real, "to": link}
    out["link_test"] = link_test(stub_dir)
    with open(os.path.join(OUT, "driver_probe.json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2)[:6000])


if __name__ == "__main__":
    main()
