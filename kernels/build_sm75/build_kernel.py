"""Phase B build kernel: compile NInfer for Turing sm_75 on Kaggle's 2x Tesla T4.

Why this kernel exists
----------------------
Phase B (build + scan) has been blocked since 2026-11-10 because every ssh GPU host died.
The Kaggle GPU probe (<kaggle-user>/kvmem-gpu-probe, version 1) settled the question:

  Tesla T4 x2, 15360 MiB each, compute_cap 7.5, driver 580.178.04, CUDA 13.0.88,
  4 CPU, 31 GB RAM, 20 GB /kaggle/working.

and the arch compile loop in that probe reported exactly one OK:

  compile_run[sm_75]: OK        <- this card
  compile_run[sm_80]: FAIL      <- no kernel image for this GPU (expected: it is sm_75)
  compile_run[sm_86]: FAIL
  compile_run[sm_89]: FAIL
  compile_run[sm_90]: FAIL
  compile_run[sm_120a]: FAIL

sm_75 is an officially supported NInfer target, not a long shot:
  - CMakeLists.txt:32 accepts 70|75|80|86|89|120a
  - CMakeLists.txt:45 sets NINFER_ARCH_TURING for 75
  - src/models/qwen3_5/program/planning/startup.cpp:1394-1398 admits capability 75
    when the build carries the Turing image, else throws
  - docs/build-linux.md:128 "Every model, KV format and drafter runs."
  - docs/build-linux.md:119-124 Turing has FP16+INT8 tensor cores, no BF16/TF32,
    no cp.async, 64 KiB shared memory per block; BF16 products run on FP16 tensor
    cores with no value narrowed.

Only documented Turing restriction: NINFER_LINEAR_ROUTES=unified is refused; the
Linear routes come from the legacy table.

This kernel therefore builds `-DCMAKE_CUDA_ARCHITECTURES=75` and nothing else, and
records every toolchain version it used so the build is reproducible elsewhere.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

WORK = "/tmp/work"           # 60 GB on Kaggle; intermediates never go to /kaggle/working
OUT = "/kaggle/working"      # only result.json + small artifacts survive the session
NINFER_SHA = "6cc1f9fa438b7094685ae417dfa40e1ca324e617"   # kvmem backends/versions.json base_commit
ARCH = "75"                  # Tesla T4. Not a list: one image, one card model.

result = {
    "stages": {},
    "ok": False,
    "arch": ARCH,
    "t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def log(*a):
    print(*a, flush=True)


def stage(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            log(f"=== STAGE {name} start")
            try:
                out = fn(*a, **k)
                result["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "detail": out}
                log(f"=== STAGE {name} OK {result['stages'][name]}")
                return out
            except Exception as e:
                result["stages"][name] = {
                    "ok": False,
                    "seconds": round(time.time() - t0, 1),
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(),
                }
                log(f"=== STAGE {name} FAIL: {e}\n{traceback.format_exc()}")
                json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)
                raise
        return run
    return deco


def sh(cmd, cwd=None, env=None, timeout=None, check=True):
    """Run a command, keep the tail of its output in the result for the record."""
    log("+", " ".join(cmd) if isinstance(cmd, list) else cmd)
    r = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout,
                       capture_output=True, text=True)
    tail = (r.stdout[-3000:] + r.stderr[-3000:]) if (r.stdout or r.stderr) else ""
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed rc={r.returncode}: {cmd}\n{tail}")
    return r.returncode, tail


@stage("probe_toolchain")
def probe_toolchain():
    """cmake >= 3.28 and a generator are hard requirements; pip can supply both."""
    info = {}
    for name, cmd in (
        ("nvidia_smi", ["nvidia-smi", "-L"]),
        ("nvcc", ["nvcc", "--version"]),
        ("gcc", ["gcc", "--version"]),
        ("cmake", ["cmake", "--version"]),
        ("ninja", ["ninja", "--version"]),
    ):
        try:
            _, tail = sh(cmd, check=False, timeout=120)
            info[name] = tail.strip().splitlines()[0] if tail.strip() else ""
        except FileNotFoundError:
            info[name] = "MISSING"
    info["disk_tmp"] = sh(["df", "-h", "/tmp"], check=False)[1].strip()
    info["mem"] = sh(["free", "-g"], check=False)[1].strip()

    # cmake 3.28 is the floor in cmake_minimum_required. Kaggle images drift, so upgrade
    # through pip rather than assuming the base image is new enough.
    rc, tail = sh(["cmake", "--version"], check=False)
    ok_cmake = rc == 0 and "version 3." in tail
    ver = ""
    if ok_cmake:
        for line in tail.splitlines():
            if line.startswith("cmake version"):
                ver = line.split()[-1]
                break
    need = True
    if ver:
        try:
            major, minor = (int(x) for x in ver.split(".")[:2])
            need = (major, minor) < (3, 28)
        except ValueError:
            need = True
    if need or not ok_cmake:
        log(f"cmake {ver or 'missing'} is below 3.28, installing from pip")
        sh([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"])
        ver = sh(["cmake", "--version"])[1].splitlines()[0]

    if sh(["ninja", "--version"], check=False)[0] != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "ninja"])

    info["cmake_resolved"] = ver
    info["ninja_resolved"] = sh(["ninja", "--version"], check=False)[1].strip()
    log(json.dumps(info, indent=2))
    return info


def looks_like_kvmem(d):
    """A kvmem tree root, not one of its subdirectories.

    prepare-backends.py lives in scripts/, so the directory holding it is NOT the tree root --
    an earlier version of this kernel took dirname(prepare-backends.py) as the root and ended up
    with /tmp/work/scripts as the "tree". VERSION at the root plus at least one of the tree's
    known members is the reliable signature.
    """
    if not os.path.isdir(d):
        return False
    if not os.path.exists(os.path.join(d, "VERSION")):
        return False
    for member in ("backends", "CMakeLists.txt", "scripts/prepare-backends.py"):
        if os.path.exists(os.path.join(d, member)):
            return True
    return False


def find_tree():
    """Locate the kvmem root by the files it must contain, not by a fixed depth.

    Kaggle's dataset mount is deeper than it looks --
    /kaggle/input/datasets/<owner>/<slug>/<archive-stem>/ -- so counting glob levels gets it
    wrong. Search for VERSION at any depth and validate its directory.
    """
    for vf in glob.glob(f"{WORK}/**/VERSION", recursive=True) + \
            glob.glob("/kaggle/input/**/VERSION", recursive=True):
        d = os.path.dirname(vf)
        if looks_like_kvmem(d):
            return d
    return None


@stage("extract_kvmem")
def extract_kvmem():
    """Kaggle may mount the dataset as a tarball or already unpacked; handle both.

    The unpacked case is what actually happens here: the dataset upload is auto-extracted to
    /kaggle/input/datasets/<owner>/<hash>/kvmem-master/. Copy it into /tmp/work so the build can
    write inside it (git init for prepare-backends, build directory, media sources).
    """
    os.makedirs(WORK, exist_ok=True)
    archives = []
    for base in glob.glob("/kaggle/input/**", recursive=True):
        if os.path.isdir(base):
            for f in glob.glob(os.path.join(base, "*.tar.gz")):
                if os.path.getsize(f) >= 1 << 20:
                    archives.append(f)
    for f in sorted(set(archives)):
        log("extracting", f, os.path.getsize(f))
        with tarfile.open(f) as tf:
            tf.extractall(WORK)

    src = find_tree()
    if src is None:
        listing = sorted(glob.glob("/kaggle/input/**", recursive=True))[:60]
        raise RuntimeError(f"no kvmem tree found. /kaggle/input sample: {listing}")

    dst = f"{WORK}/kvmem"
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    version = open(os.path.join(dst, "VERSION")).read().strip()
    n_files = sum(len(fs) for _, _, fs in os.walk(dst))
    log(f"kvmem tree: {dst} (from {src}) VERSION={version} files={n_files}")
    return {"tree": dst, "source": src, "version": version, "files": n_files}


@stage("prepare_ninfer")
def prepare_ninfer(tree):
    """Pin ninfer to base_commit, then let kvmem's own script verify and apply its patch chain."""
    n = f"{tree}/backends/ninfer"
    os.makedirs(n, exist_ok=True)

    def git(*a):
        sh(["git", *a], check=True)

    if not os.path.exists(os.path.join(n, ".git")):
        git("init", n)
        git("-C", n, "remote", "add", "origin", "https://github.com/iamwavecut/ninfer-all.git")
    rc, tail = sh(["git", "-C", n, "fetch", "--depth", "1", "origin", NINFER_SHA], check=False)
    if rc != 0:
        log("fetch by sha failed, full fetch:", tail[-400:])
        git("-C", n, "fetch", "origin")
    git("-C", n, "checkout", "--detach", NINFER_SHA)
    head = sh(["git", "-C", n, "rev-parse", "HEAD"])[1].strip()
    assert head == NINFER_SHA, f"HEAD {head} != base_commit {NINFER_SHA}"

    # A tarball has no .git, and prepare-backends.py then takes the release branch and demands
    # SOURCE-MANIFEST.json (a release artifact this commit does not carry). A local commit makes
    # it take the patch branch instead, where it verifies each patch.
    if not os.path.exists(f"{tree}/.git"):
        ident = dict(os.environ, GIT_AUTHOR_NAME="kernel", GIT_AUTHOR_EMAIL="kernel@local",
                     GIT_COMMITTER_NAME="kernel", GIT_COMMITTER_EMAIL="kernel@local")
        sh(["git", "-C", tree, "init"], env=ident)
        sh(["git", "-C", tree, "add", "-A"], env=ident)
        sh(["git", "-C", tree, "commit", "-m", "kvmem snapshot"], env=ident)
    rc, tail = sh([sys.executable, "scripts/prepare-backends.py", "--backend", "ninfer"], cwd=tree)
    return {"head": head, "prepare_tail": tail[-1500:]}


@stage("ensure_media_tools")
def ensure_media_tools():
    """cmake/NinferFFmpeg.cmake REQUIRES meson, ninja, make, and (on x86_64) nasm.

    Kaggle's image ships none of meson and nasm; configure dies at
    `Could not find NINFER_MESON using the following names: meson` before it ever reaches CUDA.
    meson installs cleanly through pip; nasm needs apt.
    """
    tools = {}
    for name in ("meson", "ninja", "make", "nasm", "pkg-config"):
        rc, tail = sh(["which", name], check=False)
        tools[name] = tail.strip() if rc == 0 else None

    if not tools["meson"]:
        sh([sys.executable, "-m", "pip", "install", "-q", "meson"])
    if not tools["ninja"]:
        sh([sys.executable, "-m", "pip", "install", "-q", "ninja"])
    missing = [t for t in ("nasm",) if not tools[t]]
    if missing:
        # Kaggle kernels run as root; if apt is unavailable the build cannot proceed on x86_64
        # because the decoders' SIMD routines are NASM sources.
        rc, tail = sh(["apt-get", "update", "-qq"], check=False, timeout=600)
        rc, tail = sh(["apt-get", "install", "-y", "-qq"] + missing, check=False, timeout=900)
        if rc != 0:
            raise RuntimeError(f"cannot install {missing}; apt said: {tail[-800:]}")

    resolved = {}
    for name in ("meson", "ninja", "make", "nasm", "pkg-config"):
        rc, tail = sh(["which", name], check=False)
        resolved[name] = tail.strip() or "MISSING"
        if resolved[name] == "MISSING":
            raise RuntimeError(f"{name} still missing after install")
    log(json.dumps(resolved, indent=2))
    return resolved


@stage("fetch_media")
def fetch_media(tree):
    """FFmpeg 9.0.2 + dav1d 1.5.4, pinned by sha256, compiled in statically. Build never downloads."""
    n = f"{tree}/backends/ninfer"
    t0 = time.time()
    sh(["bash", "scripts/fetch-media-sources.sh"], cwd=n, timeout=3600)
    d = f"{n}/third_party/media-sources"
    files = []
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        p = os.path.join(d, f)
        files.append({"name": f, "bytes": os.path.getsize(p) if os.path.isfile(p) else None})
    if not files:
        raise RuntimeError(f"fetch-media-sources.sh produced nothing in {d}")
    log(json.dumps(files, indent=2))
    return {"seconds": round(time.time() - t0, 1), "dir": d, "files": files}


CUDA_LIB_DIRS = [
    "/usr/local/cuda/lib64", "/usr/local/cuda/lib",
    "/usr/local/cuda/lib64/stubs", "/usr/local/cuda/lib/stubs",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
    "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib",
]


@stage("cuda_driver")
def ensure_cuda_driver():
    """Make CUDA::cuda_driver resolvable before configure runs.

    FindCUDAToolkit creates that imported target only when it can find a *linkable* libcuda.so.
    CUDA 13's toolkit layout leaves the driver library out of the directories CMake 3.31
    searches, so configure reports "Configuring done / Generating done" and then dies in the
    generate step at src/core/CMakeLists.txt:51 (target_link_libraries) with

        Target "ninfer_core" links to CUDA::cuda_driver but the target was not found.

    CUDA::cudart resolved fine, which is why only the driver target is missing. The runtime
    library itself is present in the image; what is absent is the unversioned name the linker
    needs. Two independent repairs, because either alone can be defeated by the search order:
      1. put a linkable libcuda.so in a directory we hand CMake through CMAKE_LIBRARY_PATH;
      2. pre-define the imported target from a CMAKE_PROJECT_INCLUDE shim, which runs at the
         end of project() -- before Dependencies.cmake calls find_package(CUDAToolkit), and
         FindCUDAToolkit guards every CUDA:: target with if(NOT TARGET ...), so ours survives.
    """
    probe = {"dirs": {}, "versioned": [], "unversioned": []}
    for d in CUDA_LIB_DIRS:
        if not os.path.isdir(d):
            probe["dirs"][d] = None
            continue
        try:
            names = sorted(n for n in os.listdir(d) if n.startswith("libcuda.so"))
        except OSError as exc:
            probe["dirs"][d] = f"unreadable: {exc}"
            continue
        probe["dirs"][d] = names
        for n in names:
            p = os.path.join(d, n)
            (probe["unversioned"] if n == "libcuda.so" else probe["versioned"]).append(p)
    log(json.dumps(probe, indent=2))

    if probe["unversioned"]:
        libcuda = probe["unversioned"][0]
        link_dir = os.path.dirname(libcuda)
    elif probe["versioned"]:
        link_dir = f"{WORK}/cuda-link"
        os.makedirs(link_dir, exist_ok=True)
        libcuda = f"{link_dir}/libcuda.so"
        if os.path.lexists(libcuda):
            os.remove(libcuda)
        os.symlink(probe["versioned"][0], libcuda)
    else:
        raise RuntimeError(
            "no libcuda.so anywhere in the image; the driver's runtime library is what "
            f"src/core/CMakeLists.txt:51 links against. Probed: {json.dumps(probe['dirs'])}"
        )

    shim = f"{WORK}/cuda-driver-shim.cmake"
    with open(shim, "w") as fh:
        fh.write(
            "# Injected by the build kernel. FindCUDAToolkit cannot see CUDA 13's driver library,\n"
            "# so the imported target src/core/CMakeLists.txt:51 needs never gets created.\n"
            "if(NOT TARGET CUDA::cuda_driver)\n"
            "  add_library(CUDA::cuda_driver UNKNOWN IMPORTED)\n"
            "  set_target_properties(CUDA::cuda_driver PROPERTIES\n"
            f'    IMPORTED_LOCATION "{libcuda}")\n'
            "endif()\n"
        )
    log(f"libcuda={libcuda} link_dir={link_dir} shim={shim}")
    return {"libcuda": libcuda, "link_dir": link_dir, "shim": shim, "probe": probe}


@stage("configure")
def configure(tree, driver):
    n = f"{tree}/backends/ninfer"
    build = f"{n}/build-sm{ARCH}"
    cmake_args = [
        "cmake", "-S", n, "-B", build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={ARCH}",
        "-DNINFER_BUILD_APPS=ON",
        f"-DCMAKE_LIBRARY_PATH={driver['link_dir']}",
        f"-DCMAKE_PROJECT_INCLUDE={driver['shim']}",
    ]
    rc, tail = sh(cmake_args, timeout=1800)
    cache = {}
    for line in open(f"{build}/CMakeCache.txt").read().splitlines():
        if ":" in line and not line.startswith("//"):
            k, _, v = line.partition("=")
            if any(s in k for s in ("CUDA_ARCHITECTURES", "CUDA_COMPILER_VERSION", "BUILD_TYPE",
                                    "CXX_COMPILER:", "CUDA_COMPILER:", "BUILD_APPS",
                                    "MEDIA_SOURCES_DIR")):
                cache[k.split(":")[0]] = v
    log(json.dumps(cache, indent=2))
    return {"build_dir": build, "cache": cache, "configure_tail": tail[-1200:]}


@stage("build")
def build(cfg):
    """4 CPUs: this is the long step. Ninja keeps going and we keep the tail for the record."""
    n = os.path.dirname(cfg["build_dir"])
    t0 = time.time()
    rc, tail = sh(["cmake", "--build", cfg["build_dir"], "--parallel", str(os.cpu_count() or 4)],
                  cwd=n, timeout=14400)
    elapsed = round(time.time() - t0, 1)
    exes = []
    for p in glob.glob(f"{cfg['build_dir']}/**/ninfer*", recursive=True):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            exes.append({"path": p, "bytes": os.path.getsize(p)})
    log(f"build finished in {elapsed}s, executables: {[e['path'] for e in exes]}")
    return {"seconds": elapsed, "build_tail": tail[-2500:], "executables": exes}


@stage("smoke")
def smoke(tree, cfg):
    """Does the binary load on a T4 and enumerate the card through NInfer's own tier logic?"""
    build_dir = cfg["build_dir"]
    found = {}
    for name in ("ninfer", "ninfer-serve", "ninfer-calibrate", "ninfer-perplexity"):
        hits = glob.glob(f"{build_dir}/**/{name}", recursive=True)
        if hits:
            found[name] = hits[0]
    log("executables found:", list(found))
    out = {"executables": found}
    for name, path in found.items():
        rc, tail = sh([path, "--help"], check=False, timeout=600)
        out[f"{name}_help_rc"] = rc
        out[f"{name}_help_head"] = "\n".join(tail.splitlines()[:40])
    # NInfer's own view of the card: the tier probe decides whether sm_75 code is what CUDA loads.
    for name in ("ninfer-serve", "ninfer"):
        if name in found:
            rc, tail = sh([found[name], "--list-devices"], check=False, timeout=600)
            out[f"{name}_list_devices_rc"] = rc
            out[f"{name}_list_devices"] = tail[-2000:]
    return out


def main():
    # Toolchain first: if cmake is too old or ninja is missing, that is cheaper to learn now than
    # after a ten-minute source prep.
    toolchain = probe_toolchain()
    media_tools = ensure_media_tools()
    driver = ensure_cuda_driver()
    tree = extract_kvmem()["tree"]
    prepare_ninfer(tree)
    multimedia = fetch_media(tree)
    cfg = configure(tree, driver)
    built = build(cfg)
    smoke_out = smoke(tree, cfg)
    result["ok"] = True
    result["toolchain"] = toolchain
    result["media_tools"] = media_tools
    result["media"] = multimedia
    result["build_seconds"] = built["seconds"]
    result["t1"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
