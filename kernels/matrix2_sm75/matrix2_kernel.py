"""Phase B scan kernel: build NInfer for Turing sm_75, then run the capacity and Host-tier scans.

Build and scan live in one kernel on purpose. A Kaggle session gets ~12 h and its filesystem does
not survive the session, so carrying the binary to a second kernel would mean either rebuilding
from scratch or round-tripping gigabytes through a dataset. Building once and scanning in the same
run is cheaper and keeps the binary and its measurements in one provenance record.

What is measured
----------------
1) Capacity: `--kv-capacity auto` lets the engine resolve its own page-group count from free
   memory, and we compare that against `tools/capacity.py`, which computes the same number from
   the artifact manifest alone. Report 16 documented two calibres that differ by 6.4x on a 6 GB
   card; this run replaces both predictions with a measurement.
2) Host tier: `--host-cache-mib {0, 2048, 8192}` x context {8K, 32K}. Two identical completions
   per row, then GET /stats, whose `context_cache.hybrid` block carries all 26 numbers
   (prefix/hybrid_cache.h:56-74 plus three index counters) cumulatively -- see read_stats() for
   why /stats and not the request log.
3) Two cards: `--devices 0,1 --stage-layers 16,16`. This is a DIFFERENT cache mode: naming
   `--devices` selects the checkpoint catalog, so the hybrid block is absent by construction and
   this row measures stage_plan.h crosses_after only.

Hard flag constraints from serve_options.cpp that shape the matrix
-----------------------------------------------------------------
- The hybrid prefix cache is the DEFAULT (:1704-1712). Naming any catalog option -- `--devices`,
  `--no-prefix-reuse`, `--graft`, or a legacy cache flag -- selects the checkpoint catalog instead.
- `--host-cache-mib` is the hybrid Host-tier flag; `--host-kv-mib` is a catalog flag, and :1723-1727
  rejects it inside hybrid mode. The two are not synonyms and mixing them fails at launch.
- Hybrid runs on one device at this pinned base: `--use-alt-prefix-caching` with `--devices` throws
  "runs on one device and cannot be combined with pipeline --devices" (:1733-1736). That is a
  property of `6cc1f9fa`, not of the engine: the fork fixed the multi-device hybrid startup on
  2026-11-10 (`af60c489cd`) and then added `--use-alt-prefix-caching` over several devices
  (`38a10aa49a`). Record the row as "base behaviour", never as an engine limit.
- Explicit `--kv-capacity N` and `--kv-headroom-mib` are mutually exclusive:
  model_instance.cpp:58-72 throws "explicit kv_capacity must not carry automatic headroom", and
  :1758-1760 makes `--kv-headroom-mib` require `--kv-capacity auto`.
- `--kv-capacity` counts TOKENS, not page groups (parse_kv_capacity, serve_options.cpp:80-85).
- `NINFER_LINEAR_ROUTES=unified` is refused on Turing (docs/build-linux.md:126-128).
"""

import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.error
import urllib.request

WORK = "/tmp/work"
OUT = "/kaggle/working"
NINFER_SHA = "6cc1f9fa438b7094685ae417dfa40e1ca324e617"
ARCH = "75"
MODEL_NAME = "qwen3_5_4b_mtp.ninfer"
# Both pinned by docs/qwen35-dense-family.md and verified in Phase A (kvmem-study/kaggle/
# convert_kernel.py): the GGUF digest is the authoritative value the docs quote, so a conversion
# here that produced different bytes would be a different model.
GGUF_REV = "86835bf9949e4d14d6860f7910b1340ad4f271a9"
GGUF_SHA = "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"
# Where Phase A left the artifact. Kaggle gives no persistent disk between sessions, so either a
# dataset carries it or this kernel regenerates it; both paths are tried in acquire_model().
MODEL_CANDIDATES = (
    "/kaggle/input/**/" + MODEL_NAME,
    f"{WORK}/**/" + MODEL_NAME,
)

result = {
    "stages": {},
    "ok": False,
    "arch": ARCH,
    "t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def log(*a):
    print(*a, flush=True)


def sha256(path, bufsize=1 << 22):
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(bufsize), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save():
    json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)


def stage(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            log(f"=== STAGE {name} start")
            try:
                out = fn(*a, **k)
                result["stages"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "detail": out}
                log(f"=== STAGE {name} OK {result['stages'][name]['seconds']}s")
                save()
                return out
            except Exception as e:
                result["stages"][name] = {
                    "ok": False,
                    "seconds": round(time.time() - t0, 1),
                    "error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc(),
                }
                log(f"=== STAGE {name} FAIL: {e}\n{traceback.format_exc()}")
                save()
                raise
        return run
    return deco


def sh(cmd, cwd=None, env=None, timeout=None, check=True):
    log("+", " ".join(cmd) if isinstance(cmd, list) else cmd)
    try:
        r = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired as e:
        tail = ((e.stdout or b"").decode("utf-8", "replace")[-2000:] if isinstance(e.stdout, bytes) else (e.stdout or ""))[-2000:]
        raise RuntimeError(f"timeout after {timeout}s: {cmd}\n{tail}")
    tail = (r.stdout or "")[-4000:] + (r.stderr or "")[-4000:]
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed rc={r.returncode}: {cmd}\n{tail}")
    return r.returncode, tail


def looks_like_kvmem(d):
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "VERSION")):
        return False
    return any(os.path.exists(os.path.join(d, m))
               for m in ("backends", "CMakeLists.txt", "scripts/prepare-backends.py"))


@stage("probe_toolchain")
def probe_toolchain():
    info = {}
    for name, cmd in (("nvidia_smi", ["nvidia-smi", "-L"]), ("nvcc", ["nvcc", "--version"]),
                      ("gcc", ["gcc", "--version"]), ("cmake", ["cmake", "--version"]),
                      ("ninja", ["ninja", "--version"])):
        try:
            _, tail = sh(cmd, check=False, timeout=120)
            info[name] = tail.strip().splitlines()[0] if tail.strip() else ""
        except FileNotFoundError:
            info[name] = "MISSING"
    info["mem"] = sh(["free", "-g"], check=False)[1].strip()
    info["disk_tmp"] = sh(["df", "-h", "/tmp"], check=False)[1].strip()

    ver = info.get("cmake", "")
    need = True
    if "cmake version" in ver:
        try:
            major, minor = (int(x) for x in ver.split()[-1].split(".")[:2])
            need = (major, minor) < (3, 28)
        except ValueError:
            need = True
    if need:
        log(f"cmake too old ({ver}), installing from pip")
        sh([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"])
        info["cmake"] = sh(["cmake", "--version"])[1].splitlines()[0]
    if sh(["ninja", "--version"], check=False)[0] != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "ninja"])
    log(json.dumps(info, indent=2))
    return info


@stage("extract_kvmem")
def extract_kvmem():
    os.makedirs(WORK, exist_ok=True)
    for base in glob.glob("/kaggle/input/**", recursive=True):
        if os.path.isdir(base):
            for f in glob.glob(os.path.join(base, "*.tar.gz")):
                if os.path.getsize(f) >= 1 << 20:
                    log("extracting", f)
                    with tarfile.open(f) as tf:
                        tf.extractall(WORK)
    src = None
    for vf in glob.glob(f"{WORK}/**/VERSION", recursive=True) + \
            glob.glob("/kaggle/input/**/VERSION", recursive=True):
        d = os.path.dirname(vf)
        if looks_like_kvmem(d):
            src = d
            break
    if src is None:
        raise RuntimeError(f"no kvmem tree. sample: {sorted(glob.glob('/kaggle/input/**', recursive=True))[:40]}")
    dst = f"{WORK}/kvmem"
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    version = open(os.path.join(dst, "VERSION")).read().strip()
    return {"tree": dst, "source": src, "version": version}


@stage("prepare_ninfer")
def prepare_ninfer(tree):
    n = f"{tree}/backends/ninfer"
    os.makedirs(n, exist_ok=True)

    def git(*a):
        sh(["git", *a], check=True)

    if not os.path.exists(os.path.join(n, ".git")):
        git("init", n)
        git("-C", n, "remote", "add", "origin", "https://github.com/iamwavecut/ninfer-all.git")
    rc, tail = sh(["git", "-C", n, "fetch", "--depth", "1", "origin", NINFER_SHA], check=False)
    if rc != 0:
        log("fetch by sha failed, full fetch:", tail[-300:])
        git("-C", n, "fetch", "origin")
    git("-C", n, "checkout", "--detach", NINFER_SHA)
    head = sh(["git", "-C", n, "rev-parse", "HEAD"])[1].strip()
    assert head == NINFER_SHA, f"HEAD {head} != {NINFER_SHA}"

    if not os.path.exists(f"{tree}/.git"):
        ident = dict(os.environ, GIT_AUTHOR_NAME="kernel", GIT_AUTHOR_EMAIL="kernel@local",
                     GIT_COMMITTER_NAME="kernel", GIT_COMMITTER_EMAIL="kernel@local")
        sh(["git", "-C", tree, "init"], env=ident)
        sh(["git", "-C", tree, "add", "-A"], env=ident)
        sh(["git", "-C", tree, "commit", "-m", "kvmem snapshot"], env=ident)
    rc, tail = sh([sys.executable, "scripts/prepare-backends.py", "--backend", "ninfer"], cwd=tree)
    return {"head": head}


@stage("acquire_model")
def acquire_model(tree):
    """The .ninfer artifact: reuse one from a dataset if mounted, else convert from GGUF here.

    The conversion command is copied verbatim from the Phase A kernel that produced the 2,827,176,704
    byte artifact (kvmem-study/kaggle/convert_kernel.py, convert()): `-m tools.convert` with
    `--model <config dir> --recipe qwen3_5_dense_gguf --components text,mtp --source gguf=<file>
    --device cpu --out <file>`, run from the ninfer tree. An earlier version of this stage invented
    `--source-gguf/--revision/--output`, which does not exist: v3 failed with
    "No module named tools.convert" because it ran from the kvmem root, not backends/ninfer.
    """
    for pat in MODEL_CANDIDATES:
        for p in glob.glob(pat, recursive=True):
            if os.path.getsize(p) > 1 << 30:
                log("found artifact", p, os.path.getsize(p))
                return {"path": p, "bytes": os.path.getsize(p), "source": "dataset"}
    log("no artifact mounted; converting from GGUF in this kernel")
    ninfer = f"{tree}/backends/ninfer"
    sh([sys.executable, "-m", "pip", "install", "-q", "gguf"], timeout=600)
    from huggingface_hub import hf_hub_download, snapshot_download
    gguf_local = hf_hub_download(repo_id="unsloth/Qwen3.5-4B-MTP-GGUF",
                                 filename="Qwen3.5-4B-Q4_K_M.gguf",
                                 revision=GGUF_REV,
                                 local_dir=f"{WORK}/model_src")
    got = sha256(gguf_local)
    if got != GGUF_SHA:
        raise RuntimeError(f"GGUF sha256 mismatch: got {got}, want {GGUF_SHA}")
    cfg = snapshot_download("Qwen/Qwen3.5-4B",
                            allow_patterns=["*.json", "*.txt", "*.jinja", "*token*"],
                            local_dir=f"{WORK}/model_src/Qwen3.5-4B-config")
    # The official 4B repository ships no generation_config.json (docs/qwen35-dense-family.md:25).
    # The conversion resources are packed byte-for-byte, so supply the minimal one from eos_token_id
    # and record where it came from instead of borrowing another model's generation settings.
    gc_path = os.path.join(cfg, "generation_config.json")
    gc_provenance = "official repo ships none"
    if not os.path.exists(gc_path):
        with open(os.path.join(cfg, "config.json"), encoding="utf-8") as handle:
            cfgjson = json.load(handle)
        text_cfg = cfgjson.get("text_config", cfgjson)
        eos = text_cfg.get("eos_token_id")
        if eos is None:
            raise RuntimeError("official config.json lacks eos_token_id")
        with open(gc_path, "w", encoding="utf-8") as handle:
            json.dump({"eos_token_id": eos}, handle)
        gc_provenance = f'minimal {{"eos_token_id": {eos}}} from official Qwen/Qwen3.5-4B config.json'
    dst = f"{WORK}/{MODEL_NAME}"
    sh([sys.executable, "-m", "tools.convert",
        "--model", cfg,
        "--recipe", "qwen3_5_dense_gguf",
        "--components", "text,mtp",
        "--source", f"gguf={gguf_local}",
        "--device", "cpu",
        "--out", dst], cwd=ninfer, timeout=7200)
    return {"path": dst, "bytes": os.path.getsize(dst), "source": "converted",
            "gguf_sha256": got, "generation_config": gc_provenance}


@stage("ensure_media_tools")
def ensure_media_tools():
    """cmake/NinferFFmpeg.cmake requires meson, ninja, make, nasm (x86_64).

    Kaggle ships neither meson nor nasm; configure fails at
    "Could not find NINFER_MESON using the following names: meson" before reaching CUDA.
    """
    tools = {}
    for name in ("meson", "ninja", "make", "nasm", "pkg-config"):
        rc, tail = sh(["which", name], check=False)
        tools[name] = tail.strip() or None
    if not tools["meson"]:
        sh([sys.executable, "-m", "pip", "install", "-q", "meson"])
    if not tools["ninja"]:
        sh([sys.executable, "-m", "pip", "install", "-q", "ninja"])
    if not tools["nasm"]:
        sh(["apt-get", "update", "-qq"], check=False, timeout=600)
        sh(["apt-get", "install", "-y", "-qq", "nasm"], check=False, timeout=900)
    resolved = {}
    for name in ("meson", "ninja", "make", "nasm"):
        rc, tail = sh(["which", name], check=False)
        resolved[name] = tail.strip() or "MISSING"
        if resolved[name] == "MISSING":
            raise RuntimeError(f"{name} still missing")
    return resolved


@stage("fetch_media")
def fetch_media(tree):
    n = f"{tree}/backends/ninfer"
    t0 = time.time()
    sh(["bash", "scripts/fetch-media-sources.sh"], cwd=n, timeout=3600)
    return {"seconds": round(time.time() - t0, 1)}


# Order matters: the GPU runtime's own directory comes first. On Kaggle's T4 image the driver lives
# in /usr/local/nvidia/lib64 (verified 2026-11-17 by kernel <kaggle-user>/kvmem-driver-probe:
# libcuda.so -> libcuda.so.1 -> libcuda.so.580.178.04, cuInit returns 0, device "Tesla T4"),
# while every cuda-toolkit directory is empty of it and /usr/lib/x86_64-linux-gnu has no match at all.
CUDA_LIB_DIRS = [
    "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib",
    "/usr/local/cuda/lib64", "/usr/local/cuda/lib",
    "/usr/local/cuda/lib64/stubs", "/usr/local/cuda/lib/stubs",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/targets/x86_64-linux/lib/stubs",
    "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/lib",
    "/opt/conda/lib", "/usr/local/lib",
]


def loader_path():
    """Return the absolute path of the libcuda the dynamic loader actually opens, or None.

    Stronger than a directory scan: it asks the loader, then reads /proc/self/maps for the real file.
    Verified on the Kaggle T4 image (kernel <kaggle-user>/kvmem-driver-probe, 2026-11-17):
    ctypes.CDLL("libcuda.so.1") succeeds and maps /usr/local/nvidia/lib64/libcuda.so.580.178.04,
    cuInit(0) returns 0 and cuDeviceGetName yields "Tesla T4".
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        del lib
    except OSError as exc:
        log(f"loader cannot open libcuda.so.1: {exc}")
        return None
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if "libcuda.so" in line:
                    return line.split()[-1]
    except OSError:
        pass
    return None


@stage("cuda_driver")
def ensure_cuda_driver():
    """Make CUDA::cuda_driver resolvable before configure runs.

    FindCUDAToolkit creates that imported target only when it can find a *linkable* libcuda.so.
    CUDA 13's toolkit layout leaves the driver library out of the directories CMake 3.31 searches,
    so configure prints "Configuring done / Generating done" and then dies in the generate step at
    src/core/CMakeLists.txt:51 (target_link_libraries) with

        Target "ninfer_core" links to CUDA::cuda_driver but the target was not found.

    CUDA::cudart resolved fine -- that is why only the driver target is missing. Two independent
    repairs, because either alone can be defeated by the search order:
      1. hand CMake a directory holding a linkable libcuda.so through CMAKE_LIBRARY_PATH;
      2. pre-define the imported target from a CMAKE_PROJECT_INCLUDE shim. That file runs at the end
         of project(), before Dependencies.cmake calls find_package(CUDAToolkit), and
         FindCUDAToolkit guards every CUDA:: target with if(NOT TARGET ...), so ours survives.
    """
    # This stage runs before extract_kvmem(), which is what normally creates WORK. Writing the shim
    # into a missing directory raises FileNotFoundError at the open() and kills the stage -- that is
    # exactly how v2 failed after the driver library was found (2026-11-17).
    os.makedirs(WORK, exist_ok=True)
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
            path = os.path.join(d, n)
            (probe["unversioned"] if n == "libcuda.so" else probe["versioned"]).append(path)
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
        # Nothing in the known directories. Ask the loader instead of guessing: if the GPU runtime
        # can open libcuda.so.1 at all, /proc/self/maps gives its real path, and that path is by
        # definition linkable and loadable at run time.
        libcuda = loader_path()
        if not libcuda:
            raise RuntimeError("no libcuda.so anywhere the known directories nor the loader; "
                               "src/core/CMakeLists.txt:51 links against it. "
                               f"Probed: {json.dumps(probe['dirs'])}")
        link_dir = f"{WORK}/cuda-link"
        os.makedirs(link_dir, exist_ok=True)
        probe["via_loader"] = libcuda

    shim = f"{WORK}/cuda-driver-shim.cmake"
    with open(shim, "w") as fh:
        fh.write("# Injected by the scan kernel: FindCUDAToolkit cannot see CUDA 13's driver "
                 "library, so the\n# imported target src/core/CMakeLists.txt:51 needs never gets "
                 "created.\n"
                 "if(NOT TARGET CUDA::cuda_driver)\n"
                 "  add_library(CUDA::cuda_driver UNKNOWN IMPORTED)\n"
                 "  set_target_properties(CUDA::cuda_driver PROPERTIES\n"
                 f'    IMPORTED_LOCATION "{libcuda}")\n'
                 "endif()\n")
    log(f"libcuda={libcuda} link_dir={link_dir} shim={shim}")
    return {"libcuda": libcuda, "link_dir": link_dir, "shim": shim, "probe": probe}


@stage("configure")
def configure(tree, driver):
    n = f"{tree}/backends/ninfer"
    build = f"{n}/build-sm{ARCH}"
    sh(["cmake", "-S", n, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_CUDA_ARCHITECTURES={ARCH}", "-DNINFER_BUILD_APPS=ON",
        f"-DCMAKE_LIBRARY_PATH={driver['link_dir']}",
        f"-DCMAKE_PROJECT_INCLUDE={driver['shim']}"], timeout=1800)
    cache = {}
    for line in open(f"{build}/CMakeCache.txt").read().splitlines():
        if ":" in line and not line.startswith("//"):
            k, _, v = line.partition("=")
            if any(s in k for s in ("CUDA_ARCHITECTURES", "CUDA_COMPILER_VERSION", "BUILD_TYPE",
                                    "CXX_COMPILER:", "CUDA_COMPILER:")):
                cache[k.split(":")[0]] = v
    return {"build_dir": build, "cache": cache}


@stage("build")
def build(cfg):
    t0 = time.time()
    # 2026-11-17 v4: 14400s (4h) timed out at [662/943] = 70.2% with zero compile errors.
    # Linear extrapolation: 943/662 * 4h = 5.70h for the full build, ~1.70h remaining.
    # Kaggle GPU kernels allow 9h; 8h leaves buffer for slower objects. Other stages took 185s.
    sh(["cmake", "--build", cfg["build_dir"], "--parallel", str(os.cpu_count() or 4)], timeout=28800)
    exes = [p for p in glob.glob(f"{cfg['build_dir']}/**/*", recursive=True)
            if os.path.isfile(p) and os.access(p, os.X_OK) and "ninfer" in os.path.basename(p)
            and "." not in os.path.basename(p)]
    return {"seconds": round(time.time() - t0, 1), "executables": exes}


def serve_help(exe):
    rc, tail = sh([exe, "--help"], check=False, timeout=300)
    return {"rc": rc, "text": tail[:6000]}


def http_get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def http_post_json(url, body, timeout=600):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", "replace")


def wait_ready(port, seconds):
    """Block until GET /health answers. Model load dominates; 2.8 GB off /kaggle/input is slow."""
    t0 = time.time()
    last = None
    while time.time() - t0 < seconds:
        try:
            status, body = http_get(f"http://127.0.0.1:{port}/health", timeout=5)
            if status == 200:
                return {"ready_after_seconds": round(time.time() - t0, 1), "health": body[:400]}
            last = f"status {status}"
        except Exception as exc:                     # noqa: BLE001 - any refusal means "not yet"
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(2.0)
    return {"ready": False, "waited_seconds": round(time.time() - t0, 1), "last": last}


def model_id(port):
    try:
        _, body = http_get(f"http://127.0.0.1:{port}/v1/models", timeout=10)
        data = json.loads(body).get("data") or []
        if data:
            return data[0].get("id") or data[0].get("root") or "default"
    except Exception:                                # noqa: BLE001 - fall back to the alias
        pass
    return "default"


# One long prompt, sent twice: the second admission is what makes snapshot_hits / reused_tokens
# move. Shorter than a prefill chunk would never publish a block, so it stays above ~64 tokens.
REUSE_PROMPT = ("The quick brown fox jumps over the lazy dog. " * 60).strip()


def send_traffic(port, repeats=2, prompt=None):
    """Two identical completions. Hybrid counters only move on the second admission."""
    mid = model_id(port)
    rows = []
    text_prompt = prompt or REUSE_PROMPT
    for i in range(repeats):
        t0 = time.time()
        try:
            status, body = http_post_json(
                f"http://127.0.0.1:{port}/v1/completions",
                {"model": mid, "prompt": text_prompt, "max_tokens": 8, "stream": False,
                 "temperature": 0},
            )
            got = json.loads(body)
            rows.append({"request": i, "status": status, "seconds": round(time.time() - t0, 2),
                         "usage": got.get("usage"),
                         "text": (got.get("choices") or [{}])[0].get("text", "")[:80],
                         "error": None if status == 200 else body[:500]})
        except urllib.error.HTTPError as exc:
            rows.append({"request": i, "status": exc.code, "seconds": round(time.time() - t0, 2),
                         "error": exc.read().decode("utf-8", "replace")[:500]})
        except Exception as exc:                     # noqa: BLE001 - recorded, not raised
            rows.append({"request": i, "status": None, "seconds": round(time.time() - t0, 2),
                         "error": f"{type(exc).__name__}: {exc}"})
    return rows


def read_stats(port):
    """GET /stats is the only channel that carries the hybrid block.

    make_stats_report (src/serve/load_report.cpp:87-118) rebuilds the throughput record with
    previous={} -- so context_cache.hybrid arrives CUMULATIVE, not per-interval, and no
    --request-log-jsonl file has to be parsed or rotated. The block is emitted only once
    snapshots, tree blocks or insertions are non-zero, so an empty result on the first request is
    the expected shape, not a failure.
    """
    try:
        _, body = http_get(f"http://127.0.0.1:{port}/stats", timeout=30)
        j = json.loads(body)
        cc = j.get("context_cache") or {}
        return {"ok": True, "hybrid": cc.get("hybrid"), "occupancy": cc.get("occupancy"),
                "capacity": j.get("capacity"), "context_cache_keys": sorted(cc)}
    except Exception as exc:                         # noqa: BLE001 - reported as a row
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def read_rss_kib(pid):
    """Peak-resident set from /proc/<pid>/status VmRSS (Linux KB). Returns None if the
    process already exited. This is the third 04-plan metric that v5 did not collect."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError):            # noqa: BLE001 - process may vanish mid-sample
        return None
    return None


def run_serve(exe, args, seconds=120, env=None, traffic=0, port=18080, tag="run", artifact=None, prompt=None,
              rss_interval=0.5):
    """Start ninfer-serve, wait for /health, optionally send traffic, read /stats, stop it.

    Capacity resolution is printed at startup, so a bare run reads it. The hybrid counters need a
    live server and real admissions, which is why traffic is a parameter rather than a second pass.

    The artifact is argv[1], not `--models-dir`: serve_options.cpp:1904-1910 makes those two
    mutually exclusive -- an artifact with a router directory throws "router mode takes its models
    from --models-dir / --models-preset, not an artifact path". Artifact mode is the direct one.

    stdout goes to a file, not a pipe: nothing drains a pipe while the client traffic runs, and a
    full 64 KiB pipe buffer would block the server mid-startup. The file also survives the
    terminate, so the startup log is complete rather than whatever happened to be buffered.

    v2 adds a VmRSS sampler thread (the 04-plan metric v5 left as UNKNOWN). It samples every
    rss_interval seconds from Popen to terminate and reports peak/samples/last. It never touches
    the process: reading /proc is side-effect free and safe while the server runs.
    """
    log_path = f"{WORK}/serve-{tag}-{port}.log"
    cmd = [exe] + ([artifact] if artifact else []) + \
          ["--no-webui", "--host", "127.0.0.1", "--port", str(port)] + args
    log("+", " ".join(cmd))
    t0 = time.time()
    ready = {"ready": False, "note": "not polled"}
    served = []
    stats = None
    with open(log_path, "w") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True,
                             env=dict(os.environ, **(env or {})))
        # v2: sample VmRSS from Popen to terminate. A daemon thread so a crash in the
        # main probe path can never hang the kernel on join(); it only reads /proc.
        rss = {"samples": [], "stop": threading.Event()}
        def _sample():
            while not rss["stop"].is_set():
                v = read_rss_kib(p.pid)
                if v is not None:
                    rss["samples"].append(v)
                rss["stop"].wait(rss_interval)
        t = threading.Thread(target=_sample, daemon=True)
        t.start()
        try:
            # Give the process a moment to either die on a flag error or bind the port.
            deadline = min(seconds, 300)
            if traffic:
                ready = wait_ready(port, deadline)
                if ready.get("ready_after_seconds") is not None:
                    served = send_traffic(port, repeats=traffic, prompt=prompt)
                    time.sleep(1.0)
                    stats = read_stats(port)
            else:
                # No client: settle for the port answering or the process exiting early.
                while time.time() - t0 < min(seconds, 120):
                    if p.poll() is not None:
                        break
                    try:
                        if http_get(f"http://127.0.0.1:{port}/health", timeout=3)[0] == 200:
                            ready = {"ready_after_seconds": round(time.time() - t0, 1)}
                            break
                    except Exception:            # noqa: BLE001 - refusal means "not yet"
                        pass
                    time.sleep(2.0)
            alive = p.poll() is None
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
            rss["stop"].set()
            t.join(timeout=2.0)
    samples = rss["samples"]
    rss_out = {"peak_kib": max(samples) if samples else None,
               "last_kib": samples[-1] if samples else None,
               "n": len(samples),
               "peak_mib": round(max(samples) / 1024, 1) if samples else None}
    text = open(log_path, errors="replace").read()
    return {"tag": tag, "alive_after_probe": alive, "returncode": p.returncode,
            "seconds": round(time.time() - t0, 1), "ready": ready, "traffic": served,
            "stats": stats, "rss": rss_out, "log_path": log_path, "log": text[-12000:],
            "lines": text.count("\n")}



def make_long_prompt(tokens_approx):
    """Repeat a fixed sentence to approximate a long context prompt."""
    unit = "The quick brown fox jumps over the lazy dog. "
    n = max(1, int(tokens_approx * 0.8 / len(unit.split())))
    return (unit * n).strip()


@stage("scan_matrix")
def scan_matrix(cfg, model):
    """04-plan B x ctx x H matrix, projected onto the LEGAL engine space.

    startup.cpp:1331-1332 rejects explicit kv_capacity < max_context, so the original 04 cells
    with B < ctx are structurally illegal, not engineering failures. We keep them as rows with
    status=STRUCTURAL_NO and run only the legal ones, so the report can show both the plan and
    what the engine actually admits.
    """
    hits = glob.glob(f"{cfg['build_dir']}/**/ninfer-serve", recursive=True)
    if not hits:
        raise RuntimeError("ninfer-serve not found")
    serve = hits[0]
    common = ["--kv-dtype", "bf16", "--max-concurrency", "1"]

    main = []
    for ctx in (8192, 32768, 131072):
        for budget in (8192, 16384, 32768, 65536, 131072):
            tag = f"m-b{budget}-ctx{ctx}"
            row = {"tag": tag, "budget": budget, "max_context": ctx, "host_mib": 2048}
            if budget < ctx:
                row.update({"status": "STRUCTURAL_NO",
                            "reason": "startup.cpp:1331 kv_capacity must be at least max_context"})
                main.append(row)
                continue
            args = common + ["--kv-capacity", str(budget), "--max-context", str(ctx),
                             "--host-cache-mib", "2048"]
            prompt = make_long_prompt(min(ctx, 65536))
            row.update({"args": args, "prompt_chars": len(prompt),
                        "run": run_serve(serve, args, seconds=1200, prompt=prompt,
                                         traffic=2, port=18080, tag=tag, artifact=model)})
            main.append(row)

    hsweep = []
    for host_mib in (64, 512, 2048, 8192):
        ctx, budget = 32768, 65536
        tag = f"h-{host_mib}"
        args = common + ["--kv-capacity", str(budget), "--max-context", str(ctx),
                         "--host-cache-mib", str(host_mib)]
        prompt = make_long_prompt(ctx)
        hsweep.append({"tag": tag, "budget": budget, "max_context": ctx, "host_mib": host_mib,
                       "args": args, "prompt_chars": len(prompt),
                       "run": run_serve(serve, args, seconds=1200, prompt=prompt,
                                        traffic=2, port=18080, tag=tag, artifact=model)})

    # v3: int8 monotone comparison. At the same headroom, int8 halves KV bytes/token
    # (report 16: 1.03125 vs 2.00 MiB/page-group on 4B), so the engine must admit MORE
    # page groups than bf16. This is a *shape/monotone* probe, not an absolute prediction:
    # v5 measured bf16 5323 @hr1024 / 5579 @hr512; int8 must exceed both.
    # No traffic (capacity is read from the startup log), so it is fast.
    # v2 FAILED: used "int8_g64" (capacity.py's internal name); the CLI's parse_kv_dtype
    # (serve_options.cpp:62-77) accepts only "int8". Fixed to "int8" for v3.
    int8 = []
    for headroom in (1024, 512):
        tag = f"int8-hr{headroom}"
        args = ["--kv-dtype", "int8", "--kv-capacity", "auto",
                "--kv-headroom-mib", str(headroom), "--max-context", "8192",
                "--max-concurrency", "1"]
        int8.append({"tag": tag, "headroom_mib": headroom, "args": args,
                     "run": run_serve(serve, args, seconds=120, traffic=0,
                                      port=18080, tag=tag, artifact=model)})
    return {"main": main, "hsweep": hsweep, "int8": int8}


def main():
    toolchain = probe_toolchain()
    ensure_media_tools()
    driver = ensure_cuda_driver()
    tree = extract_kvmem()["tree"]
    prepare_ninfer(tree)
    model = acquire_model(tree)["path"]
    fetch_media(tree)
    cfg = configure(tree, driver)
    build(cfg)
    matrix = scan_matrix(cfg, model)
    result["ok"] = True
    result["toolchain"] = toolchain
    result["cuda_driver"] = driver
    result["model"] = model
    result["matrix"] = matrix
    result["t1"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()
    log("DONE")


if __name__ == "__main__":
    main()
