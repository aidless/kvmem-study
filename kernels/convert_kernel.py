import os, sys, json, hashlib, subprocess, tarfile, time, traceback, shutil, glob

WORK = "/tmp/work"          # 中间产物全部放 /tmp，避免输出目录塞满拖慢下载
OUT = "/kaggle/working"     # 只放 result.json 与最终 .ninfer
NINFER_SHA = "6cc1f9fa438b7094685ae417dfa40e1ca324e617"   # kvmem backends/versions.json ninfer base_commit
GGUF_SHA   = "3874209241c9a397e2f62cd3f70f80fd2dfbf0dfccb6838416bdb48a714e8630"  # docs/qwen35-dense-family.md 权威值
GGUF_REV   = "86835bf9949e4d14d6860f7910b1340ad4f271a9"   # docs/qwen35-dense-family.md pin

def log(*a):
    print(*a, flush=True)

def sha256(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

result = {"stages": {}, "ok": False, "t0": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

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
                result["stages"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                          "trace": traceback.format_exc()}
                log(f"=== STAGE {name} FAIL: {e}\n{traceback.format_exc()}")
                json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)
                raise
        return run
    return deco

@stage("extract_kvmem")
def extract_kvmem():
    # 新版 Kaggle 挂载路径不定（/kaggle/input/<slug> 或 /kaggle/input/datasets/<owner>/<hash>），全树搜索
    roots, tars = [], []
    for dirpath, dirs, files in os.walk("/kaggle/input"):
        if "VERSION" in files or "prepare-backends.py" in files:
            roots.append(dirpath); dirs[:] = []
        for f in files:
            if f.endswith((".tar.gz", ".tgz")):
                tars.append(os.path.join(dirpath, f))
        if len(roots) + len(tars) > 20:
            dirs[:] = []
    log("SEARCH:", "roots=", roots, "tars=", tars)
    for tar in tars:
        with tarfile.open(tar) as t:
            t.extractall(WORK)
    if not roots:
        roots = [p for p in glob.glob(f"{WORK}/kvmem*") if os.path.isdir(p) and os.path.exists(os.path.join(p, "VERSION"))]
    assert roots, "kvmem tree not found anywhere under /kaggle/input"
    root = roots[0]
    if root != f"{WORK}/kvmem":
        if os.path.exists(f"{WORK}/kvmem"):
            shutil.rmtree(f"{WORK}/kvmem")
        shutil.copytree(root, f"{WORK}/kvmem")
    ver = open(f"{WORK}/kvmem/VERSION").read().strip()
    return {"version": ver, "root": f"{WORK}/kvmem"}

@stage("prepare_ninfer")
def prepare_ninfer():
    n = f"{WORK}/kvmem/backends/ninfer"
    os.makedirs(n, exist_ok=True)
    def git(*a, **k):
        subprocess.run(["git", *a], check=True, capture_output=True, text=True, **k)
    if not os.path.exists(os.path.join(n, ".git")):
        git("init", n)
        git("-C", n, "remote", "add", "origin", "https://github.com/iamwavecut/ninfer-all.git")
    r = subprocess.run(["git", "-C", n, "fetch", "--depth", "1", "origin", NINFER_SHA],
                       capture_output=True, text=True)
    if r.returncode != 0:
        log("fetch by sha failed, fallback full clone:", r.stderr[-500:])
        git("-C", n, "fetch", "origin")
    git("-C", n, "checkout", "--detach", NINFER_SHA)
    head = subprocess.run(["git", "-C", n, "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert head == NINFER_SHA, f"HEAD {head} != base_commit {NINFER_SHA}"
    # tarball 无 .git：prepare-backends 会走导出包分支并要求 SOURCE-MANIFEST.json（release 工件，master 没有）。
    # 给 kvmem 根建 git 仓库后走正常路径：手工放置的 backends/ninfer 已钉在 base_commit，补丁逐个 verify。
    rootgit = f"{WORK}/kvmem/.git"
    if not os.path.exists(rootgit):
        ident = dict(os.environ, GIT_AUTHOR_NAME="kernel", GIT_AUTHOR_EMAIL="kernel@local",
                     GIT_COMMITTER_NAME="kernel", GIT_COMMITTER_EMAIL="kernel@local")
        subprocess.run(["git", "-C", f"{WORK}/kvmem", "init"], check=True, capture_output=True)
        subprocess.run(["git", "-C", f"{WORK}/kvmem", "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "-C", f"{WORK}/kvmem", "commit", "-m", "kvmem master snapshot"],
                       check=True, capture_output=True, env=ident)
    subprocess.run([sys.executable, "scripts/prepare-backends.py", "--backend", "ninfer"],
                   cwd=f"{WORK}/kvmem", check=True)
    return {"head": head}

@stage("download_model")
def download_model():
    gguf_local = None
    # 主通道：HF pinned revision（Kaggle 出口对 HF CDN 快）
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        gguf_local = hf_hub_download(repo_id="unsloth/Qwen3.5-4B-MTP-GGUF",
                                     filename="Qwen3.5-4B-Q4_K_M.gguf",
                                     revision=GGUF_REV,
                                     local_dir=f"{WORK}/model_src")
    except Exception as e:
        log("HF download failed, fallback ModelScope:", e)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "modelscope"], check=True)
        from modelscope import snapshot_download as ms_snapshot
        ms_snapshot("unsloth/Qwen3.5-4B-MTP-GGUF",
                    local_dir=f"{WORK}/model_src/ms_gguf",
                    allow_patterns=["Qwen3.5-4B-Q4_K_M.gguf"])
        gguf_local = f"{WORK}/model_src/ms_gguf/Qwen3.5-4B-Q4_K_M.gguf"
    got = sha256(gguf_local)
    assert got == GGUF_SHA, f"GGUF sha256 mismatch: got {got}, want {GGUF_SHA}"
    # config/tokenizer/generation_config（只要文本小文件）
    from huggingface_hub import snapshot_download
    cfg = snapshot_download("Qwen/Qwen3.5-4B",
                            allow_patterns=["*.json", "*.txt", "*.jinja", "*token*"],
                            local_dir=f"{WORK}/model_src/Qwen3.5-4B-config")
    # 官方 4B 仓库没有 generation_config.json（docs/qwen35-dense-family.md:25 明确说明）。
    # 按 weight-conversion.md 指引：从官方 config.json 的 eos_token_id 生成最小资源，
    # 记录来源，不借用其他模型的 generation 设置。
    gc_path = os.path.join(cfg, "generation_config.json")
    gc_provenance = "official repo ships none"
    if not os.path.exists(gc_path):
        cfgjson = json.load(open(os.path.join(cfg, "config.json")))
        tc = cfgjson.get("text_config", cfgjson)
        eos = tc.get("eos_token_id")
        assert eos is not None, "official config.json lacks eos_token_id; need explicit resource"
        json.dump({"eos_token_id": eos}, open(gc_path, "w"))
        gc_provenance = f"minimal {{\"eos_token_id\": {eos}}} derived from official Qwen/Qwen3.5-4B config.json"
    return {"gguf": gguf_local, "gguf_sha256": got, "config_dir": cfg, "generation_config_provenance": gc_provenance}

@stage("convert")
def convert():
    ninfer = f"{WORK}/kvmem/backends/ninfer"
    out = f"{OUT}/qwen3_5_4b_mtp.ninfer"
    cfg = result["stages"]["download_model"]["detail"]["config_dir"]
    gguf = result["stages"]["download_model"]["detail"]["gguf"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf"], check=True)
    subprocess.run([sys.executable, "-m", "tools.convert",
                    "--model", cfg,
                    "--recipe", "qwen3_5_dense_gguf",
                    "--components", "text,mtp",
                    "--source", f"gguf={gguf}",
                    "--device", "cpu",
                    "--out", out],
                   cwd=ninfer, check=True)
    size = os.path.getsize(out)
    sha = sha256(out)
    # 参考值：docs/qwen35-dense-family.md 完整 32+1 层 .ninfer = 2,827,176,704 bytes
    return {"out": out, "size": size, "sha256": sha,
            "ref_size": 2827176704, "size_matches_ref": size == 2827176704}

stages = {}
try:
    stages["extract_kvmem"] = extract_kvmem()
    stages["prepare_ninfer"] = prepare_ninfer()
    stages["download_model"] = download_model()
    stages["convert"] = convert()
    result["ok"] = True
finally:
    for k, v in stages.items():
        result.setdefault("stages", {}).setdefault(k, {}).update(v if isinstance(v, dict) else {"detail": v})
    result["t1"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(result, open(f"{OUT}/result.json", "w"), indent=2)

if result["ok"]:
    d = result["stages"]["convert"]["detail"]
    log(f"=== ALL DONE: {d['out']} size={d['size']} sha256={d['sha256']} matches_ref={d['size_matches_ref']}")
else:
    log("=== FAILED, see result.json")
    sys.exit(1)
