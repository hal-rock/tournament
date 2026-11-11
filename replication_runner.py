"""Run the replication's jobs on one multi-GPU node (PLAN_replication.md, "Orchestration and tracking").

    uv run python replication_runner.py replication/jobs.yaml            # on the node, under pod.py run
    uv run python replication_runner.py replication/jobs.yaml --dry-run  # print the plan: jobs, GPUs, downloads, done

The jobs file:

    gpus: 8                     # GPUs on the node
    disk_budget_gb: 3800        # weights on disk at once (downloaded + downloading), HF cache only
    max_downloads: 2            # repos downloading at once
    score_python: /root/vllm-env/bin/python   # the interpreter for score_replication.py (scripts/node_setup.sh)
    jobs:
      - name: llama-405b-base
        config: configs/replication/llama-405b.yaml
        stage: base             # runs score_replication.py <config> --stage base (args: [extra flags])
                                # gpus: from the config's vllm tensor_parallel_size x pipeline_parallel_size
      - name: patch-llama-405b
        cmd: "uv run python report_patch.py ..."   # any other command
        repos: [meta-llama/Llama-3.1-405B-Instruct]
        done_marker: outputs/...                    # it is done when this path exists
        gpus: 8

Jobs start in the listed order, each as soon as its weights are on disk and enough GPUs are free (no job jumps the queue,
so an 8-GPU job is never starved by smaller ones behind it). Each runs with its own CUDA_VISIBLE_DEVICES and logs to
logs/<name>.log. Repos download ahead of time, in job order, within the disk budget; a repo's weights are deleted once no
unfinished job needs them (after a failure they stay, for a retry). A scoring job whose results files all exist counts as
done and is skipped, so the runner can be relaunched after a crash.

Log markers (replication/runner.log and stdout): RUNNER_START, DOWNLOAD_START/DONE/FAILED, JOB_START, JOB_EXIT, DELETE,
RUNNER_DONE.
"""

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from replication import expand_conditions

LOG = Path("replication/runner.log")
os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")  # no second copy of a 1 TB model in the xet cache
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_jobs(path):
    spec = yaml.safe_load(open(path))
    jobs, configs = [], {}
    for j in spec["jobs"]:
        j = dict(j)
        if "config" in j:
            cfg = configs.setdefault(j["config"], yaml.safe_load(open(j["config"])))
            (m,) = [x for x in cfg["models"] if x["stage"] == j["stage"]]
            j["repos"] = [m["name"]]
            j["download"] = {m["name"]: {**m.get("download", {}), "revision": m.get("revision")}}
            j["results"] = [Path("outputs/replication") / cfg["run_name"] / "results" / f"{j['stage']}__{c}.csv.gz"
                            for c in expand_conditions(m["conditions"])]
            v = m.get("vllm", {})
            n = v.get("tensor_parallel_size", 1) * v.get("pipeline_parallel_size", 1)
            assert j.setdefault("gpus", n) == n, f"{j['name']}: gpus {j['gpus']} but the config's vLLM uses {n}"
            j["cmd"] = " ".join([spec.get("score_python", "uv run python"), "score_replication.py", j["config"],
                                 "--stage", j["stage"], *map(str, j.get("args", []))])
        else:
            j.setdefault("repos", [])
            j.setdefault("download", {})
        jobs.append(j)
    return spec, jobs


def job_done(j):
    if "results" in j:
        return all(p.exists() for p in j["results"])
    return "done_marker" in j and Path(j["done_marker"]).exists()


def repo_dir(repo):
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return hf_home / "hub" / ("models--" + repo.replace("/", "--"))


class Downloader:
    """Background snapshot_download per repo; state per repo: pending / downloading / done / failed / deleted."""

    def __init__(self, dry_run=False):
        self.state, self.lock, self.dry_run = {}, threading.Lock(), dry_run

    def start(self, repo, spec):
        with self.lock:
            self.state[repo] = "downloading"
        log(f"DOWNLOAD_START {repo} gb={spec.get('gb', '?')}")
        threading.Thread(target=self._run, args=(repo, spec), daemon=True).start()

    def _run(self, repo, spec):
        t0 = time.time()
        for attempt in range(3):
            try:
                if not self.dry_run:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo, revision=spec.get("revision"), allow_patterns=spec.get("allow_patterns"),
                                      max_workers=16)
                    shutil.rmtree(Path(os.environ.get("HF_HOME", "")) / "xet", ignore_errors=True)  # chunk cache copy
                with self.lock:
                    self.state[repo] = "done"
                log(f"DOWNLOAD_DONE {repo} minutes={(time.time() - t0) / 60:.1f}")
                return
            except Exception as e:  # noqa: BLE001 - network errors of every kind; retried, then reported
                log(f"DOWNLOAD_RETRY {repo} attempt={attempt + 1} error={e!r}"[:500])
                time.sleep(30)
        with self.lock:
            self.state[repo] = "failed"
        log(f"DOWNLOAD_FAILED {repo}")

    def get(self, repo):
        with self.lock:
            return self.state.get(repo, "pending")

    def delete(self, repo):
        with self.lock:
            self.state[repo] = "deleted"
        if not self.dry_run:
            shutil.rmtree(repo_dir(repo), ignore_errors=True)
        log(f"DELETE {repo}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--fake", action="store_true", help="no downloads or deletes (tests)")
    ap.add_argument("--poll", type=float, default=10)
    a = ap.parse_args(argv)
    spec, jobs = load_jobs(a.jobs)
    n_gpus, budget = spec["gpus"], spec.get("disk_budget_gb", 3800)
    sizes = {r: float(j["download"].get(r, {}).get("gb", 0)) for j in jobs for r in j["repos"]}

    if a.dry_run:
        for j in jobs:
            print(f"{j['name']:40} gpus={j['gpus']} done={job_done(j)} repos={j['repos']} cmd: {j['cmd']}")
        print(f"weights: {sum(sizes.values()):.0f} GB over {len(sizes)} repos; budget {budget} GB at once")
        return 0

    log(f"RUNNER_START jobs={len(jobs)} gpus={n_gpus} budget_gb={budget}")
    dl = Downloader(dry_run=a.fake)
    for j in jobs:  # already finished: never download for them
        if job_done(j):
            j["status"] = "done"
            log(f"JOB_SKIP {j['name']} (results exist)")
    for r in sizes:  # weights already on disk from an earlier launch
        if repo_dir(r).exists() and not a.fake:
            dl.state[r] = "done"
    free, running = set(range(n_gpus)), {}
    Path("logs").mkdir(exist_ok=True)

    def needed(repo):  # by a job that hasn't finished successfully
        return any(repo in j["repos"] and j.get("status") not in ("done",) for j in jobs)

    while True:
        # Downloads, in job order, within the budget.
        on_disk = sum(sizes[r] for r in sizes if dl.get(r) in ("downloading", "done", "failed"))
        active = sum(dl.get(r) == "downloading" for r in sizes)
        for j in jobs:
            if j.get("status"):
                continue
            for r in j["repos"]:
                if dl.get(r) == "pending" and active < spec.get("max_downloads", 2) and on_disk + sizes[r] <= budget:
                    dl.start(r, j["download"].get(r, {}))
                    on_disk += sizes[r]
                    active += 1
        # Finished jobs.
        for name, (proc, gpus, j, t0) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            del running[name]
            free |= gpus
            j["status"] = "done" if code == 0 and (job_done(j) or "results" not in j and "done_marker" not in j) else "failed"
            log(f"JOB_EXIT {name} code={code} status={j['status']} minutes={(time.time() - t0) / 60:.1f}")
            for r in j["repos"]:
                if not needed(r) and dl.get(r) == "done":
                    dl.delete(r)
        # Start the next job in order, if it can.
        nxt = next((j for j in jobs if not j.get("status")), None)
        if nxt is not None:
            states = [dl.get(r) for r in nxt["repos"]]
            if "failed" in states:
                nxt["status"] = "failed"
                log(f"JOB_EXIT {nxt['name']} code=- status=failed (download failed)")
                continue
            if all(s == "done" for s in states) and len(free) >= nxt["gpus"]:
                gpus = set(sorted(free)[:nxt["gpus"]])
                free -= gpus
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, sorted(gpus)))}
                logf = open(Path("logs") / f"{nxt['name']}.log", "a")
                proc = subprocess.Popen(nxt["cmd"], shell=True, env=env, stdout=logf, stderr=subprocess.STDOUT)
                nxt["status"] = "running"
                running[nxt["name"]] = (proc, gpus, nxt, time.time())
                log(f"JOB_START {nxt['name']} gpus={sorted(gpus)} cmd: {nxt['cmd']}")
                continue
        if not running and all(j.get("status") in ("done", "failed") for j in jobs):
            failed = [j["name"] for j in jobs if j["status"] == "failed"]
            log(f"RUNNER_DONE failed={failed}")
            return 1 if failed else 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
