"""replication_runner.py with fake downloads and shell commands: order, GPU assignment, failures, the disk budget."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import replication_runner as rr  # noqa: E402


def _jobs(tmp_path, jobs, **kw):
    path = tmp_path / "jobs.yaml"
    path.write_text(yaml.safe_dump({"gpus": 8, "disk_budget_gb": kw.get("budget", 3000), "jobs": jobs}))
    return str(path)


def _job(name, gpus, repos=(), code=0, sleep=0.3):
    return dict(name=name, gpus=gpus, repos=list(repos), download={r: {"gb": 1000} for r in repos},
                cmd=f"echo $CUDA_VISIBLE_DEVICES > {name}.gpus; sleep {sleep}; exit {code}")


def _events(tmp_path):
    return [line.split()[1:3] for line in (tmp_path / "replication/runner.log").read_text().splitlines()]


def test_order_gpus_and_failures(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = [_job("a-base", 4, ["org/a-base"]), _job("a-final", 4, ["org/a-final"]),
            _job("big", 8, ["org/big"], code=3), _job("after", 2)]
    assert rr.main([_jobs(tmp_path, jobs), "--fake", "--poll", "0.05"]) == 1  # one job failed
    assert Path("a-base.gpus").read_text().strip() == "0,1,2,3"
    assert Path("a-final.gpus").read_text().strip() == "4,5,6,7"  # the two halves ran side by side
    assert Path("big.gpus").read_text().strip() == "0,1,2,3,4,5,6,7"
    ev = _events(tmp_path)
    starts = [e[1] for e in ev if e[0] == "JOB_START"]
    assert starts == ["a-base", "a-final", "big", "after"]  # in order; the failure doesn't stop later jobs
    # "big" starts only after both halves exited
    i_big = ev.index(["JOB_START", "big"])
    assert ["JOB_EXIT", "a-base"] in ev[:i_big] and ["JOB_EXIT", "a-final"] in ev[:i_big]
    # weights of finished jobs are deleted; the failed job's stay for a retry
    deleted = {e[1] for e in ev if e[0] == "DELETE"}
    assert deleted == {"org/a-base", "org/a-final"}
    assert "RUNNER_DONE" in [e[0] for e in ev]


def test_disk_budget_limits_downloads(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = [_job("one", 8, ["org/one"]), _job("two", 8, ["org/two"]), _job("three", 8, ["org/three"])]
    assert rr.main([_jobs(tmp_path, jobs, budget=2000), "--fake", "--poll", "0.05"]) == 0
    ev = _events(tmp_path)
    # "three" can't start downloading until "one" is deleted (2 x 1000 GB fit in the budget, not 3)
    assert ev.index(["DOWNLOAD_START", "org/three"]) > ev.index(["DELETE", "org/one"])


def test_scoring_jobs_resolve_from_configs(tmp_path):
    cfg = {"run_name": "x", "models": [{"name": "org/x-base", "stage": "base", "conditions": "base",
                                         "vllm": {"tensor_parallel_size": 4},
                                         "download": {"gb": 10, "allow_patterns": ["*.safetensors"]}}]}
    (tmp_path / "x.yaml").write_text(yaml.safe_dump(cfg))
    path = _jobs(tmp_path, [dict(name="x-base", config=str(tmp_path / "x.yaml"), stage="base")])
    _, (j,) = rr.load_jobs(path)
    assert j["repos"] == ["org/x-base"] and len(j["results"]) == 4 and j["gpus"] == 4
    bad = _jobs(tmp_path, [dict(name="x-base", config=str(tmp_path / "x.yaml"), stage="base", gpus=8)])
    import pytest
    with pytest.raises(AssertionError):
        rr.load_jobs(bad)
    assert j["cmd"].endswith(f"score_replication.py {tmp_path / 'x.yaml'} --stage base")
    assert str(j["results"][0]) == "outputs/replication/x/results/base__assistant.csv.gz"


def test_existing_repo_directories_are_not_taken_as_downloaded(tmp_path, monkeypatch):
    """A directory left by a tokenizer download must not count as the weights (or against the budget)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    for r in ("org/big1", "org/big2"):
        rr.repo_dir(r).mkdir(parents=True)
    jobs = [_job("first", 4, ["org/first"]), _job("big1", 8, ["org/big1"]), _job("big2", 8, ["org/big2"])]
    assert rr.main([_jobs(tmp_path, jobs, budget=2000), "--fake", "--poll", "0.05"]) == 0
    ev = _events(tmp_path)
    assert ["DOWNLOAD_START", "org/first"] in ev and ["DOWNLOAD_START", "org/big1"] in ev


def test_delete_frees_shared_blobs_but_keeps_ones_still_used(tmp_path, monkeypatch):
    """huggingface_hub 1.32's layout: blobs in a shared hub/blobs/, snapshots link to them."""
    import os
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    blobs = tmp_path / "hf" / "hub" / "blobs" / "ab"
    blobs.mkdir(parents=True)
    for name, size in (("w1", 100), ("w2", 200), ("shared", 50)):
        (blobs / name).write_bytes(b"x" * size)
    for repo, files in (("org/a", ["w1", "w2", "shared"]), ("org/b", ["shared"])):
        snap = rr.repo_dir(repo) / "snapshots" / "rev"
        snap.mkdir(parents=True)
        for f in files:
            os.symlink(blobs / f, snap / f"{f}.safetensors")
    assert rr.delete_repo_weights("org/a") == 300
    assert not (blobs / "w1").exists() and not (blobs / "w2").exists() and (blobs / "shared").exists()
    assert not rr.repo_dir("org/a").exists() and rr.repo_dir("org/b").exists()
