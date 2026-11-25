"""Launch a RunPod pod for one job through the GraphQL API (the REST host is not always reachable).

The pod runs `bash -c START` where START fetches a bootstrap from S3. Credentials are passed as pod env vars from
this machine's environment: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, and WANDB_API_KEY (from
WANDB_API_KEY or WANDB). RUNPOD_API_KEY is read from the environment when present (the RunPod API is also reached
through an authenticating proxy here, so it may be absent).

    python scripts/runpod/launch.py cpu  --name b0-build --bootstrap s3://xray-hands/work/b0/build/bootstrap.sh
    python scripts/runpod/launch.py gpu  --name b0-encode --gpu "NVIDIA A100 80GB PCIe" --bootstrap s3://.../boot.sh
    python scripts/runpod/launch.py list
    python scripts/runpod/launch.py terminate --pod-id XXX
"""
import argparse
import json
import os
import sys
import urllib.request

API = "https://api.runpod.io/graphql"


def gql(query, variables=None):
    req = urllib.request.Request(API, json.dumps({"query": query, "variables": variables or {}}).encode(),
                                 {"Content-Type": "application/json"})
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    out = json.load(urllib.request.urlopen(req, timeout=60))
    if out.get("errors"):
        raise SystemExit(json.dumps(out["errors"], indent=1))
    return out["data"]


def pod_env(extra):
    env = {k: os.environ[k] for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION")
           if k in os.environ}
    wb = os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB")
    if wb:
        env["WANDB_API_KEY"] = wb
    env.update(extra)
    return [{"key": k, "value": v} for k, v in env.items()]


def start_cmd(bootstrap, pre=""):
    return (f"bash -c \"{pre}(command -v aws || pip install -q awscli) >/dev/null 2>&1; "
            f"aws s3 cp {bootstrap} /b.sh && bash /b.sh\"")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["cpu", "gpu", "list", "terminate", "stop"])
    ap.add_argument("--name", default="job")
    ap.add_argument("--bootstrap")
    ap.add_argument("--image", default=None)
    ap.add_argument("--gpu", default="NVIDIA A100 80GB PCIe")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud", default="ALL", help="SECURE, COMMUNITY or ALL")
    ap.add_argument("--cpu-instance", default="cpu3g-16-64")
    ap.add_argument("--disk", type=int, default=100, help="container disk GB")
    ap.add_argument("--min-vcpu", type=int, default=8)
    ap.add_argument("--min-mem", type=int, default=60)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE, repeatable")
    ap.add_argument("--pod-id")
    a = ap.parse_args()
    extra = dict(e.split("=", 1) for e in a.env)

    if a.what == "list":
        d = gql("query { myself { clientBalance currentSpendPerHr pods { id name desiredStatus costPerHr "
                "machine { gpuDisplayName } runtime { uptimeInSeconds } } } }")
        print(json.dumps(d["myself"], indent=1))
        return
    if a.what in ("terminate", "stop"):
        m = "podTerminate" if a.what == "terminate" else "podStop"
        print(gql(f'mutation {{ {m}(input: {{podId: "{a.pod_id}"}}) {"{ id }" if m == "podStop" else ""} }}'))
        return
    if not a.bootstrap:
        sys.exit("--bootstrap is required")
    if a.what == "cpu":
        inp = {"instanceId": a.cpu_instance, "cloudType": "SECURE", "containerDiskInGb": a.disk,
               "imageName": a.image or "python:3.11-bookworm", "name": a.name, "startSsh": False,
               "dockerArgs": start_cmd(a.bootstrap, "pip install -q awscli >/dev/null 2>&1; "),
               "env": pod_env(extra)}
        d = gql("mutation($i: deployCpuPodInput!) { deployCpuPod(input: $i) { id costPerHr machineId } }", {"i": inp})
        print(json.dumps(d, indent=1))
    else:
        inp = {"cloudType": a.cloud, "gpuCount": a.gpu_count, "gpuTypeId": a.gpu, "containerDiskInGb": a.disk,
               "volumeInGb": 0, "minVcpuCount": a.min_vcpu, "minMemoryInGb": a.min_mem,
               "imageName": a.image or "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", "name": a.name,
               "dockerArgs": start_cmd(a.bootstrap, "apt-get -qq update >/dev/null && apt-get -qq install -y "
                                                    "python3-pip curl git >/dev/null && "),
               "env": pod_env(extra)}
        d = gql("mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) "
                "{ id costPerHr machineId machine { gpuDisplayName } } }", {"i": inp})
        print(json.dumps(d, indent=1))


if __name__ == "__main__":
    main()
