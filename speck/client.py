import argparse
import json
import logging
from pathlib import Path
import random
import time
import urllib.error

import torch

from .backends import ann_result, executor_result
from .wire import atomic_save, Transport


def run(args):
    credentials = json.loads(Path(args.credentials).read_text())
    transport = Transport(args.server, credentials["token"], proxy=args.proxy)
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    cache_path = state / "pending.pt"
    pending = torch.load(cache_path, map_location="cpu", weights_only=True) if cache_path.exists() else None
    torch.set_num_threads(1)
    while True:
        network_operation = True
        try:
            job = transport.request("/job")
            if job["status"] == "done":
                print(json.dumps(dict(status="done", rounds=job["round"])), flush=True)
                return
            if job["status"] == "failed":
                raise RuntimeError("Coordinator stopped after a round failure; inspect history.json")
            if job["status"] == "wait":
                time.sleep(args.poll_seconds)
                continue
            if job["client_id"] != credentials["id"] or job["role"] != credentials["role"]:
                raise RuntimeError("Credential ID/role differs from coordinator")
            if pending is None or pending["version"] != job["version"]:
                network_operation = False
                seed = job["spec"]["options"]["seed"] + 1009 * job["round"] + job["shard"]
                random.seed(seed)
                torch.manual_seed(seed)
                start = time.perf_counter()
                if job["role"] == "ann":
                    if args.device.startswith("cuda") and not torch.cuda.is_available():
                        raise RuntimeError("CUDA requested but unavailable; install the correct PyTorch build")
                    result = ann_result(job, args.device, args.data_root, args.indices)
                else:
                    result = executor_result(job, job["executor_mode"], args.data_root,
                                             args.indices, args.speck_device)
                result["compute_seconds"] = time.perf_counter() - start
                pending = dict(client_id=credentials["id"], round=job["round"],
                               version=job["version"], result=result)
                atomic_save(pending, cache_path)
            network_operation = True
            acknowledgement = transport.request("/result", pending)
            print(json.dumps(dict(client_id=credentials["id"], round=job["round"],
                                  status=acknowledgement["status"],
                                  compute_seconds=pending["result"]["compute_seconds"])), flush=True)
            pending = None
            cache_path.unlink(missing_ok=True)
        except urllib.error.HTTPError as error:
            if error.code == 409:
                pending = None
                cache_path.unlink(missing_ok=True)
                logging.warning("Round advanced; dropping stale result and resynchronizing")
            elif error.code in {401, 403, 400, 404}:
                raise RuntimeError(f"Coordinator rejected request ({error.code}): {error.read().decode()}") from error
            else:
                logging.warning("HTTP error %s; retrying", error.code)
            time.sleep(args.poll_seconds)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            if not network_operation:
                raise
            logging.warning("Connection unavailable (%s); retrying", error)
            time.sleep(args.poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SyNAX ANN or Speck worker")
    parser.add_argument("--server", required=True)
    parser.add_argument("--proxy", help="Local Tailscale userspace HTTP proxy, e.g. http://127.0.0.1:1055")
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--state-dir", required=True, help="Unique writable directory for this worker")
    parser.add_argument("--device", default="cuda:0", help="ANN training device; executors use their selected backend")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--indices", help="JSON list of local TRAINING sample indices")
    parser.add_argument("--speck-device", default="speck2fdevkit:1")
    parser.add_argument("--poll-seconds", type=float, default=2)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    run(args)


if __name__ == "__main__":
    main()
