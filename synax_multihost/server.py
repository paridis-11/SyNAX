import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import math
from pathlib import Path
import random
import threading
import time
import uuid

import torch

from .backends import (convert, cpu_state, initial_state, new_model,
                       parameter_delta)
from .algorithm import global_state_update, sample_clients, scheduled_lr, snn_guidance
from .wire import MAX_BODY, atomic_save, digest, pack, private_address, unpack


class Conflict(ValueError):
    pass


class Federation:
    def __init__(self, roster, backend, checkpoint, options, output, rounds,
                 timeout=7200, min_ann=None, min_executors=None, executor_mode=None,
                 resume=False):
        self.lock = threading.RLock()
        self.roster = roster
        if (not roster or len({item["id"] for item in roster}) != len(roster)
                or len({item["token"] for item in roster}) != len(roster)
                or any(item["role"] not in {"ann", "executor"} or len(item["token"]) < 32
                       for item in roster)):
            raise ValueError("Roster requires unique IDs, unique tokens and valid roles")
        self.ann_count = sum(item["role"] == "ann" for item in roster)
        self.executor_count = len(roster) - self.ann_count
        self.min_ann, self.min_executors = min_ann, min_executors
        if (min_ann is not None and not 1 <= min_ann <= self.ann_count
                or min_executors is not None and not 0 <= min_executors <= self.executor_count
                or self.ann_count == 0):
            raise ValueError("Invalid ANN/executor quorum")
        if rounds < 1 or timeout <= 0:
            raise ValueError("rounds and timeout must be positive")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.rounds, self.timeout = rounds, timeout
        self.executor_mode = executor_mode or ("speck" if backend == "event" else "toy")
        if (backend == "toy" and self.executor_mode != "toy"
                or backend == "event" and self.executor_mode not in {"speck", "sinabs"}):
            raise ValueError("Executor mode incompatible with backend")
        self.round, self.results, self.receipts, self.history = 0, {}, {}, []
        self.status, self.version = "active", uuid.uuid4().hex
        state_path = self.output / "state.pt"
        if resume:
            saved = torch.load(state_path, map_location="cpu", weights_only=True)
            if saved["members"] != [(item["id"], item["role"]) for item in roster]:
                raise ValueError("Resume roster differs from saved roster")
            if saved["spec"]["backend"] != backend or saved["executor_mode"] != self.executor_mode:
                raise ValueError("Resume backend/mode differs from checkpoint")
            self.spec = saved["spec"]
            if self.spec.get("algorithm_version") != "synax-static-v2":
                raise ValueError("Checkpoint predates static-aligned round policies; start a new run")
            if saved["total_rounds"] != rounds:
                raise ValueError("Changing total rounds changes the cosine schedule; start a new run")
            if any(self.spec["options"].get(name) != value for name, value in options.items()):
                raise ValueError("Resume options differ from saved run; start a new run to change them")
            self.model = new_model(self.spec)
            self.model.load_state_dict(saved["ann_state"], strict=True)
            self.snn = convert(self.model, self.spec)
            self.snn.load_state_dict(saved["snn_state"], strict=False)
            for name in ("round", "status", "version", "results", "receipts", "history",
                         "ann_guidance", "snn_guidance", "sampled"):
                setattr(self, name, saved[name])
            self.sampling_rng = random.Random()
            self.sampling_rng.setstate(saved["sampling_rng"])
            if self.status == "failed":
                self.status, self.version, self.results = "active", uuid.uuid4().hex, {}
            if self.round >= rounds:
                self.status = "done"
            elif self.status == "done":
                self.status = "active"
        else:
            if state_path.exists():
                raise ValueError("Output contains a checkpoint; use --resume or a new directory")
            self.spec, self.model, self.snn = initial_state(backend, checkpoint, options)
            self.ann_guidance = {name: torch.zeros_like(value.detach().cpu())
                                 for name, value in self.model.named_parameters()}
            self.snn_guidance = {name: torch.zeros_like(value.detach().cpu())
                                 for name, value in cpu_state(self.snn).items() if value.is_floating_point()}
            self.sampling_rng = random.Random(self.spec["options"]["seed"])
            self.sampled = sample_clients(roster, self.spec["options"]["sample_ratio"], self.sampling_rng)
        self.deadline = time.time() + timeout
        self.save()

    def save(self):
        value = {name: getattr(self, name) for name in (
            "spec", "round", "status", "version", "results", "receipts", "history",
            "ann_guidance", "snn_guidance", "executor_mode", "sampled")}
        value.update(ann_state=cpu_state(self.model), snn_state=cpu_state(self.snn),
                     members=[(item["id"], item["role"]) for item in self.roster],
                     sampling_rng=self.sampling_rng.getstate(), total_rounds=self.rounds)
        atomic_save(value, self.output / "state.pt")
        if self.spec["backend"] == "event":
            atomic_save(dict(round=self.round, args=self.spec["options"],
                             algorithm_version=self.spec["algorithm_version"],
                             ann_model=value["ann_state"], snn_model=value["snn_state"]),
                        self.output / "latest.pt")
        temporary = self.output / "history.json.tmp"
        temporary.write_text(json.dumps(self.history, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output / "history.json")

    def member(self, authorization):
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            return None
        token = authorization[len(prefix):]
        for item in self.roster:
            if hmac.compare_digest(token, item["token"]):
                return item
        return None

    def check_deadline(self):
        if self.status != "active" or time.time() <= self.deadline:
            return
        ann = sum(self.role(client) == "ann" for client in self.results)
        executors = len(self.results) - ann
        required_ann = (sum(self.role(client) == "ann" for client in self.sampled)
                        if self.min_ann is None else self.min_ann)
        required_executors = (sum(self.role(client) == "executor" for client in self.sampled)
                              if self.min_executors is None else self.min_executors)
        if ann >= required_ann and executors >= required_executors:
            self.advance()
        else:
            self.status = "failed"
            self.history.append(dict(round=self.round, status="timeout",
                                     missing=[client for client in self.sampled if client not in self.results]))
            self.save()

    def role(self, client):
        return next(item["role"] for item in self.roster if item["id"] == client)

    def job(self, member):
        with self.lock:
            self.check_deadline()
            if self.status != "active":
                return dict(status=self.status, round=self.round)
            if member["id"] in self.results or member["id"] not in self.sampled:
                return dict(status="wait", round=self.round)
            common = dict(status="job", version=self.version, round=self.round,
                          client_id=member["id"], role=member["role"], spec=self.spec,
                          shard=self.roster.index(member), shards=len(self.roster),
                          round_lr=scheduled_lr(self.spec["options"]["lr"], self.round, self.rounds))
            if member["role"] == "ann":
                common.update(ann_state=cpu_state(self.model), ann_guidance=self.ann_guidance)
            else:
                common.update(snn_state=cpu_state(self.snn), snn_guidance=self.snn_guidance,
                              executor_mode=self.executor_mode)
            return common

    def accept(self, member, value, body_hash):
        with self.lock:
            self.check_deadline()
            key = value.get("version", "") + ":" + member["id"]
            if key in self.receipts:
                if self.receipts[key] != body_hash:
                    raise Conflict("Conflicting duplicate result")
                return dict(status="accepted", duplicate=True)
            if (self.status != "active" or value.get("version") != self.version
                    or value.get("round") != self.round or value.get("client_id") != member["id"]
                    or member["id"] not in self.sampled):
                raise Conflict("Stale round or mismatched client; fetch a fresh job")
            result = value["result"]
            samples = result.get("samples")
            if type(samples) is not int or samples < 1:
                raise ValueError("A result must contain a positive sample count")
            if member["role"] == "ann":
                delta = result.get("delta", {})
                if set(delta) != set(self.ann_guidance):
                    raise ValueError("Parameter names mismatch")
                for name, reference in self.ann_guidance.items():
                    value_tensor = delta[name]
                    if (not isinstance(value_tensor, torch.Tensor) or value_tensor.shape != reference.shape
                            or not value_tensor.is_floating_point() or not torch.isfinite(value_tensor).all()):
                        raise ValueError(f"Invalid parameter update: {name}")
            else:
                alpha = result.get("alpha")
                if type(alpha) not in {int, float} or not math.isfinite(alpha):
                    raise ValueError("Non-finite or missing alpha")
                if result.get("mode") != self.executor_mode:
                    raise ValueError("Executor backend does not match coordinator")
            self.results[member["id"]] = result
            self.receipts[key] = body_hash
            if len(self.results) == len(self.sampled):
                try:
                    self.advance()
                except Exception as error:
                    self.fail_round(error)
                    raise
            else:
                self.save()
            return dict(status="accepted", duplicate=False)

    def fail_round(self, error):
        self.status = "failed"
        self.history.append(dict(round=self.round, status="error", error=str(error)))
        self.save()

    def advance(self):
        ann = [value for key, value in self.results.items() if self.role(key) == "ann"]
        executors = [value for key, value in self.results.items() if self.role(key) == "executor"]
        options = self.spec["options"]
        old_ann, old_snn = cpu_state(self.model), cpu_state(self.snn)
        alpha_mean = sum(value["alpha"] for value in executors) / len(executors) if executors else 0.0
        alpha = max(options["zero_min"], min(options["zero_max"], alpha_mean))
        candidate = new_model(self.spec)
        average = {name: sum(value["delta"][name] for value in ann) / len(ann)
                   for name in self.ann_guidance}
        candidate.load_state_dict(global_state_update(old_ann, dict(candidate.named_parameters()),
                                  average, self.ann_guidance, alpha,
                                  options["eta_ann"], options["eta_snn"]), strict=True)
        if any(not torch.isfinite(value).all() for value in candidate.parameters()):
            raise ValueError("Aggregated model contains non-finite parameters")
        next_snn = convert(candidate, self.spec)
        next_guidance = snn_guidance(old_snn, cpu_state(next_snn))
        self.ann_guidance = parameter_delta(old_ann, cpu_state(candidate), self.ann_guidance)
        self.model, self.snn, self.snn_guidance = candidate, next_snn, next_guidance
        summary = dict(round=self.round, status="completed", alpha_mean=alpha_mean,
                       alpha_applied=alpha, ann_clients=len(ann), executor_clients=len(executors),
                       sampled=list(self.sampled),
                       missing=[client for client in self.sampled if client not in self.results],
                       executor_results=executors)
        self.history.append(summary)
        print(json.dumps(summary, allow_nan=False), flush=True)
        completed_version = self.version
        self.round += 1
        self.version, self.results = uuid.uuid4().hex, {}
        self.receipts = {key: value for key, value in self.receipts.items()
                         if key.startswith(completed_version + ":")}
        self.status = "done" if self.round >= self.rounds else "active"
        if self.status == "active":
            self.sampled = sample_clients(self.roster, options["sample_ratio"], self.sampling_rng)
        self.deadline = time.time() + self.timeout
        self.save()


def make_server(federation, host, port):
    if not private_address(host) or ipaddress.ip_address(host).version != 4:
        raise ValueError("Bind to a specific IPv4 LAN/Tailscale/loopback IP, not 0.0.0.0")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, value, binary=False):
            body = pack(value) if binary else json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/x-synax-tensors" if binary else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            member = federation.member(self.headers.get("Authorization"))
            if member is None:
                self.reply(401, dict(error="Unauthorized"))
            elif self.path == "/job":
                self.reply(200, federation.job(member), binary=True)
            elif self.path == "/status":
                with federation.lock:
                    federation.check_deadline()
                    self.reply(200, dict(status=federation.status, round=federation.round,
                                         submitted=list(federation.results), sampled=federation.sampled))
            else:
                self.reply(404, dict(error="Unknown endpoint"))

        def do_POST(self):
            member = federation.member(self.headers.get("Authorization"))
            if member is None:
                self.reply(401, dict(error="Unauthorized"))
                return
            if self.path != "/result":
                self.reply(404, dict(error="Unknown endpoint"))
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid message length")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete message")
                value = unpack(body)
                if not isinstance(value, dict):
                    raise ValueError("Expected a result dictionary")
                self.reply(200, federation.accept(member, value, digest(body)))
            except Conflict as error:
                self.reply(409, dict(error=str(error)))
            except Exception as error:
                self.reply(400, dict(error=str(error)))

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        def service_actions(self):
            with federation.lock:
                try:
                    federation.check_deadline()
                except Exception as error:
                    logging.exception("Round commit failed")
                    federation.fail_round(error)

        def get_request(self):
            connection, address = super().get_request()
            connection.settimeout(30)
            return connection, address

    return Server((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SyNAX multi-host coordinator")
    parser.add_argument("--roster", required=True)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--backend", choices=["toy", "event"], default="event")
    parser.add_argument("--checkpoint")
    parser.add_argument("--options", help="JSON hyperparameter overrides")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--round-timeout", type=float, default=7200)
    parser.add_argument("--min-ann", type=int)
    parser.add_argument("--min-executors", type=int)
    parser.add_argument("--executor-mode", choices=["speck", "sinabs", "toy"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    roster = json.loads(Path(args.roster).read_text())
    options = json.loads(Path(args.options).read_text()) if args.options else {}
    federation = Federation(roster, args.backend, args.checkpoint, options, args.output,
                            args.rounds, args.round_timeout, args.min_ann, args.min_executors,
                            args.executor_mode, args.resume)
    server = make_server(federation, args.bind, args.port)
    print(json.dumps(dict(listening=f"http://{args.bind}:{server.server_port}",
                         backend=args.backend, executor_mode=federation.executor_mode,
                         ann_clients=federation.ann_count, executor_clients=federation.executor_count)), flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
