import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch
import urllib.error

import torch

from speck import backends
from speck.provision import provision
from speck.server import Conflict, Federation, make_server
from speck.wire import atomic_save, digest, pack, Transport, unpack


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.roster = provision(self.root / "credentials", 1, 1)
        self.engine = Federation(self.roster, "toy", None, {}, self.root / "run", 3)

    def tearDown(self):
        self.directory.cleanup()

    def result(self, member, alpha=0.25):
        job = self.engine.job(member)
        if member["role"] == "ann":
            result = dict(delta={name: torch.full_like(value, 0.1)
                                 for name, value in job["ann_guidance"].items()}, samples=10)
        else:
            result = dict(alpha=alpha, mode="toy", samples=10)
        return dict(client_id=member["id"], round=job["round"], version=job["version"], result=result)

    def submit(self, member, result):
        body = pack(result)
        return self.engine.accept(member, unpack(body), digest(body))

    def test_round_update_and_feedback_sign(self):
        before = backends.cpu_state(self.engine.model)
        self.submit(self.roster[0], self.result(self.roster[0]))
        self.submit(self.roster[1], self.result(self.roster[1]))
        for name, value in before.items():
            torch.testing.assert_close(self.engine.model.state_dict()[name], value + 0.1)
            torch.testing.assert_close(self.engine.ann_guidance[name], torch.full_like(value, -0.1))
        self.assertAlmostEqual(sum(float(x.square().sum()) for x in backends.normalized(self.engine.snn_guidance).values()), 1, places=5)
        for name, value in backends.normalized(self.engine.ann_guidance).items():
            torch.testing.assert_close(backends.normalized(self.engine.snn_guidance)[name], value)
        before = backends.cpu_state(self.engine.model)
        self.submit(self.roster[0], self.result(self.roster[0]))
        self.submit(self.roster[1], self.result(self.roster[1], alpha=0.25))
        count = sum(value.numel() for value in before.values())
        increment = 0.1 + self.engine.spec["options"]["eta_snn"] * 0.25 / math.sqrt(count)
        for name, value in before.items():
            torch.testing.assert_close(self.engine.model.state_dict()[name], value + increment)

    def test_feedback_is_descent_on_quadratic_objective(self):
        with torch.no_grad():
            for model in (self.engine.model, self.engine.snn):
                for parameter in model.parameters():
                    parameter.zero_()
        self.submit(self.roster[0], self.result(self.roster[0]))
        self.submit(self.roster[1], self.result(self.roster[1], alpha=0.0))
        job = self.engine.job(self.roster[1])
        epsilon = job["spec"]["options"]["zo_epsilon"]
        def objective(model):
            return sum(float(value.detach().double().square().sum()) / 2
                       for value in model.parameters())
        derivative = (objective(backends.perturbed_snn(job, 1))
                      - objective(backends.perturbed_snn(job, -1))) / (2 * epsilon)
        before = objective(self.engine.model)
        ann = self.result(self.roster[0])
        for value in ann["result"]["delta"].values():
            value.zero_()
        self.submit(self.roster[0], ann)
        self.submit(self.roster[1], self.result(self.roster[1], alpha=derivative))
        self.assertLess(objective(self.engine.model), before)

    def test_feedback_step_is_independent_of_guidance_magnitude(self):
        def advance_once(amplitude, output):
            engine = Federation(self.roster, "toy", None, {}, output, 1)
            engine.ann_guidance = {name: torch.full_like(value, amplitude)
                                   for name, value in engine.ann_guidance.items()}
            before = backends.cpu_state(engine.model)
            old = self.engine
            self.engine = engine
            try:
                ann = self.result(self.roster[0])
                for value in ann["result"]["delta"].values():
                    value.zero_()
                self.submit(self.roster[0], ann)
                self.submit(self.roster[1], self.result(self.roster[1], alpha=0.25))
            finally:
                self.engine = old
            return parameter_norm(backends.parameter_delta(backends.cpu_state(engine.model), before, engine.ann_guidance))
        def parameter_norm(delta):
            return math.sqrt(sum(float(value.double().square().sum()) for value in delta.values()))
        small = advance_once(0.1, self.root / "small-guidance")
        large = advance_once(10.0, self.root / "large-guidance")
        self.assertAlmostEqual(small, large, places=6)
        self.assertAlmostEqual(small, 0.1 * 0.25, places=6)

    def test_duplicate_and_stale(self):
        result = self.result(self.roster[0])
        self.submit(self.roster[0], result)
        self.assertTrue(self.submit(self.roster[0], result)["duplicate"])
        altered = copy.deepcopy(result)
        altered["result"]["samples"] = 11
        with self.assertRaises(Conflict):
            self.submit(self.roster[0], altered)
        self.submit(self.roster[1], self.result(self.roster[1]))
        self.assertTrue(self.submit(self.roster[0], result)["duplicate"])
        result["version"] = "old-version"
        with self.assertRaises(Conflict):
            self.submit(self.roster[0], result)

    def test_nonfinite_wrong_shape_and_backend_rejected(self):
        result = self.result(self.roster[0])
        name = next(iter(result["result"]["delta"]))
        result["result"]["delta"][name].fill_(float("nan"))
        with self.assertRaises(ValueError):
            self.submit(self.roster[0], result)
        result = self.result(self.roster[0])
        result["result"]["delta"][name] = torch.ones(1)
        with self.assertRaises(ValueError):
            self.submit(self.roster[0], result)
        with self.assertRaises(ValueError):
            self.submit(self.roster[1], self.result(self.roster[1], alpha=float("nan")))
        result = self.result(self.roster[1])
        result["result"]["mode"] = "speck"
        with self.assertRaises(ValueError):
            self.submit(self.roster[1], result)
        self.assertEqual(self.engine.results, {})

    def test_authentication(self):
        self.assertIsNone(self.engine.member("Bearer " + "x" * 32))
        self.assertIsNone(self.engine.member(None))
        self.assertEqual(self.engine.member("Bearer " + self.roster[0]["token"]), self.roster[0])

    def test_resume_preserves_pending_and_version(self):
        self.submit(self.roster[0], self.result(self.roster[0]))
        resumed = Federation(self.roster, "toy", None, {}, self.root / "run", 3, resume=True)
        self.assertEqual(resumed.version, self.engine.version)
        self.assertEqual(list(resumed.results), [self.roster[0]["id"]])
        self.assertEqual(resumed.job(self.roster[0])["status"], "wait")
        self.engine = resumed
        self.submit(self.roster[1], self.result(self.roster[1]))
        self.assertEqual(resumed.round, 1)

    def test_cached_result_keeps_duplicate_identity(self):
        result = self.result(self.roster[0])
        original = pack(result)
        atomic_save(result, self.root / "pending.pt")
        reloaded = torch.load(self.root / "pending.pt", weights_only=True)
        self.assertEqual(digest(original), digest(pack(reloaded)))

    def test_commit_failure_does_not_change_global_model(self):
        before = backends.cpu_state(self.engine.model)
        self.submit(self.roster[0], self.result(self.roster[0]))
        with patch("speck.server.convert", side_effect=ValueError("conversion failed")):
            with self.assertRaises(ValueError):
                self.submit(self.roster[1], self.result(self.roster[1]))
        self.assertEqual(self.engine.status, "failed")
        self.assertEqual(self.engine.round, 0)
        for name, value in before.items():
            torch.testing.assert_close(self.engine.model.state_dict()[name], value)

    def test_timeout_and_explicit_partial_round(self):
        self.engine.deadline = time.time() - 1
        self.assertEqual(self.engine.job(self.roster[0])["status"], "failed")
        resumed = Federation(self.roster, "toy", None, {}, self.root / "run", 3, resume=True)
        self.assertEqual(resumed.status, "active")
        self.assertNotEqual(resumed.version, self.engine.version)
        self.engine = Federation(self.roster, "toy", None, {}, self.root / "partial", 3,
                                 min_ann=1, min_executors=0)
        self.submit(self.roster[0], self.result(self.roster[0]))
        self.engine.deadline = time.time() - 1
        self.engine.check_deadline()
        self.assertEqual(self.engine.round, 1)
        self.assertEqual(self.engine.history[-1]["missing"], [self.roster[1]["id"]])

    def test_toy_forward_only_feedback(self):
        job = self.engine.job(self.roster[1])
        result = backends.executor_result(job, "toy", "", None, "")
        self.assertEqual(result["alpha"], 0)
        for value in job["snn_guidance"].values():
            value.fill_(0.05)
        result = backends.executor_result(job, "toy", "", None, "")
        self.assertTrue(math.isfinite(result["alpha"]))
        self.assertAlmostEqual(result["alpha"], (result["loss_plus"] - result["loss_minus"]) / 0.02)

    def test_http_auth_errors_and_tensor_round_trip(self):
        server = make_server(self.engine, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            transport = Transport(url, self.roster[0]["token"])
            job = transport.request("/job")
            self.assertEqual(job["client_id"], "ann-1")
            transport.request("/result", self.result(self.roster[0]))
            self.assertEqual(transport.request("/job")["status"], "wait")
            with self.assertRaises(urllib.error.HTTPError) as error:
                Transport(url, "x" * 32).request("/job")
            self.assertEqual(error.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_public_http_and_wildcard_bind_rejected(self):
        with self.assertRaises(ValueError):
            Transport("http://8.8.8.8:8765", self.roster[0]["token"])
        with self.assertRaises(ValueError):
            make_server(self.engine, "0.0.0.0", 0)

    def test_explicit_userspace_proxy_and_no_ambient_proxy(self):
        import urllib.request
        transport = Transport("http://100.64.0.1:8765", self.roster[0]["token"], proxy="http://127.0.0.1:1055")
        handlers = [handler for handler in transport.opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]
        self.assertEqual(handlers[0].proxies, {"http": "http://127.0.0.1:1055", "https": "http://127.0.0.1:1055"})
        with patch.dict(os.environ, {"HTTP_PROXY": "http://8.8.8.8:1234"}):
            transport = Transport("http://127.0.0.1:8765", self.roster[0]["token"])
            self.assertTrue(all(not handler.proxies for handler in transport.opener.handlers
                                if isinstance(handler, urllib.request.ProxyHandler)))
        with self.assertRaises(ValueError):
            Transport("http://100.64.0.1:8765", self.roster[0]["token"], proxy="http://8.8.8.8:1055")

    def test_automatic_partitions_are_disjoint(self):
        partitions = [backends.local_data(self.engine.spec, "", shard, 6)[2] for shard in range(6)]
        self.assertEqual(sorted(sum(partitions, [])), list(range(96)))
        self.assertEqual(len(set(sum(partitions, []))), 96)

    def test_nmnist_order_is_canonical_across_hosts(self):
        class Raw:
            data = ["/train/NMNIST/Train/1/b.bin", "/train/NMNIST/Train/0/a.bin"]
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return self.data[index], index
        helper = types.SimpleNamespace(NMNIST=lambda **kwargs: Raw())
        spec = dict(backend="event", options=dict(backends.DEFAULTS, dataset="nmnist"))
        with patch.object(backends, "event_module", return_value=helper):
            _, raw, indices = backends.local_data(spec, "/train", 0, 1)
        self.assertEqual(raw[0][0], "/train/NMNIST/Train/0/a.bin")
        self.assertEqual(raw[1][0], "/train/NMNIST/Train/1/b.bin")


class HardwareInterfaceTests(unittest.TestCase):
    def hardware_run(self, identical=False, fail_forward=False):
        model = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.2)
        calls = []
        class Config:
            def __init__(self, model):
                self.cnn_layers = [types.SimpleNamespace(weights=model.weight.detach().tolist(), biases=[])]
                if identical:
                    self.cnn_layers[0].weights = [[0], [0]]
        class DeviceModel:
            def apply_configuration(self, config):
                calls.append(("config", config.cnn_layers[0].weights))
        device = types.SimpleNamespace(get_model=lambda: DeviceModel())
        class Net:
            samna_device = device
            def reset_states(self):
                calls.append(("reset",))
                self.samna_device.get_model().apply_configuration(self.samna_config)
            def __call__(self, events):
                calls.append(("events", events))
                if fail_forward:
                    raise RuntimeError("board failure")
                return [types.SimpleNamespace(feature=0)]
        class Dyn:
            def __init__(self, snn, **kwargs):
                self.model = snn
                self.chip_layers_ordering = [3]
            def make_config(self, **kwargs):
                return Config(self.model)
            def to(self, **kwargs):
                return Net()
        fake_samna = types.ModuleType("samna")
        fake_samna.device = types.SimpleNamespace(close_device=lambda item: calls.append(("close",)))
        fake_samna.speck2f = types.SimpleNamespace(event=types.SimpleNamespace(Spike=object))
        fake_dyn = types.ModuleType("sinabs.backend.dynapcnn")
        fake_dyn.DynapcnnNetwork = Dyn
        fake_helper = types.SimpleNamespace(make_samna_events=lambda events, cls: [
            types.SimpleNamespace(sample=sample, timestamp=index, layer=-1)
            for index, sample in enumerate(events)])
        def replay(net, events, drain):
            self.assertTrue(all(event.layer == 3 for event in events))
            return net([event.sample for event in events])
        job = dict(spec=dict(backend="event", options=dict(backends.DEFAULTS, input_shape=[1, 1, 1], drop_last=False), num_classes=2),
                   shard=0, shards=1, round=1,
                   snn_state=backends.cpu_state(model), snn_guidance={"weight": torch.ones_like(model.weight)})
        dataset, raw = [(torch.ones(1), 0), (torch.ones(1), 1)], [([0], 0), ([1], 1)]
        with patch.dict(sys.modules, {"samna": fake_samna, "sinabs.backend.dynapcnn": fake_dyn}), \
                patch.object(backends, "new_model", return_value=model), \
                patch.object(backends, "convert", side_effect=lambda model, spec: copy.deepcopy(model)), \
                patch.object(backends, "replay_events", side_effect=replay), \
                patch.object(backends, "event_module", return_value=fake_helper), \
                patch.object(backends, "local_data", return_value=(dataset, raw, [0, 1])):
            if identical or fail_forward:
                with self.assertRaises((ValueError, RuntimeError)):
                    backends.executor_result(job, "speck", "", None, "speck2fdevkit:1")
                return calls
            result = backends.executor_result(job, "speck", "", None, "speck2fdevkit:1")
        self.assertEqual(result["mode"], "speck")
        self.assertNotEqual(*result["probe_weight_hashes"])
        return calls

    def test_same_events_base_restored_and_device_closed(self):
        calls = self.hardware_run()
        events = [call[1] for call in calls if call[0] == "events"]
        self.assertEqual(events[:2], events[2:])
        self.assertEqual(len([call for call in calls if call[0] == "reset"]), 4)
        configs = [call[1] for call in calls if call[0] == "config"]
        self.assertEqual(configs[0], configs[1])
        self.assertEqual(configs[1], configs[2])
        self.assertEqual(configs[3], configs[4])
        self.assertNotEqual(configs[0], configs[3])
        self.assertEqual(calls[-1][0], "close")
        self.assertEqual(calls[-2][1], [[0.20000000298023224], [0.20000000298023224]])

    def test_identical_quantized_probes_stop_before_opening_device(self):
        self.assertEqual(self.hardware_run(identical=True), [])

    def test_forward_failure_restores_base_and_closes_device(self):
        calls = self.hardware_run(fail_forward=True)
        self.assertEqual(calls[-1][0], "close")
        self.assertEqual(calls[-2][0], "config")

    def test_replay_waits_for_last_input_and_disables_timestamps(self):
        clock = [0.0]
        events = [types.SimpleNamespace(timestamp=1_000_000)]
        received = []
        fake_io = types.ModuleType("sinabs.backend.dynapcnn.io")
        fake_io.reset_timestamps = lambda device: received.append("reset")
        fake_io.enable_timestamps = lambda device: received.append("enable")
        fake_io.disable_timestamps = lambda device: received.append("disable")
        net = types.SimpleNamespace(device="speck2fdevkit:1",
            samna_output_buffer=types.SimpleNamespace(get_events=lambda: []),
            samna_input_buffer=types.SimpleNamespace(write=lambda samples: received.append("write")))
        def sleep(seconds):
            clock[0] += seconds
        with patch.dict(sys.modules, {"sinabs.backend.dynapcnn.io": fake_io}), \
                patch.object(backends.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(backends.time, "sleep", side_effect=sleep):
            self.assertEqual(backends.replay_events(net, events, 0.2), [])
        self.assertGreaterEqual(clock[0], 1.2)
        self.assertEqual(received, ["reset", "enable", "write", "disable"])


class MultiProcessTests(unittest.TestCase):
    def test_coordinator_and_three_independent_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster = provision(root / "credentials", 2, 1)
            # Reserve an ephemeral port without keeping it bound during subprocess startup.
            import socket
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            url = f"http://127.0.0.1:{port}"
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
            project = str(Path(__file__).resolve().parents[2])
            processes, streams = [], []
            def start(name, arguments):
                stream = (root / (name + ".log")).open("w")
                streams.append(stream)
                process = subprocess.Popen([sys.executable, "-m", *arguments], cwd=project,
                                           env=env, stdout=stream, stderr=subprocess.STDOUT)
                processes.append(process)
                return process
            try:
                coordinator = start("server", ["speck.server", "--roster", str(root / "credentials/roster.json"),
                    "--backend", "toy", "--bind", "127.0.0.1", "--port", str(port),
                    "--output", str(root / "run"), "--rounds", "3", "--round-timeout", "60"])
                transport = Transport(url, roster[0]["token"], timeout=2)
                deadline = time.time() + 30
                while True:
                    if coordinator.poll() is not None:
                        self.fail((root / "server.log").read_text())
                    try:
                        transport.request("/status")
                        break
                    except urllib.error.URLError:
                        if time.time() > deadline:
                            self.fail("Coordinator did not start")
                        time.sleep(0.1)
                workers = [start(member["id"], ["speck.client", "--server", url,
                    "--credentials", str(root / "credentials" / (member["id"] + ".json")),
                    "--state-dir", str(root / member["id"]), "--device", "cpu", "--poll-seconds", "0.1"])
                    for member in roster]
                for worker, member in zip(workers, roster):
                    self.assertEqual(worker.wait(timeout=60), 0, (root / (member["id"] + ".log")).read_text())
                self.assertEqual(transport.request("/status")["status"], "done")
                history = json.loads((root / "run/history.json").read_text())
                self.assertEqual(len(history), 3)
                self.assertTrue(all(item["ann_clients"] == 2 and item["executor_clients"] == 1 for item in history))
                self.assertEqual(history[0]["executor_results"][0]["skipped"], "zero guidance")
                self.assertIn("loss_plus", history[1]["executor_results"][0])
                saved = torch.load(root / "run/state.pt", weights_only=True)
                self.assertEqual(saved["round"], 3)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                for stream in streams:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
