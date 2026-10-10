import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from speck import algorithm, backends
from speck.provision import provision
from speck.server import Federation
from speck.wire import digest, pack


def reference(name):
    path = Path(__file__).resolve().parents[1] / "references" / (name + ".py")
    spec = importlib.util.spec_from_file_location("reference_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReferenceAlignmentTests(unittest.TestCase):
    def test_reference_excerpts_match_manifest(self):
        root = Path(__file__).resolve().parents[1] / "references"
        manifest = json.loads((root / "manifest.json").read_text())
        for name, value in manifest.items():
            self.assertEqual(hashlib.sha256((root / (name + ".py")).read_bytes()).hexdigest(),
                             value["excerpt_sha256"])

    def test_global_update_and_guidance_match_static_function(self):
        source = reference("static_update")
        source.ann2snn = lambda model, *args, **kwargs: copy.deepcopy(model)
        torch.manual_seed(9)
        base = backends.new_model(dict(backend="toy"))
        base.register_buffer("running_mean", torch.zeros(2))
        base.register_buffer("counter", torch.tensor(0))
        for alpha in (-0.5, 0.0, 0.5):
            model = copy.deepcopy(base)
            old = backends.cpu_state(model)
            guidance = {name: torch.randn_like(value.float()) for name, value in old.items()}
            guidance["running_mean"].fill_(1e6)
            delta = {name: torch.full_like(value, 0.03) if value.is_floating_point()
                     else torch.zeros_like(value) for name, value in old.items()}
            node = types.SimpleNamespace(model=model, snn_model=copy.deepcopy(model),
                g_t=copy.deepcopy(guidance), G_t={name: torch.zeros_like(value) for name, value in old.items()},
                args=None, test_data=None)
            expected = algorithm.global_state_update(old, dict(model.named_parameters()),
                                                      delta, guidance, alpha, 1.0, 0.1)
            source.update_global_models(node, delta, alpha, eta_ann=1.0, eta_snn=0.1)
            actual = backends.cpu_state(node.model)
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value, rtol=1e-6, atol=1e-7)
            for name, value in algorithm.snn_guidance(old, backends.cpu_state(node.snn_model)).items():
                torch.testing.assert_close(value, node.G_t[name])

    def test_global_correction_also_matches_event_normalized_correction(self):
        source = reference("event_update")
        model = backends.new_model(dict(backend="toy"))
        state = backends.cpu_state(model)
        guidance = {name: torch.full_like(value, 0.4) for name, value in state.items()}
        expected = algorithm.global_state_update(state, dict(model.named_parameters()),
                                                 {}, guidance, -0.3, 1.0, 0.1)
        source.apply_normalized_executor_correction(model, guidance, -0.3, 0.1)
        for name, value in expected.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=1e-6, atol=1e-7)

    def test_event_guidance_matches_static_direction(self):
        source = reference("event_update")
        old = backends.new_model(dict(backend="toy"))
        new = copy.deepcopy(old)
        with torch.no_grad():
            for parameter in new.parameters():
                parameter.add_(0.2)
        static = algorithm.unit_direction(algorithm.snn_guidance(backends.cpu_state(old), backends.cpu_state(new)))
        event = source.compute_snn_guidance(new, old)
        for name, value in static.items():
            torch.testing.assert_close(value, event[name], rtol=1e-6, atol=1e-7)

    def test_lr_schedule_matches_static_and_event_without_warmup(self):
        static, event = reference("static_schedule"), reference("event_update")
        for rounds in (3, 80, 100, 500):
            for index in range(rounds):
                actual = algorithm.scheduled_lr(0.001, index, rounds)
                self.assertEqual(actual, static.LR_scheduler(0.001, index, rounds))
                self.assertAlmostEqual(actual, event.scheduled_lr(0.001, index, rounds, 0), places=15)

    def test_ann_local_updates_match_actual_static_training_function(self):
        source = reference("static_update")
        class Meter:
            def update(self, *args, **kwargs):
                pass
        class Progress:
            def __init__(self, loader):
                self.loader = loader
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __iter__(self):
                return iter(self.loader)
            def set_description(self, description):
                pass
        source.AverageMeter, source.tqdm = Meter, Progress
        torch.manual_seed(3)
        model = backends.new_model(dict(backend="toy"))
        class TupleSequential(torch.nn.Sequential):
            def forward(self, value):
                return (super().forward(value),)
        reference_model = TupleSequential(*copy.deepcopy(list(model.children())))
        x, y = torch.randn(8, 4), torch.tensor([0, 1] * 4)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        options = dict(backends.DEFAULTS, E=2, lr=0.03, optimizer="sgd", momentum=0.0,
                       weight_decay=0.0, zero_momentum=0.4, batch_size=2)
        guidance = {name: torch.full_like(value, 0.002) for name, value in model.state_dict().items()}
        node = types.SimpleNamespace(model=reference_model, device=torch.device("cpu"),
            train_data=loader, optimizer=torch.optim.SGD(reference_model.parameters(), lr=0.03),
            num=1, args=types.SimpleNamespace(dataset="toy"))
        gamma = 1 / (options["E"] * len(loader) * 0.03)
        for epoch in range(options["E"]):
            source.train_synax_ANN(node, epoch, 0, guidance, beta=0.4, gamma=gamma)
        job = dict(spec=dict(backend="toy", options=options, num_classes=2),
                   ann_state=backends.cpu_state(model), ann_guidance=guidance,
                   shard=0, shards=1, round=0, round_lr=0.03)
        with patch.object(backends, "local_data", return_value=(dataset, None, list(range(8)))), \
                patch.object(backends, "DataLoader", return_value=loader), \
                patch.object(backends, "new_model", side_effect=lambda *args: copy.deepcopy(model)):
            result = backends.ann_result(job, "cpu", "", None)
        for name, value in result["delta"].items():
            expected = reference_model.state_dict()[name] - model.state_dict()[name]
            torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-7)

    def test_optimizer_settings_and_steps_match_static_factory(self):
        source = reference("static_optimizer")
        for name in ("sgd", "adam", "adamw"):
            options = dict(backends.DEFAULTS, optimizer=name, lr=0.03, momentum=0.7,
                           opt_eps=1e-5, weight_decay=0.001)
            model = backends.new_model(dict(backend="toy"))
            expected_model = copy.deepcopy(model)
            actual = backends.make_optimizer(model, options, options["lr"])
            expected = source.init_optimizer(expected_model, types.SimpleNamespace(**options))
            actual_group = {key: value for key, value in actual.param_groups[0].items() if key != "params"}
            expected_group = {key: value for key, value in expected.param_groups[0].items() if key != "params"}
            self.assertEqual(actual_group, expected_group)
            for _ in range(3):
                for parameter, reference_parameter in zip(model.parameters(), expected_model.parameters()):
                    parameter.grad = torch.full_like(parameter, 0.07)
                    reference_parameter.grad = torch.full_like(reference_parameter, 0.07)
                actual.step()
                expected.step()
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, expected_model.state_dict()[key])

    def test_sampling_keeps_at_least_one_ann_and_supports_partial_participation(self):
        roster = [dict(id="ann-1", role="ann")] + [dict(id=f"snn-{i}", role="executor") for i in range(9)]
        rng = random.Random(5)
        for _ in range(100):
            selected = algorithm.sample_clients(roster, 0.2, rng)
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(set(selected)), 2)
            self.assertIn("ann-1", selected)

    def test_mean_then_clip_and_sampler_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster = provision(root / "credentials", 1, 2)
            engine = Federation(roster, "toy", None, {}, root / "run", 3)
            for member, alpha in zip(roster, (None, 10.0, -0.4)):
                job = engine.job(member)
                result = (dict(delta={name: torch.zeros_like(value) for name, value in job["ann_guidance"].items()}, samples=1)
                          if member["role"] == "ann" else dict(alpha=alpha, samples=1, mode="toy"))
                payload = dict(client_id=member["id"], version=job["version"], round=job["round"], result=result)
                engine.accept(member, payload, digest(pack(payload)))
            self.assertAlmostEqual(engine.history[0]["alpha_mean"], 4.8)
            self.assertEqual(engine.history[0]["alpha_applied"], 0.5)
            resumed = Federation(roster, "toy", None, {}, root / "run", 3, resume=True)
            self.assertEqual(resumed.sampled, engine.sampled)
            self.assertEqual(resumed.sampling_rng.getstate(), engine.sampling_rng.getstate())

    def test_unselected_clients_wait_and_round_lr_is_broadcast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster = provision(root / "credentials", 1, 3)
            engine = Federation(roster, "toy", None, {"sample_ratio": 0.5}, root / "run", 3)
            self.assertEqual(len(engine.sampled), 2)
            for member in roster:
                job = engine.job(member)
                self.assertEqual(job["status"], "job" if member["id"] in engine.sampled else "wait")
                if job["status"] == "job":
                    self.assertEqual(job["round_lr"], 0.001)


if __name__ == "__main__":
    unittest.main()
