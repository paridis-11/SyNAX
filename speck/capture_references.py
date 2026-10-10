"""Capture selected math functions without importing either training runner."""

import argparse
import ast
import hashlib
import json
from pathlib import Path


def extract(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    selected = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected[node.name] = ast.get_source_segment(source, node)
    if set(selected) != set(names):
        raise ValueError(f"Missing reference functions in {path}: {set(names) - set(selected)}")
    return selected, hashlib.sha256(source.encode()).hexdigest()


def capture(root):
    output = Path(__file__).parent / "references"
    output.mkdir(exist_ok=True)
    sources = {
        "static_update": ("static/SyNAX_normalized_snn.py", ["update_global_models", "train_synax_ANN"]),
        "static_schedule": ("static/utils/utils.py", ["LR_scheduler"]),
        "static_optimizer": ("static/Node/Node.py", ["init_optimizer"]),
        "event_update": ("event/synax_neuro.py", ["apply_update", "apply_normalized_executor_correction",
                                                 "compute_snn_guidance", "scheduled_lr"]),
    }
    manifest = {}
    for name, (relative, functions) in sources.items():
        selected, source_hash = extract(root / relative, functions)
        text = "# Extracted reference functions; do not run as a training script.\n"
        text += "from __future__ import annotations\nimport copy\nimport math\nimport torch\nfrom torch import nn\n\n\n"
        text += "\n\n\n".join(selected.values()) + "\n"
        text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
        path = output / (name + ".py")
        path.write_text(text)
        manifest[name] = dict(source=relative, source_sha256=source_hash,
                              excerpt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                              functions=functions)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True)
    capture(Path(parser.parse_args().release_root))
