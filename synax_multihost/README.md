# Multi-host SyNAX for Speck

One coordinator aggregates updates from ANN workers and scalar feedback from
Speck workers. Each worker runs on its own Linux machine. Tailscale supplies
private IP connectivity; the application uses authenticated HTTP over that network.
LAN addresses work with the same code.

## Topology

For two ANN clients and four Speck executors:

- One coordinator process, which can share an Orin or GPU server with an ANN worker.
- Two ANN worker processes, on Orin or ordinary CUDA servers.
- Four executor processes, each on a host with a locally connected Speck device.

Several Speck boards can share a host if each worker has a different
`--speck-device`, credential file and state directory. Device indices must be
checked on that host. Speck itself does not run Tailscale or Python.

## Files and Environment

Run commands from the project root, containing both `synax_multihost/` and the
existing `synax_speck_event.py`. Copy the same sources to every host.
The event backend reuses that module's models, data utilities, conversion and
event formatting. ANN training follows the formal static runner's algorithm.
It supports N-MNIST and DVS Gesture, not the software ResNet/ViT deployment path.

The toy smoke test requires Python >=3.10 and PyTorch >=2.0. The event backend
also requires NumPy, Tonic, tqdm and Sinabs. Speck workers need the compatible
Samna/Sinabs hardware backend and USB access. Use the versions already validated
with the existing hardware scripts before upgrading them.

On Orin, install the NVIDIA PyTorch build compatible with the installed JetPack,
Python and CUDA versions, rather than a desktop x86 wheel. ANN workers should
verify `torch.cuda.is_available()` before running.
See [NVIDIA's Jetson installation guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

## Network and Credentials

Install Tailscale and sign in on each host. On the coordinator:

```bash
tailscale ip -4
```

On each worker, check the coordinator's address:

```bash
tailscale ping <coordinator-tailscale-ip>
```

Allow worker-to-coordinator TCP port 8765 in the tailnet policy and host firewall.
Bind the coordinator to its specific Tailscale IP, not `0.0.0.0`. Do not expose
the HTTP service to the public internet or enable Funnel. Tailscale encrypts
traffic between tailnet devices; HTTPS may be used in other managed deployments.
See [Tailscale connectivity](https://tailscale.com/kb/1452/connect-to-devices)
and [network security](https://tailscale.com/kb/1429/secure).

Generate credentials on the coordinator:

```bash
python -m synax_multihost.provision --directory synax_multihost/credentials --ann-clients 2 --executors 4
```

Keep `synax_multihost/credentials/roster.json` on the coordinator. Transfer only `ann-1.json` to
the first ANN worker, `ann-2.json` to the second, and one executor JSON file to
each Speck worker, using SSH/SCP with normal host-key verification. Do not commit
these files or model checkpoints to GitHub.

### Containers Without a TUN Interface

If the coordinator uses Tailscale userspace networking, its Tailscale IP may not
be bindable by ordinary applications. Run the coordinator with `--bind 127.0.0.1`
and publish that local service privately with Tailscale Serve:

```bash
tailscale serve --bg http://127.0.0.1:8765
tailscale serve status
```

Use the displayed private HTTPS URL as the worker's `--server`, and permit its
HTTPS port in the tailnet policy. Do not use Funnel. If a worker also uses
userspace networking, configure its existing Tailscale daemon with a loopback
HTTP proxy (`--outbound-http-proxy-listen=127.0.0.1:1055`) and add
`--proxy http://127.0.0.1:1055` to the worker command. Ambient `HTTP_PROXY` settings
are deliberately ignored; only this explicit proxy is used. HTTPS certificate
verification remains enabled.
See [Serve](https://tailscale.com/docs/reference/tailscale-cli/serve) and
[userspace networking](https://tailscale.com/docs/concepts/userspace-networking).

## Coordinator

```bash
python -m synax_multihost.server \
  --roster synax_multihost/credentials/roster.json \
  --bind <coordinator-tailscale-ip> --port 8765 \
  --backend event --checkpoint /path/to/best_sinabs_snn.pt \
  --output synax_multihost/runs/multihost --rounds 80
```

Architecture, dataset and conversion settings come from the checkpoint. Existing
training hyperparameters are inherited where present; missing values use the
defaults in `backends.py`. Use `--options synax_multihost/options.example.json`
to explicitly override training parameters. The run's resolved options
are stored in `synax_multihost/runs/multihost/state.pt` under `spec.options`.

The coordinator is CPU-based; ANN training happens on the worker GPUs.
The default executor mode for the event backend is real `speck` hardware.
For software-only integration testing, use `--executor-mode sinabs` explicitly.

## ANN Worker

```bash
CUDA_VISIBLE_DEVICES=0 python -m synax_multihost.client \
  --server http://<coordinator-tailscale-ip>:8765 \
  --credentials synax_multihost/credentials/ann-1.json --state-dir synax_multihost/runs/ann-1 \
  --device cuda:0 --data-root /path/to/training-data
```

Run the same command on the second host using its `ann-2.json` and a different
state directory. Selecting physical GPU 0 above maps it to `cuda:0` within that
process. Orin uses the same worker entry point.

## Speck Worker

```bash
python -m synax_multihost.client \
  --server http://<coordinator-tailscale-ip>:8765 \
  --credentials synax_multihost/credentials/executor-1.json --state-dir synax_multihost/runs/executor-1 \
  --data-root /path/to/training-data --speck-device speck2fdevkit:1
```

Repeat using the other executor credentials on their hosts. N-MNIST uses the
same `save_to` root as the existing Tonic loader; DVS Gesture uses the directory
containing `train/<class>/*.npz`, with raw `x`, `y`, `t`, `p` events. Do not point
adaptation workers at the test split.

By default, workers deterministically divide the **same ordered training dataset**
into non-overlapping IID shards, using roster order and the shared seed. For
existing non-IID or subject-specific partitions, pass `--indices local_indices.json`,
a list of training-set indices. With different private local datasets, pass a
list covering the desired local samples instead of using the shared-dataset split.
Dataset contents and ordering must agree when using automatic partitioning.
N-MNIST files are explicitly ordered by their relative path on every host;
index files refer to this canonical order, not an unsorted filesystem traversal.
Raw samples never leave the worker; only model updates or feedback are uploaded.

## Learning and Recovery

The authoritative algorithm is `static/SyNAX_normalized_snn.py`. Local gradient
mixing, aggregation, guidance signs, learning-rate scheduling and client sampling
are tested against the reference functions.

```text
delta = equal mean of ANN client parameter updates
alpha = clipped equal mean of executor directional derivatives
w_next = w + eta_ann * delta - eta_snn * alpha * normalize(previous_ANN_guidance)
previous_ANN_guidance_next = w - w_next
executor_guidance_next = previous_SNN_parameters - next_SNN_parameters
probe_direction = normalize(executor_guidance)
```

The first round has zero historical guidance, so its executor feedback is
explicitly skipped. ANN optimizer state is reset each round, matching the existing
runner. Learning rate follows its cosine schedule, and gamma is calculated using
the current round's learning rate and K = E * number of local minibatches.
`sample_ratio` controls uniform sampling with a replacement when necessary to
ensure at least one ANN client. The default 1.0 retains full participation for
the physical hardware setup; 0.2 selects 20% of clients as in the static protocol.
`zero_momentum` is the ANN gradient-mixing coefficient (beta), `lr` is the local
learning rate, and `zo_epsilon` is the two-sided probe radius. No heuristic sign flip,
adaptive gain, relative-step cap or feedback decay from later ablation experiments
is enabled implicitly. Feedback is averaged first and then clipped, matching
the static code. The saved algorithm version is `synax-static-v2`;
checkpoints from earlier multi-host drafts cannot be resumed.
Training and feedback discard incomplete minibatches by default. For a tiny
smoke-test shard, explicitly select `drop_last=false` or reduce batch size.

Speck workers perturb the SNN in both directions, quantize using the existing
Sinabs backend, and replay the same cached raw samples for both probes. Replay
uses the mapped input core and waits for the last input timestamp plus
`hw_drain_seconds` (default 0.2 s) before collecting the final output. They
monitor the output layer and calculate cross-entropy from output class spike
counts on the **host CPU**, then return the directional derivative. Identical
quantized probe parameters cause an explicit error; adjust `zo_epsilon` rather than
silently treating such a probe as useful feedback. The base configuration is
restored after probing, and device resources are closed.

The derivative is a double-precision Python scalar (8-byte numeric payload).
HTTP headers, serialization, round IDs and diagnostics add overhead: the complete
network message is **not** 8 bytes. This version also sends full model states and
guidance downstream each round. It does not measure system energy or implement
on-chip backpropagation/STDP.

Each job has a unique version and round. Results are validated, cached before
upload and accepted idempotently. Stale results are rejected. A worker reconnects
by fetching the latest full state; missing rounds do not require replaying old
updates. The coordinator saves accepted results and completed rounds atomically.

Default round timeout is 7200 seconds. Raw-event replay and per-sample chip resets
can make a hardware probe much slower than its software equivalent. For initial
device smoke tests, explicitly reduce `feedback_batches` and measure round time
before selecting a formal timeout. A timeout without all selected clients stops the
run. To allow partial rounds, explicitly set `--min-ann` and `--min-executors`;
such rounds are recorded with missing clients and are not the full-participation
paper protocol. Completed workers wait for the next round. Workers exit when all
rounds finish; the coordinator stays available until stopped with Ctrl-C.

Restart the coordinator with the same arguments plus `--resume`. The total number
of rounds cannot change on resume because it defines the cosine schedule. It reloads the
saved model, guidance and accepted results, and renews the round timeout. A failed
round gets a new job version. Restart a worker with the same state directory to
reuse any cached result. Do not run two processes with the same worker credentials.
Sampled clients and the coordinator's sampling RNG state are checkpointed. Batch
ordering and GPU numerics can differ across physical hosts; matching the algorithm
does not imply bitwise-identical trajectories to a single-process experiment.

`history.json` records feedback, backend, timing and missing clients. The event
backend exports `latest.pt` for the existing Speck evaluation scripts. This code
has no automatic held-out accuracy evaluator; evaluate saved models separately
on data excluded from all worker training shards.

## Tests

```bash
python -m unittest discover -s synax_multihost/tests -v
```

The suite includes a coordinator and independent CPU worker processes over
loopback HTTP, protocol validation/recovery, and mocked Speck event/configuration
tests. Real Tailscale links, Orin CUDA and connected Speck boards require a device
smoke test before using the implementation for reported hardware experiments.
