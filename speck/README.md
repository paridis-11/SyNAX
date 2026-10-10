# SyNAX on Speck

Run one coordinator, ANN workers on Orin or CUDA servers, and executor workers
on hosts connected to Speck. Devices communicate over Tailscale or a LAN.

## Environment

Copy the repository to each host and run commands from the repository root.
Use Python 3.10 or later with PyTorch, NumPy, Tonic, tqdm and Sinabs.
Speck hosts also need Samna and USB access to the board.
On Orin, install PyTorch following the
[NVIDIA Jetson guide](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html).

## Network and Credentials

Sign in to the same Tailscale network on each host. Get the coordinator's IP:

```bash
tailscale ip -4
```

Use this address in `--bind` and `--server`, with TCP port 8765 available.
For a LAN, use the coordinator's private LAN IP instead.

On the coordinator, generate credentials for two ANN clients and four executors:

```bash
python -m speck.provision --directory speck/credentials --ann-clients 2 --executors 4
```

Keep `roster.json` on the coordinator. Copy each worker's JSON file to
`speck/credentials/` on its host: `ann-1.json`, `ann-2.json`, or
`executor-1.json` through `executor-4.json`. These generated files stay local.

## Coordinator

```bash
python -m speck.server \
  --roster speck/credentials/roster.json \
  --bind <coordinator-ip> --port 8765 \
  --backend event --checkpoint /path/to/best_sinabs_snn.pt \
  --output speck/runs/multihost --rounds 80
```

The checkpoint supplies the model and dataset settings.
Add `--options speck/options.example.json` to set training parameters.

## ANN Worker

On each Orin or CUDA server:

```bash
python -m speck.client \
  --server http://<coordinator-ip>:8765 \
  --credentials speck/credentials/ann-1.json \
  --state-dir speck/runs/ann-1 \
  --device cuda:0 --data-root /path/to/training-data
```

For the second ANN worker, replace `ann-1` with `ann-2` in both paths.

## Speck Worker

On each host connected to a Speck board:

```bash
python -m speck.client \
  --server http://<coordinator-ip>:8765 \
  --credentials speck/credentials/executor-1.json \
  --state-dir speck/runs/executor-1 \
  --data-root /path/to/training-data --speck-device speck2fdevkit:1
```

Replace `executor-1` with the worker's ID and select its board using
`--speck-device`. Several boards can share one host with separate worker
processes, credentials and state directories.

## Data and Outputs

For N-MNIST, `--data-root` is the Tonic `save_to` directory.
For DVS Gesture, it is the directory containing `train/<class>/*.npz`
with raw `x`, `y`, `t` and `p` events. See [DATA.md](../DATA.md).

The default IID split uses the same training dataset on every worker.
To supply a local partition, add `--indices /path/to/local_indices.json`,
containing a JSON list of training-sample indices.

Results are saved under `--output`: `history.json` contains round statistics,
`state.pt` stores the coordinator state, and `latest.pt` stores the latest model.
To resume, repeat the coordinator command with `--resume` and restart workers
with their existing state directories.

## Tailscale Userspace Networking

For a coordinator in a container without a TUN interface, use
`--bind 127.0.0.1` and publish the service through Tailscale Serve:

```bash
tailscale serve --bg http://127.0.0.1:8765
```

Use the displayed private HTTPS URL for the workers' `--server`.
If a worker's Tailscale daemon has a local HTTP proxy, add
`--proxy http://127.0.0.1:1055`.

## Tests

```bash
python -m unittest discover -s speck/tests -v
```
