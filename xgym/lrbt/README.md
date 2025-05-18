# Lerobot Dataset


## generating a Lerobot dataset

```mermaid
graph LR
    A[Get Videos] --> B

    subgraph B["Process Data"]
        B0[Process Frame]
        B0 --> C
        C --> B0
        B0 --> B1

        B1[Remap HaMer Keys]
        B1 --> B2[Add Camera Transform Local-to-Global]
        B2 --> B3[**Perspective Projection:**
        Compute 2D Keypoints kp2d from kp3d and
        Scaled Focal Length
        ]
        B3 --> B4[Filter Duplicate Detections]
    end
    C[Hamer Server]


    B --> D[Lerobot Preprocessing]

    subgraph D["Lerobot Preprocessing"]
        D1[Add Task and Language Embedding]
        D1 --> D2[Add Frames]
        D2 --> D3[Push to Hub]
    end
```

## using Lerobot to train a model

```mermaid
graph TD
    A[1\. Start: HF Parquet Data] --> B[Transformation Steps]

    subgraph B[2\. **Transformation Steps**]
        B1[2\.1 **Remap** State Keys to Embodiment]
        B1 --> B2[2\.2 **Remap** Embodiment Keys to Shared if Possible]
        B2 --> B3[2\.3 **Create Shared Features**
        that Don’t Exist]
        B3 --> B4[2\.4 **Design Action**
        from State Key Target Chunks, Rewrite State as Current Timestep]
    end

    B --> C[3\. Model Normalization with Stats]
    C --> D[Train]
```

## Scripts

The files in this directory convert collected episodes into LeRobot datasets.
All scripts expose a command line interface via [tyro](https://github.com/brentyi/tyro). Run them with `--help` for full options.

### from_memmap.py

Convert raw memmap recordings to a dataset.

```bash
python -m xgym.lrbt.from_memmap \
    --dir /path/to/memmaps \
    --branch v0.1 \
    --repo-id username/dataset
```

Arguments:

- `--dir` – directory containing `.dat` files and a `task-*.npy` description.
- `--branch` – dataset branch pushed to the Hugging Face hub.
- `--repo-id` – destination repository in the format `user/name`.

### from_mano_npz.py

Processes MANO `.npz` outputs produced by the HaMer server. Uses the same
arguments as `from_memmap.py` plus connection information for HaMer:

```bash
python -m xgym.lrbt.from_mano_npz \
    --dir /path/to/npz \
    --branch v0.1 \
    --repo-id username/dataset \
    --hamer.host localhost \
    --hamer.port 8765
```

### convert.py

Example conversion script for RLDS-style datasets. Key options include
`--embodiment`, `--task` and `--version`:

```bash
python -m xgym.lrbt.convert \
    --embodiment single \
    --task lift \
    --version v0.1
```

### util.py

Helper utilities used by the other scripts. In particular,
`get_taskinfo(path)` reads a `task-*.npy` file from `path` and returns the task
and language strings.
