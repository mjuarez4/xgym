# XGym

XGym is a robotics and reinforcement learning toolkit that wraps hardware control, data logging, and dataset preparation for the **XArm7** robot. It relies on ROS for communication with hardware and provides conversion tools to package collected data into `LeRobot` datasets.

## Features

- **Hardware drivers** for the XArm7 and peripheral devices like the SpaceMouse and foot pedals.
- **Gymnasium environments** for tasks such as lifting or stacking objects.
- **Data collection nodes** that record synchronized sensor streams to memmap files.
- **Dataset builders** to convert episodes into TensorFlow datasets.

## Installation

we use [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra ctrl --extra data
```

* `ctrl` extra installs joystick input libraries and the XArm SDK
* `data` extra installs TensorFlow and dataset dependencies.

Ensure your user is in the `plugdev` group for access to USB devices:

```bash
sudo usermod -aG plugdev $USER
```

## Directory structure

- `xgym/gyms/` – Gymnasium environments and wrappers.
- `xgym/nodes/` – ROS nodes for cameras, robot control, and controllers.
- `xgym/rlds/` – Dataset builders for TFDS.
- `scripts/` – Launch files and utilities for data collection and evaluation.

## Usage

1. **Run a camera or controller node** to start streaming data. Example:
   ```bash
   # TODO(codex) fix
   ```
2. **Collect data** with the writer node:
   ```bash
   # TODO(codex) fix
   ```
3. **Convert memmaps** to LeRobot datasets using the scripts in `lrbt`:
   ```bash
   python xgym/lrbt/from_memmap.py --help # TODO(codex) fix
   ```

## Scripts

TODO(codex) add scripts documentation with purpose and usage

## Contributing

Pull requests are welcome. Please format code with `black` before submitting.

TODO(codex) fix git precommit hook

## License

This project is licensed under the MIT License.
