# Config file index and resolution order.

Resolution order:

1. `default.yaml` defines the base configuration for all subsystems.
2. Pipeline modules load their specific file (e.g. `configs/detection/yolov8n.yaml`)
   and deep-merge it into the defaults.
3. Environment variables prefixed with `SAFEWATCH_` override any key
   (e.g. `SAFEWATCH_LOG_LEVEL=DEBUG`, `SAFEWATCH_API_PORT=8080`).

Naming convention: one YAML file per pipeline stage / subsystem under the
matching directory. Configs shipped here are validation-only skeletons:
every key used by this optimization pipeline must have a default.

Subsystem home:

- `detection/` - YOLOv8 pose model profiles (n/s/m)
- `tracking/`  - ByteTrack tracker profiles
- `features/`  - feature extraction enablement and thresholds
- `models/`    - behavior classifier profiles
- `training/`  - training run definitions (phase 1 baseline, phase 2 temporal)
- `inference/` - online vs offline inference profiles
- `alerting/`  - alert rule catalog and dispatch channels
- `api/`       - server, auth, websocket, limits
- `experiment/`- experiment tracking / MLflow

Adding a new config: create the file, reference it from `default.yaml`
(or the stage loader), and add a unit test in `tests/unit/test_config.py`.