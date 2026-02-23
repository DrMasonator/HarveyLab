# Optical Refinement and Imaging On-axis Navigator (ORION)

ORION is a laser alignment and beam analysis tool. It combines hardware control (motorized Z stage + camera) with live imaging, beam characterization, and caustic measurements.

## Requirements

- Python 3.10+
- Dependencies:
  - `numpy`
  - `PyQt5`
  - `pyqtgraph`
  - `matplotlib`
  - Hardware drivers (optional): `pylablib`, `ids-peak`

Install core dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python /Users/mason/PycharmProjects/HarveyLab/ORION/main.py
```

Simulation mode (no hardware):

```bash
python /Users/mason/PycharmProjects/HarveyLab/ORION/main.py --sim
```

## Settings

The application loads configuration from:

- `~/.orion_config.json`

You can edit settings in the UI via **ORION → Settings** (or the **Settings** button in the Controls panel). Changes persist across runs.

## Tests

```bash
python -m unittest discover -s /Users/mason/PycharmProjects/HarveyLab/ORION/tests
```

## Notes

- Hardware initialization errors are reported in logs and will abort startup.
- If you’re running on a low‑spec machine, consider using the `RED` or `GREEN` Bayer modes for faster processing.
