# Architecture

`pipeline/inference.py` dispatches to one of the perception or detection
modules. Shared defaults live in `config.py`; baseline collection lives in
`calibration/baseline.py`; alert implementations live in `alerting/`.
