$ErrorActionPreference = "Stop"
python validate_config.py configs\*.yaml
python -m pytest tests
python -m compileall train.py sample.py evaluate.py monitor.py benchmark_sampler.py validate_config.py overfit_one_batch.py make_report.py src tests
