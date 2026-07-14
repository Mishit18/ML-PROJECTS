# Results Log

Use this directory for final experiment tables. Keep generated images and checkpoints in `runs/`; keep only compact CSV/JSON summaries here.

Recommended final table:

```text
run,params_m,schedule,weights,sampler,steps,num_samples,fid,is_mean,is_std,samples_per_second,train_steps
```

After each evaluation, copy the row from `runs/<run_name>/metrics/metrics.csv` into `results/final_metrics.csv`.
