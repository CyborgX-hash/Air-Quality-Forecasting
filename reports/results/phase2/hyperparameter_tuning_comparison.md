# Phase 2 LSTM Hyperparameter Tuning & Ablation Study

**Station:** Aotizhongxin | **Horizons:** [1, 3, 6, 12, 24] | **Total Runtime:** 37.14s

**Selection Criterion:** Lowest Macro Validation MAE (Strict Validation-Only Selection)

|   rank | experiment_id             | phase           |   window_size |   lstm_units |   lstm_layers |   dropout_rate |   learning_rate |   batch_size | loss_function   |   best_epoch |   val_loss |   macro_val_mae |   macro_val_rmse |   macro_val_r2 |   duration_s |   mae_1h |   mae_3h |   mae_6h |   mae_12h |   mae_24h |
|-------:|:--------------------------|:----------------|--------------:|-------------:|--------------:|---------------:|----------------:|-------------:|:----------------|-------------:|-----------:|----------------:|-----------------:|---------------:|-------------:|---------:|---------:|---------:|----------:|----------:|
|      1 | EXP03_Window_W72          | Lookback Window |            72 |           64 |             1 |            0.2 |           0.001 |           32 | mae             |            2 |     0.3944 |           31.08 |            52.4  |         0.5779 |        17.95 |    13.91 |    21.76 |    30.43 |     40.15 |     49.13 |
|      2 | EXP02_Window_W48          | Lookback Window |            48 |           64 |             1 |            0.2 |           0.001 |           32 | mae             |            2 |     0.3989 |           31.43 |            53.21 |         0.5658 |        12.04 |    14    |    21.98 |    30.82 |     40.7  |     49.65 |
|      3 | EXP01_Baseline_W24_U64_L1 | Baseline        |            24 |           64 |             1 |            0.2 |           0.001 |           32 | mae             |            1 |     0.4007 |           31.57 |            52.48 |         0.5796 |         7.15 |    13.94 |    22.29 |    30.28 |     40.98 |     50.36 |
