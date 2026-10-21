|   lag_hours |   pearson_correlation | interpretation                                  |
|------------:|----------------------:|:------------------------------------------------|
|          -5 |                0.8385 | Pred(t) vs Actual(t - 5h) [Lag-5 memory]        |
|          -4 |                0.8768 | Pred(t) vs Actual(t - 4h) [Lag-4 memory]        |
|          -3 |                0.9154 | Pred(t) vs Actual(t - 3h) [Lag-3 memory]        |
|          -2 |                0.9577 | Pred(t) vs Actual(t - 2h) [Lag-2 memory]        |
|          -1 |                0.9932 | Pred(t) vs Actual(t - 1h) [Lag-1 memory]        |
|           0 |                0.9795 | Pred(t) vs Actual(t) [Target synchronous]       |
|           1 |                0.9432 | Pred(t) vs Actual(t + 1h) [Lead-1 anticipation] |
|           2 |                0.9049 | Pred(t) vs Actual(t + 2h) [Lead-2 anticipation] |
|           3 |                0.8666 | Pred(t) vs Actual(t + 3h) [Lead-3 anticipation] |
|           4 |                0.829  | Pred(t) vs Actual(t + 4h) [Lead-4 anticipation] |
|           5 |                0.7935 | Pred(t) vs Actual(t + 5h) [Lead-5 anticipation] |
