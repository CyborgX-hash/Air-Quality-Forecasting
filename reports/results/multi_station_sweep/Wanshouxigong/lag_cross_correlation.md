|   lag_hours |   pearson_correlation | interpretation                                  |
|------------:|----------------------:|:------------------------------------------------|
|          -5 |                0.8408 | Pred(t) vs Actual(t - 5h) [Lag-5 memory]        |
|          -4 |                0.8755 | Pred(t) vs Actual(t - 4h) [Lag-4 memory]        |
|          -3 |                0.9121 | Pred(t) vs Actual(t - 3h) [Lag-3 memory]        |
|          -2 |                0.954  | Pred(t) vs Actual(t - 2h) [Lag-2 memory]        |
|          -1 |                0.9921 | Pred(t) vs Actual(t - 1h) [Lag-1 memory]        |
|           0 |                0.9706 | Pred(t) vs Actual(t) [Target synchronous]       |
|           1 |                0.9342 | Pred(t) vs Actual(t + 1h) [Lead-1 anticipation] |
|           2 |                0.8974 | Pred(t) vs Actual(t + 2h) [Lead-2 anticipation] |
|           3 |                0.8614 | Pred(t) vs Actual(t + 3h) [Lead-3 anticipation] |
|           4 |                0.827  | Pred(t) vs Actual(t + 4h) [Lead-4 anticipation] |
|           5 |                0.7919 | Pred(t) vs Actual(t + 5h) [Lead-5 anticipation] |
