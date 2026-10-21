|   lag_hours |   pearson_correlation | interpretation                                  |
|------------:|----------------------:|:------------------------------------------------|
|          -5 |                0.8463 | Pred(t) vs Actual(t - 5h) [Lag-5 memory]        |
|          -4 |                0.8829 | Pred(t) vs Actual(t - 4h) [Lag-4 memory]        |
|          -3 |                0.9218 | Pred(t) vs Actual(t - 3h) [Lag-3 memory]        |
|          -2 |                0.961  | Pred(t) vs Actual(t - 2h) [Lag-2 memory]        |
|          -1 |                0.9935 | Pred(t) vs Actual(t - 1h) [Lag-1 memory]        |
|           0 |                0.9804 | Pred(t) vs Actual(t) [Target synchronous]       |
|           1 |                0.9456 | Pred(t) vs Actual(t + 1h) [Lead-1 anticipation] |
|           2 |                0.9083 | Pred(t) vs Actual(t + 2h) [Lead-2 anticipation] |
|           3 |                0.872  | Pred(t) vs Actual(t + 3h) [Lead-3 anticipation] |
|           4 |                0.8381 | Pred(t) vs Actual(t + 4h) [Lead-4 anticipation] |
|           5 |                0.806  | Pred(t) vs Actual(t + 5h) [Lead-5 anticipation] |
