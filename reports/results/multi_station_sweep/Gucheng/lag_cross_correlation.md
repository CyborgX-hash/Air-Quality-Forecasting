|   lag_hours |   pearson_correlation | interpretation                                  |
|------------:|----------------------:|:------------------------------------------------|
|          -5 |                0.8357 | Pred(t) vs Actual(t - 5h) [Lag-5 memory]        |
|          -4 |                0.8724 | Pred(t) vs Actual(t - 4h) [Lag-4 memory]        |
|          -3 |                0.9103 | Pred(t) vs Actual(t - 3h) [Lag-3 memory]        |
|          -2 |                0.9528 | Pred(t) vs Actual(t - 2h) [Lag-2 memory]        |
|          -1 |                0.99   | Pred(t) vs Actual(t - 1h) [Lag-1 memory]        |
|           0 |                0.9775 | Pred(t) vs Actual(t) [Target synchronous]       |
|           1 |                0.9388 | Pred(t) vs Actual(t + 1h) [Lead-1 anticipation] |
|           2 |                0.9007 | Pred(t) vs Actual(t + 2h) [Lead-2 anticipation] |
|           3 |                0.863  | Pred(t) vs Actual(t + 3h) [Lead-3 anticipation] |
|           4 |                0.8226 | Pred(t) vs Actual(t + 4h) [Lead-4 anticipation] |
|           5 |                0.7821 | Pred(t) vs Actual(t + 5h) [Lead-5 anticipation] |
