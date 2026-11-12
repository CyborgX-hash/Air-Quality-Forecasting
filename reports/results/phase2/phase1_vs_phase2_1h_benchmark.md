| station       | model                       |   test_mae |   test_rmse |   test_r2 |   test_mbe |   n_obs |   mae_improvement_pct |   rmse_improvement_pct |   r2_delta |
|:--------------|:----------------------------|-----------:|------------:|----------:|-----------:|--------:|----------------------:|-----------------------:|-----------:|
| Aotizhongxin  | Gradient Boosting (Tuned)   |    10.228  |     19.4863 |    0.9576 |    -0.5699 |    4920 |                  7.72 |                   7.48 |     0.0072 |
| Aotizhongxin  | Gradient Boosting (Default) |    10.6596 |     20.0683 |    0.955  |    -0.5517 |    4920 |                  3.82 |                   4.72 |     0.0046 |
| Aotizhongxin  | Linear Regression           |    10.8088 |     19.5187 |    0.9574 |    -0.5931 |    4920 |                  2.47 |                   7.33 |     0.007  |
| Aotizhongxin  | Persistence (Naive Lag-1)   |    11.0831 |     21.0616 |    0.9504 |     0.1254 |    4920 |                  0    |                   0    |     0      |
| Aotizhongxin  | LSTM (Champion W72)         |    13.562  |     22.3737 |    0.9439 |    -1.1889 |    4958 |                -22.37 |                  -6.23 |    -0.0065 |
| Changping     | Gradient Boosting (Tuned)   |    10.331  |     19.9354 |    0.9405 |    -0.1661 |    4973 |                  2.89 |                   1.85 |     0.0023 |
| Changping     | Gradient Boosting (Default) |    10.4691 |     19.8162 |    0.9412 |    -0.0463 |    4973 |                  1.59 |                   2.43 |     0.003  |
| Changping     | Persistence (Naive Lag-1)   |    10.6384 |     20.3107 |    0.9382 |     0.0207 |    4973 |                  0    |                   0    |     0      |
| Changping     | Linear Regression           |    10.8379 |     19.6168 |    0.9424 |     0.1354 |    4973 |                 -1.88 |                   3.42 |     0.0042 |
| Changping     | LSTM (Champion W72)         |    13.0899 |     24.5801 |    0.9091 |    -0.9231 |    5011 |                -23.04 |                 -21.02 |    -0.0291 |
| Dingling      | Gradient Boosting (Tuned)   |     8.2898 |     16.0865 |    0.9519 |     0.6043 |    4695 |                  5.34 |                   7.71 |     0.0083 |
| Dingling      | Gradient Boosting (Default) |     8.4147 |     16.3218 |    0.9505 |     0.4787 |    4695 |                  3.92 |                   6.36 |     0.0069 |
| Dingling      | Persistence (Naive Lag-1)   |     8.7578 |     17.4304 |    0.9436 |     0.1687 |    4695 |                  0    |                   0    |     0      |
| Dingling      | Linear Regression           |     8.9019 |     16.5685 |    0.949  |     0.7741 |    4695 |                 -1.65 |                   4.94 |     0.0054 |
| Dingling      | LSTM (Champion W72)         |    12.1741 |     20.0982 |    0.9257 |     4.4976 |    4765 |                -39.01 |                 -15.31 |    -0.0179 |
| Dongsi        | Gradient Boosting (Tuned)   |    11.1081 |     22.5228 |    0.9531 |    -0.2121 |    4723 |                  6.07 |                   4.98 |     0.005  |
| Dongsi        | Gradient Boosting (Default) |    11.1621 |     22.1571 |    0.9546 |     0.0004 |    4723 |                  5.62 |                   6.53 |     0.0065 |
| Dongsi        | Linear Regression           |    11.5451 |     22.0649 |    0.955  |    -0.2006 |    4723 |                  2.38 |                   6.91 |     0.0069 |
| Dongsi        | Persistence (Naive Lag-1)   |    11.8264 |     23.7038 |    0.9481 |     0.2507 |    4723 |                  0    |                   0    |     0      |
| Dongsi        | LSTM (Champion W72)         |    14.496  |     25.2222 |    0.9411 |    -2.776  |    4778 |                -22.57 |                  -6.41 |    -0.007  |
| Guanyuan      | Gradient Boosting (Tuned)   |    10.2662 |     18.9944 |    0.9593 |     0.477  |    4846 |                  8.48 |                   8.74 |     0.0081 |
| Guanyuan      | Gradient Boosting (Default) |    10.6082 |     19.5409 |    0.957  |     0.4911 |    4846 |                  5.43 |                   6.12 |     0.0058 |
| Guanyuan      | Linear Regression           |    10.8242 |     19.2921 |    0.958  |    -0.1352 |    4846 |                  3.51 |                   7.31 |     0.0068 |
| Guanyuan      | Persistence (Naive Lag-1)   |    11.2177 |     20.8144 |    0.9512 |     0.0167 |    4846 |                  0    |                   0    |     0      |
| Guanyuan      | LSTM (Champion W72)         |    13.7178 |     23.3328 |    0.9391 |    -2.4299 |    4898 |                -22.29 |                 -12.1  |    -0.0121 |
| Gucheng       | Gradient Boosting (Tuned)   |    10.544  |     21.6312 |    0.9549 |    -1.1803 |    4832 |                  3.96 |                   5.62 |     0.0055 |
| Gucheng       | Gradient Boosting (Default) |    10.673  |     21.8636 |    0.9539 |    -1.1054 |    4832 |                  2.78 |                   4.61 |     0.0045 |
| Gucheng       | Persistence (Naive Lag-1)   |    10.9787 |     22.9202 |    0.9494 |     0.1885 |    4832 |                  0    |                   0    |     0      |
| Gucheng       | Linear Regression           |    11.0811 |     21.2872 |    0.9563 |    -1.8789 |    4832 |                 -0.93 |                   7.12 |     0.0069 |
| Gucheng       | LSTM (Champion W72)         |    16.7452 |     28.8676 |    0.92   |    -8.2495 |    4887 |                -52.52 |                 -25.95 |    -0.0294 |
| Huairou       | Gradient Boosting (Tuned)   |     8.4206 |     18.9275 |    0.9399 |     0.37   |    4899 |                  5.18 |                   4.14 |     0.0053 |
| Huairou       | Gradient Boosting (Default) |     8.5448 |     18.8265 |    0.9405 |     0.1588 |    4899 |                  3.78 |                   4.65 |     0.0059 |
| Huairou       | Linear Regression           |     8.8278 |     18.7994 |    0.9407 |     0.3572 |    4899 |                  0.59 |                   4.79 |     0.0061 |
| Huairou       | Persistence (Naive Lag-1)   |     8.8804 |     19.7443 |    0.9346 |     0.0508 |    4899 |                  0    |                   0    |     0      |
| Huairou       | LSTM (Champion W72)         |    11.0841 |     20.5071 |    0.9288 |     0.6717 |    4965 |                -24.82 |                  -3.86 |    -0.0058 |
| Nongzhanguan  | Gradient Boosting (Tuned)   |    10.3369 |     20.3241 |    0.9577 |    -0.2911 |    4835 |                  6.25 |                   4.88 |     0.0045 |
| Nongzhanguan  | Gradient Boosting (Default) |    10.6564 |     20.4517 |    0.9571 |     0.0056 |    4835 |                  3.36 |                   4.28 |     0.0039 |
| Nongzhanguan  | Linear Regression           |    10.8819 |     19.985  |    0.9591 |    -0.2327 |    4835 |                  1.31 |                   6.46 |     0.0059 |
| Nongzhanguan  | Persistence (Naive Lag-1)   |    11.0265 |     21.3657 |    0.9532 |     0.1049 |    4835 |                  0    |                   0    |     0      |
| Nongzhanguan  | LSTM (Champion W72)         |    13.0314 |     22.8001 |    0.9472 |    -2.7684 |    4901 |                -18.18 |                  -6.71 |    -0.006  |
| Shunyi        | Gradient Boosting (Tuned)   |    10.2047 |     20.4124 |    0.9529 |     0.8147 |    4725 |                  9.38 |                   6.54 |     0.0068 |
| Shunyi        | Linear Regression           |    10.3819 |     19.4147 |    0.9574 |     0.222  |    4725 |                  7.81 |                  11.11 |     0.0113 |
| Shunyi        | Gradient Boosting (Default) |    10.4565 |     20.8347 |    0.951  |     0.5916 |    4725 |                  7.15 |                   4.61 |     0.0049 |
| Shunyi        | Persistence (Naive Lag-1)   |    11.2616 |     21.8415 |    0.9461 |    -0.015  |    4725 |                  0    |                   0    |     0      |
| Shunyi        | LSTM (Champion W72)         |    14.3889 |     25.3633 |    0.9269 |    -1.8334 |    4782 |                -27.77 |                 -16.12 |    -0.0192 |
| Tiantan       | Gradient Boosting (Tuned)   |    10.2698 |     18.8796 |    0.9608 |     0.1674 |    4903 |                  8.1  |                   9.4  |     0.0085 |
| Tiantan       | Gradient Boosting (Default) |    10.3967 |     19.0884 |    0.9599 |     0.7277 |    4903 |                  6.96 |                   8.4  |     0.0076 |
| Tiantan       | Linear Regression           |    10.7141 |     18.902  |    0.9607 |     0.3614 |    4903 |                  4.12 |                   9.29 |     0.0084 |
| Tiantan       | Persistence (Naive Lag-1)   |    11.175  |     20.8384 |    0.9523 |     0.0194 |    4903 |                  0    |                   0    |     0      |
| Tiantan       | LSTM (Champion W72)         |    13.2159 |     21.6524 |    0.9484 |     1.158  |    4944 |                -18.26 |                  -3.91 |    -0.0039 |
| Wanliu        | Gradient Boosting (Tuned)   |     9.503  |     19.315  |    0.9557 |     0.4587 |    4951 |                  4.84 |                   3.54 |     0.0033 |
| Wanliu        | Gradient Boosting (Default) |     9.8748 |     19.8455 |    0.9532 |     0.5891 |    4951 |                  1.11 |                   0.89 |     0.0008 |
| Wanliu        | Linear Regression           |     9.9545 |     18.6587 |    0.9586 |    -0.1153 |    4951 |                  0.31 |                   6.82 |     0.0062 |
| Wanliu        | Persistence (Naive Lag-1)   |     9.9859 |     20.0239 |    0.9524 |     0.023  |    4951 |                  0    |                   0    |     0      |
| Wanliu        | LSTM (Champion W72)         |    13.8521 |     23.9968 |    0.9317 |    -2.9466 |    4989 |                -38.72 |                 -19.84 |    -0.0207 |
| Wanshouxigong | Linear Regression           |    11.4169 |     24.2066 |    0.9464 |    -0.134  |    4899 |                  3.68 |                   5.77 |     0.0068 |
| Wanshouxigong | Gradient Boosting (Tuned)   |    11.5123 |     25.2797 |    0.9415 |    -1.0843 |    4899 |                  2.87 |                   1.59 |     0.0019 |
| Wanshouxigong | Gradient Boosting (Default) |    11.5307 |     25.3953 |    0.941  |    -1.0311 |    4899 |                  2.72 |                   1.14 |     0.0014 |
| Wanshouxigong | Persistence (Naive Lag-1)   |    11.8528 |     25.6889 |    0.9396 |     0.0798 |    4899 |                  0    |                   0    |     0      |
| Wanshouxigong | LSTM (Champion W72)         |    14.7763 |     27.3497 |    0.9314 |    -4.9036 |    4937 |                -24.67 |                  -6.47 |    -0.0082 |