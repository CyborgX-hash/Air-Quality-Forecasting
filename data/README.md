# Beijing Multi-Site Air Quality Dataset

This project utilizes the **Beijing Multi-Site Air Quality Dataset** hosted by the UCI Machine Learning Repository.

- **Source**: UCI Machine Learning Repository
- **URL**: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- **Dataset DOI / Reference**: Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences*, 473(2205), 20170457.
- **Coverage**: Hourly air pollutant and meteorological records from March 1, 2013 to February 28, 2017 across 12 monitoring stations in Beijing (420,768 total hourly observations; 35,064 records per station).

## Directory Structure

Place the 12 extracted station CSV files into `data/raw/`:

```
data/
├── README.md
└── raw/
    ├── PRSA_Data_Aotizhongxin_20130301-20170228.csv
    ├── PRSA_Data_Changping_20130301-20170228.csv
    ├── PRSA_Data_Dingling_20130301-20170228.csv
    ├── PRSA_Data_Dongsi_20130301-20170228.csv
    ├── PRSA_Data_Guanyuan_20130301-20170228.csv
    ├── PRSA_Data_Gucheng_20130301-20170228.csv
    ├── PRSA_Data_Huairou_20130301-20170228.csv
    ├── PRSA_Data_Nongzhanguan_20130301-20170228.csv
    ├── PRSA_Data_Shunyi_20130301-20170228.csv
    ├── PRSA_Data_Tiantan_20130301-20170228.csv
    ├── PRSA_Data_Wanliu_20130301-20170228.csv
    └── PRSA_Data_Wanshouxigong_20130301-20170228.csv
```

## Quick Download Command

You can download and extract the dataset using:

```bash
mkdir -p data/raw
curl -o data/raw/air_quality.zip "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
unzip -q data/raw/air_quality.zip -d data/raw_extracted/
mv data/raw_extracted/PRSA_Data_20130301-20170228/*.csv data/raw/
rm -rf data/raw/air_quality.zip data/raw_extracted/
```
