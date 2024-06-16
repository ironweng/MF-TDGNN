# Multivariate Time Series Forecasting based on Deep Graph Construction and Feature Graph Learning


## Data Preparation

### Multivariate time series datasets

Download Solar-Energy, Traffic, Electricity, Exchange-rate datasets from https://github.com/laiguokun/multivariate-time-series-data.

### Traffic datasets
The traffic data files for Los Angeles (METR-LA) and the Bay Area (PEMS-BAY) are  provided by [DCRNN](https://github.com/chnsh/DCRNN_PyTorch).
And the PEMS-D7 datasets are get from https://github.com/zyplanet/TPGNN.

Once you have all the data sets, create a separate folder under `/data` for each data set. You get a file path like this 
`data/{METR-LA,PEMS-BAY,electricity,exchange_rate,solar_energy,traffic}`
, and place each data set in the corresponding folder.

```bash
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY,electricity,exchange_rate,solar_energy,traffic}
```

### Data processing
Run the following commands to generate train/test/val dataset at  `data/{METR-LA,PEMS-BAY}/{train,val,test}.npz`.

```bash
# METR-LA
python  data/generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python  data/generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```
In addition, you need to create a `/data/fre_data` folder to store the frequency domain data.
Run the following commands to generate frequency domain data.

```bash
# Create data directorie
mkdir -p data/{fre_data}

python  data/generate_fre_data.py 
```

## How to run

When you train the model, you can run:

```bash
# Use METR-LA dataset
python trainM.py --config_filename=data/parameters/metr_la.yaml 

# Use PEMS-BAY dataset
python trainM.py --config_filename=data/parameters/pems_bay.yaml 

# Use Solar-EnergyY dataset
python trainS.py --config_filename=data/parameters/solar_energy.yaml 

# Use Traffic dataset
python trainS.py --config_filename=data/parameters/traffic.yaml 

# Use Electricity dataset
python trainS.py --config_filename=data/parameters/electricity.yaml 

# Use Exchange-rate dataset
python trainS.py --config_filename=data/parameters/exchange_rate.yaml 
```

Hyperparameters can be modified in the `metr_la.yaml`, `pems_bay.yaml`, `solar_energy.yaml`, `traffic.yaml`, `electricity.yaml` and `exchange_rate.yaml`  files.

