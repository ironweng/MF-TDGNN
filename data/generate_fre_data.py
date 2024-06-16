import pandas as pd
from utils import StandardScaler
import pickle
import numpy as np
import yaml

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

set1={'electricity','solar_energy','traffic','exchange_rate'}

for element in set1:
    if(element=='solar_energy'):
        df=np.loadtxt('solar-energy/solar_AL.txt',delimiter=',')
    else:
        df = np.loadtxt(f'{element}/{element}.txt',delimiter=',')

    with open(f'parameters/{element}.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_rate = config.get('train', {}).get('datasets_train_rate', None)

    num_samples = df.shape[0]
    num_train = round(num_samples * train_rate)
    df = df[:num_train]
    scaler = StandardScaler(mean=df.mean(), std=df.std())
    train_feas = scaler.transform(df)

    spectrum = np.fft.fft(train_feas)
    amplitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    save_outpath1 = f'fre_data/{element}_amplitude.pkl'
    pkl_save(save_outpath1, amplitude)
    save_outpath2 = f'fre_data/{element}_phase.pkl'
    pkl_save(save_outpath2, phase)


with open(f'parameters/pems_bay.yaml', 'r') as file:
    config = yaml.safe_load(file)
train_rate_bay = config.get('train', {}).get('datasets_train_rate', None)

df_bay = pd.read_hdf('PEMS-BAY/pems-bay.h5')
num_samples = df_bay.shape[0]
num_train = round(num_samples * train_rate_bay)
df_bay = df_bay[:num_train].values
scaler = StandardScaler(mean=df_bay.mean(), std=df_bay.std())
train_feas = scaler.transform(df_bay)

spectrum_bay = np.fft.fft(train_feas)
amplitude_bay = np.abs(spectrum_bay)
phase_bay = np.angle(spectrum_bay)

save_outpath_bay1=f'fre_data/pems_bay_amplitude.pkl'
pkl_save(save_outpath_bay1,amplitude_bay)
save_outpath_bay2=f'fre_data/pems_bay_phase.pkl'
pkl_save(save_outpath_bay2,phase_bay)



with open(f'parameters/metr_la.yaml', 'r') as file:
    config = yaml.safe_load(file)
train_rate_la = config.get('train', {}).get('datasets_train_rate', None)

df_la = pd.read_hdf('METRLA/metr-la.h5')
num_samples = df_la.shape[0]
num_train = round(num_samples * train_rate_la)
df_la = df_la[:num_train].values
scaler = StandardScaler(mean=df_la.mean(), std=df_la.std())
train_feas = scaler.transform(df_la)

spectrum_la = np.fft.fft(train_feas)
amplitude_la = np.abs(spectrum_la)
phase_la = np.angle(spectrum_la)

save_outpath_la1=f'fre_data/metr_la_amplitude.pkl'
pkl_save(save_outpath_la1,amplitude_la)
save_outpath_la2=f'fre_data/metr_la_phase.pkl'
pkl_save(save_outpath_la2,phase_la)