import math

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model import MFDGNNModel
from utils import load_dataset,StandardScaler
from loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss
import pandas as pd
import os
import pickle
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import time
from utils import DataLoaderS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def pkl_save(path, var):
#     with open(path, 'wb') as f:
#         pickle.dump(var, f)

def pkl_load(file_path):
    with open(file_path, 'rb') as f:
        data=pickle.load(f)
    return data

class MFDGNNSupervisor:
    def __init__(self, **configparams):
        self._configparams = configparams
        self._data_configparams = configparams.get('data')
        self._model_configparams = configparams.get('model')
        self._train_configparams = configparams.get('train')
        self.sampling = float(0.5)
        self.max_grad_norm = self._train_configparams.get('max_grad_norm', 1.)
        self.ANNEAL_RATE = 0.00003
        self.temp_min = 0.1
        # self.save_adj_name = save_adj_name
        self.epoch_use_graph_optimization = self._train_configparams.get('epoch_use_graph_optimization')
        self.use_frequency=self._train_configparams.get('use_frequency')
        self.batch_size=self._data_configparams.get('batch_size')
        self.train_rate=self._train_configparams.get('datasets_train_rate')
        self.patience =self._train_configparams.get('patience')




        if self._data_configparams['dataset_dir'] == 'data/METR-LA':
            self.df = pd.read_hdf('data/METR-LA/metr-la.h5')
            self.amp_file='metr_la_amplitude'
            self.pha_file='metr_la_phase'
            self._data = load_dataset(**self._data_configparams)
            self.standard_scaler = self._data['scaler']
        elif self._data_configparams['dataset_dir'] == 'data/PEMS-BAY':
            self.df = pd.read_hdf('data/PEMS-BAY/pems-bay.h5')  # df(34272,207),(numsamples,nodes)
            self.amp_file = 'pems_bay_amplitude'
            self.pha_file = 'pems_bay_phase'
            self._data = load_dataset(**self._data_configparams)
            self.standard_scaler = self._data['scaler']
        elif self._data_configparams['dataset_dir'] == 'data/solar-energy':
            self.df=np.loadtxt('data/solar-energy/solar_AL.txt', delimiter=',')
            self.amp_file = 'solar_energy_amplitude'
            self.pha_file = 'solar_energy_phase'
        elif self._data_configparams['dataset_dir'] == 'data/electricity':
            self.df=np.loadtxt('data/electricity/electricity.txt', delimiter=',')
            self.amp_file = 'electricity_amplitude'
            self.pha_file = 'electricity_phase'
        elif self._data_configparams['dataset_dir'] == 'data/exchange_rate':
            self.df=np.loadtxt('data/exchange_rate/exchange_rate.txt', delimiter=',')
            self.amp_file = 'exchange_rate_amplitude'
            self.pha_file = 'exchange_rate_phase'
        elif self._data_configparams['dataset_dir'] == 'data/traffic':
            self.df=np.loadtxt('data/traffic/traffic.txt', delimiter=',')
            self.amp_file = 'traffic_amplitude'
            self.pha_file = 'traffic_phase'

        num_samples = self.df.shape[0]
        num_train = round(num_samples * self.train_rate)

        if isinstance(self.df, np.ndarray):
            df=self.df[:num_train]
        else:
            df = self.df[:num_train].values

        scaler = StandardScaler(mean=df.mean(), std=df.std())
        train_feas = scaler.transform(df)
        self._train_feas = torch.Tensor(train_feas).to(device)


        k = self._train_configparams.get('knn_k')
        knn_metric = 'cosine'  # 这段代码用于构建一个图（邻接矩阵），k ，即每个传感器与其最近的 k个传感器建立连接。
        from sklearn.neighbors import kneighbors_graph  # k最近邻用余弦相似度算，使用 kneighbors_graph 函数计算k最近邻图的邻接矩阵。
        g = kneighbors_graph(train_feas.T, k, metric=knn_metric)  # train_feas.T确保每行代表一个特征（传感器），每列代表一个样本。
        g = np.array(g.todense(), dtype=np.float32)  # 将稀疏矩阵表示的邻接矩阵转换为密集矩阵，最终g包含了表示传感器之间关联关系的邻接矩阵。
        self.adj_mx = torch.Tensor(g).to(device)
        self.num_nodes = int(self._model_configparams.get('num_nodes', 1))
        self.input_dim = int(self._model_configparams.get('input_dim', 1))
        self.seq_len = int(
            self._model_configparams.get('seq_len'))  # for the encoder 这个参数用于编码器（encoder）。这是MFDGNN模型中用于处理时间序列数据的长度。
        self.output_dim = int(self._model_configparams.get('output_dim', 1))
        # self.use_curriculum_learning = bool(
        #     self._model_configparams.get('use_curriculum_learning', False))
        self.horizon = int(
            self._model_configparams.get('horizon', 1))  # for the decoder 这个参数用于解码器（decoder）。它表示模型在未来的多少个时间步骤上进行预测。


        self.MFDGNN_model=MFDGNNModel(self.sampling, **self._model_configparams)
        print("Model created")
        self.epochs = self._train_configparams.get('epochs')


    def trainM(self, **configparams):
        configparams.update(self._train_configparams)
        return self._trainM(**configparams)

    def evaluateM(self, dataset='val',gumbel_soft=True):
    
        with torch.no_grad():
            self.MFDGNN_model = self.MFDGNN_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mapes = []
            # rmses = []
            mses = []
            temp = self.sampling

            l_3 = []
            m_3 = []
            r_3 = []
            l_6 = []
            m_6 = []
            r_6 = []
            l_12 = []
            m_12 = []
            r_12 = []

            for batch_idx, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output, mid_output = self.MFDGNN_model(x, self._train_feas,temp, gumbel_soft)

                loss_t = self._compute_loss(y, output)
                pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
                true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                compute_loss = torch.nn.BCELoss()
                loss_g = compute_loss(pred, true_label)
                loss = loss_t + loss_g
                losses.append((loss_t.item() + loss_g.item()))

                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(output)
                mapes.append(masked_mape_loss(y_pred, y_true).item())
                # rmses.append(masked_rmse_loss(y_pred, y_true).item())
                mses.append(masked_mse_loss(y_pred, y_true).item())

                # Followed the DCRNN TensorFlow Implementation
                l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
                m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
                r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
                l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
                m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
                r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
                l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
                m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
                r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())

                # if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
            mean_loss = np.mean(losses)
            mean_mape = np.mean(mapes)
            mean_rmse = np.sqrt(np.mean(mses))
            # mean_rmse = np.mean(rmses) #another option


                # Followed the DCRNN PyTorch Implementation
            result1 = 'Test: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_loss, mean_mape, mean_rmse)
                # print(result1)

                # Followed the DCRNN TensorFlow Implementation
            result2 = 'Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_3), np.mean(m_3),
                                                                                           np.sqrt(np.mean(r_3)))
                # print(result2)
            result3 = 'Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_6), np.mean(m_6),
                                                                                           np.sqrt(np.mean(r_6)))
                # print(result3)
            result4 = 'Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(np.mean(l_12), np.mean(m_12),
                                                                                           np.sqrt(np.mean(r_12)))

            return result1,result2,result3,result4


    def _trainM(self, base_lr,
               test_every_n_epochs=10, epsilon=1e-8, **configparams):
        # steps is used in learning rate - will see if need to use it?
        # print("fre_data",fre_data.shape)
        path_la_amp = f'data/fre_data/{self.amp_file}.pkl'
        path_la_pha = f'data/fre_data/{self.pha_file}.pkl'
        fre_data_amp = pkl_load(path_la_amp)
        fre_data_amp = torch.Tensor(fre_data_amp).to(device)
        fre_data_pha = pkl_load(path_la_pha)
        fre_data_pha = torch.Tensor(fre_data_pha).to(device)
        fre_data = torch.cat((fre_data_amp, fre_data_pha), dim=0)

        self.MFDGNN_model = self.MFDGNN_model.train()
        min_train_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.MFDGNN_model.parameters(), lr=base_lr, eps=epsilon)


        print('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        print("num_batches:{}".format(num_batches))
        batches_complete = 0
        for epoch in range(1,  self.epochs+ 1):
            print("Num of epoch:", epoch)
            train_iterator = self._data['train_loader'].get_iterator()
            losses = []
            start_time = time.time()
            temp = self.sampling
            gumbel_soft = True

            for batch_idx, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()
                x, y = self._prepare_data(x,y)
                # x: shape (batch_size, seq_len, num_sensor, input_dim)->(seq_len, batch_size, num_sensor * input_dim)
                # y: shape (batch_size, horizon, num_sensor, input_dim)->(horizon, batch_size, num_sensor * output_dim)

                label = self.use_frequency
                self.MFDGNN_model.to(device)
                output, mid_output = self.MFDGNN_model(label, x, self._train_feas,temp, y, batches_complete,fre_data)
                # output(12,64,207)  mid_output(207,207)
                # if batch_idx % 100 == 1:
                #    temp = np.maximum(temp * np.exp(-self.ANNEAL_RATE * batch_idx), self.temp_min)
                loss_t = self._compute_loss(y, output)  # loss_t是预测的loss
                true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                if (epoch > self.epoch_use_graph_optimization):
                    pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)  # lossg是表示构造的邻接矩阵和先验邻接矩阵(KNN)的接近程度
                    loss = loss_t + 0.3*loss_g
                    losses.append((loss_t.item() + loss_g.item()))
                else:
                    loss = loss_t
                    losses.append(loss_t.item())
                batches_complete += 1
                # print("batches_complete",batches_complete)
                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.MFDGNN_model.parameters(), self.max_grad_norm)
                optimizer.step()
            train_loss=np.mean(losses)
            # print('Epoch [{}/{}] training complete'.format(epoch, self.epochs))
            end_time = time.time()
            epoch_time=end_time-start_time
            result = 'Epoch [{}/{}] ({}) train_loss: {:.4f},train_time:{:.2f}s'.format(epoch, self.epochs,batches_complete, train_loss,epoch_time)
            print(result)

            if (epoch % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss,result_step3,result_step6,result_step12= self.evaluateM(dataset='test',gumbel_soft=gumbel_soft)
                print("Epoch",epoch)
                print(test_loss)
                print(result_step3)
                print(result_step6)
                print(result_step12)

            if train_loss < min_train_loss:
                wait = 0
                min_train_loss = train_loss

            elif train_loss >= min_train_loss:
                wait += 1
                if wait == self.patience:
                    print('Early stopping at epoch:', epoch)
                    break

    def evaluateS(self,model,Data, evaluateL2, evaluateL1,temp):
        model.eval()
        total_loss = 0
        total_loss_l1 = 0
        n_samples = 0
        correlations = []

        for X, Y in Data.get_batches(Data.test[0], Data.test[1], self.batch_size, False):
            X = X.permute(1, 0, 2)
            Y = Y.unsqueeze(0).repeat(self.horizon, 1, 1)
            model.to(device)
            with torch.no_grad():
                output, _ = self.MFDGNN_model(X, self._train_feas, temp, Y)
            output = torch.squeeze(output)
            Y= torch.squeeze(Y)

            scale = Data.scale.expand(output.size(0), Data.m)
            total_loss += evaluateL2(output*scale , Y*scale ).item()
            total_loss_l1 += evaluateL1(output*scale , Y*scale ).item()
            n_samples += (output.size(0) * Data.m)

            for i in range(output.size(1)):
                output_series = output[:, i]
                Y_series = Y[:, i]

                if output_series.std() > 0 and Y_series.std() > 0:  # Ensure std deviation is not zero
                    covariance = torch.mean((output_series - output_series.mean()) * (Y_series - Y_series.mean()))
                    correlation = covariance / (output_series.std() * Y_series.std())
                    correlations.append(abs(correlation.item()))

        mean_correlation = sum(correlations) / len(correlations)
        rse = math.sqrt(total_loss / n_samples) / Data.rse
        rae = (total_loss_l1 / n_samples) / Data.rae


        return rse, rae,mean_correlation

    def trainS(self, base_lr=0.001,
               test_every_n_epochs=10, epsilon=1e-8, **configparams):

        path_la_amp = f'data/fre_data/{self.amp_file}.pkl'
        path_la_pha = f'data/fre_data/{self.pha_file}.pkl'
        fre_data_amp = pkl_load(path_la_amp)
        fre_data_amp = torch.Tensor(fre_data_amp).to(device)
        fre_data_pha = pkl_load(path_la_pha)
        fre_data_pha = torch.Tensor(fre_data_pha).to(device)
        fre_data = torch.cat((fre_data_amp, fre_data_pha), dim=0)

        self.MFDGNN_model = self.MFDGNN_model.train()
        min_train_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.MFDGNN_model.parameters(), lr=base_lr, eps=epsilon)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=float(lr_decay_ratio))
        criterion = nn.MSELoss().to(device)
        evaluateL2 = nn.MSELoss().to(device)
        evaluateL1 = nn.L1Loss().to(device)
        print('Start training ...')
        Data = DataLoaderS(self.df,self.train_rate, 0.1, device,self.horizon , self.seq_len, 2)
        # this will fail if model is loaded with a changed batch_size
        num_batches = Data.train[0].shape[0]//self.batch_size+1
        print("num_batches:",num_batches)
        # print("self._epoch_num",self._epoch_num)
        batches_seen = 0
        # print("batches_seen",batches_seen)
        for epoch in range(1,  self.epochs+ 1):
            print("Num of epoch:", epoch)

            losses = []
            # start_time = time.time()
            temp = self.sampling
            for x, y in Data.get_batches(Data.train[0], Data.train[1], self.batch_size, True):
                optimizer.zero_grad()
                # x: shape (batch_size, seq_len, num_sensor, input_dim)->(seq_len, batch_size, num_sensor * input_dim)
                # y: shape (batch_size, horizon, num_sensor, input_dim)->(horizon, batch_size, num_sensor * output_dim)
                # print("x", x.shape)
                # print("y", y.shape)
                # print("self._train_feas", self._train_feas.shape)
                x = x.permute(1, 0, 2).to(device)
                y=y.unsqueeze(0).repeat(self.horizon, 1, 1).to(device)
                label = self.use_frequency
                self.MFDGNN_model.to(device)
                output, mid_output = self.MFDGNN_model(label, x, self._train_feas, temp, y, batches_seen,fre_data)

                scale = Data.scale.expand(output.size(0), output.size(1), Data.m)
                loss_t = criterion(y*scale,output*scale)
                true_label = self.adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
                if (epoch > self.epoch_use_graph_optimization):
                    pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
                    compute_loss = torch.nn.BCELoss()
                    loss_g = compute_loss(pred, true_label)  # lossg是表示构造的邻接矩阵和先验邻接矩阵(KNN)的接近程度
                    loss = loss_t + loss_g
                    losses.append((loss_t.item() + loss_g.item()))
                else:
                    loss = loss_t
                    losses.append(loss_t.item())
                batches_seen += 1
                # print("batches_seen",batches_seen)
                loss.backward()
                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.MFDGNN_model.parameters(), self.max_grad_norm)
                optimizer.step()
            train_loss = np.mean(losses)
            print("training complete")
            end_time = time.time()

            result = 'Epoch [{}/{}] ({}) train_loss: {:.4f}'.format(epoch, self.epochs, batches_seen, train_loss)
            print(result)

            if (epoch % test_every_n_epochs) == test_every_n_epochs - 1:
                rse, rae, correlation = self.evaluateS(self.MFDGNN_model,Data,evaluateL2,evaluateL1,temp)

                print("Epoch", epoch)
                print('rse',rse)
                print('rae',rae)
                print('corr',correlation)

            if train_loss < min_train_loss:
                wait = 0
                min_train_loss = train_loss

            elif train_loss >= min_train_loss:
                wait += 1
                if wait == self.patience:
                    print('Early stopping at epoch:' ,epoch)
                    break

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true)
