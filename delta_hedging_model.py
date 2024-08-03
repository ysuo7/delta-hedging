import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy.stats as stats
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
import argparse


def load_data(ticker, end_date):
    data = yf.download(ticker, start='2023-01-01', end=end_date)
    data['Returns'] = data['Adj Close'].pct_change()
    sigma = data['Returns'].std() * np.sqrt(252)  # Annualized volatility
    tbill_data = yf.download('^IRX', start='2023-01-01', end=end_date)
    risk_free_rate = tbill_data['Adj Close'].iloc[-1] / 100
    return data, risk_free_rate, sigma


def gbm_price(S, r, sigma, T, n, num_simulations):
    dt = T / n # time step in terms of years

    paths = np.zeros((n+1,num_simulations))
    paths[0] = S
    for t in range(1,n+1):
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(num_simulations))

    price_array = paths.T
    return price_array


def d1(S0, K, dt, r, sigma):
    M1 = ((np.exp(r * dt) - 1) / (r * dt)) * S0
    return (np.log(M1 / K) + (sigma ** 2 * dt / 2)) / (sigma * np.sqrt(dt))


def d2(S0, K, dt, r, sigma):
    return d1(S0, K, dt, r, sigma) - sigma * np.sqrt(dt)


def price_call_BS(S0, K, dt, r, sigma):

    M1 = ((np.exp(r * dt) - 1) / (r * dt)) * S0
    return (stats.norm.cdf(d1(S0, K, dt, r, sigma)) * M1 -
            stats.norm.cdf(d2(S0, K, dt, r, sigma)) * K) * np.exp(-r * dt)


def sequence_to_span(X_asset, X_option, span_length = 3):
    X_len = len(X_asset)
    asset_span = [X_asset[n:n+span_length] for n in range(X_len) if n < X_len - span_length +1 ]
    span_option = X_option[span_length-1:]
    return asset_span,span_option


def generate_span_dataset(X_asset, X_option, span_length = 3):
    asset_span = []
    span_call =[]
    for Xa,Xo in zip(X_asset, X_option):
        data = sequence_to_span(Xa, Xo, span_length = span_length)

        span   = torch.from_numpy(np.array(data[0])).float()
        option = torch.from_numpy(np.array(data[1])).float()

        asset_span.extend(span)
        span_call.extend(option)
    return asset_span, span_call


class SpanDataset(Dataset):
    def __init__(self,span_X,span_C):
        self.span_X = span_X
        self.span_C = span_C

    def __len__(self):
        return len(self.span_X)

    def __getitem__(self, idx):
        X = self.span_X[idx]
        Sn = X[-1]
        Sn_1 = X[-2]
        Sbar = torch.mean(X).float()
        C = self.span_C[idx]
        return X, C, Sn, Sn_1, Sbar


def get_dataloader(data, batch_size = 256, shuffle=True, drop_last=True):
    return DataLoader(data, batch_size = batch_size, shuffle = shuffle, drop_last = drop_last)


class SpanMLP(nn.Module):
    def __init__(self, span_length):
        super(SpanMLP, self).__init__()
        self.lin1 = nn.Linear(span_length, 32)
        self.tanh1 = nn.Tanh()

        self.lin2 = nn.Linear(32, 4)
        self.tanh2 = nn.Tanh()

        self.lin3 = nn.Linear(4,1)
        self.tanh = nn.Tanh()

        self.span_length = span_length

    def forward(self, X):
        X = X.float()
        X = torch.add(X,torch.range(start=0,end=self.span_length-1,step=1))
        X = self.tanh1(self.lin1(X))
        X = self.tanh2(self.lin2(X))
        out = self.tanh(self.lin3(X))
        return out


def get_model(MODEL_TYPE:str, span_length:int):
    if MODEL_TYPE not in ['SpanMLP']:
        raise ValueError("Invalid Model Type.")

    if MODEL_TYPE == 'SpanMLP':
        model = SpanMLP(span_length)

    return model


def resolve_shape(vector):
    if len(vector.shape) == 1:
        return torch.unsqueeze(vector,-1)
    return vector


def single_epoch_train(model, optimizer, trainloader, loss_func, epoch, model_type:str, K):
    running_loss = 0.0
    outputs_list = []  # List to store outputs for plotting
    Sbar_list = []
    if model_type not in ['SpanMLP']:
        raise ValueError('Please use an available type of model. Available Model: MLP')

    model.train()
    for i, data in enumerate(trainloader):
        span, C, Sn, Sn_1, Sbar = data
        C    = resolve_shape(C)
        Sn   = resolve_shape(Sn)
        Sn_1 = resolve_shape(Sn_1)
        Sbar = resolve_shape(Sbar)
        optimizer.zero_grad()

        outputs = model(span)
        outputs_list.append(outputs.detach().numpy())
        Sbar_list.append(Sbar.detach().numpy())

        loss = loss_func(outputs * (Sn - Sn_1) - C + torch.max(Sbar - K, torch.zeros(256, 1)), torch.zeros(256, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('[%d] loss: %.6f' % (epoch + 1, running_loss))
    return outputs_list, Sbar_list




def hedging(ticker, start_date, end_date, K):
    data, risk_free_rate, sigma = load_data(ticker, end_date)
    expiry_date = pd.to_datetime(end_date)
    current_date = pd.to_datetime(start_date)
    r = risk_free_rate
    days_to_expiry = (expiry_date - current_date).days
    T = days_to_expiry / 365
    n = days_to_expiry
    S = data.iloc[-1]['Adj Close']
    num_simulations = 10000
    X_asset = gbm_price(S, r, sigma, T, n, num_simulations)
    X_call = price_call_BS(X_asset, K, T/n, r, sigma)
    span_X, span_C = generate_span_dataset(X_asset, X_call, span_length = 3)
    ds = SpanDataset(span_X,span_C)
    trainloader = get_dataloader(ds, shuffle=True, drop_last = True)
    MODEL_TYPE='SpanMLP'
    span_length = 3
    net = get_model(MODEL_TYPE, span_length)
    optimizer = optim.Adam(net.parameters(),lr=0.001)
    loss_func = nn.L1Loss()
    all_outputs = []
    Sbar_outputs = []

    for epoch in range(5):
        epoch_outputs, Sbar_outputs = single_epoch_train(net, optimizer, trainloader, loss_func, epoch, MODEL_TYPE, K)
        all_outputs.extend(epoch_outputs)
        Sbar_outputs.extend(Sbar_outputs)

    all_outputs = [item for sublist in all_outputs for item in sublist]
    Y = pd.Series(all_outputs)
    return Y.quantile(0.05)

def main():
    # Argument parser to take input from command line
    parser = argparse.ArgumentParser(description='Delta Hedging Model')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=True, help='End date in YYYY-MM-DD format')

    args = parser.parse_args()
    print("Arguments:", args)
    # Call hedging function with arguments
    result = hedging(args.ticker, args.start_date, args.end_date)
    print("Delta Hedging Result:", result)

if __name__ == "__main__":
    main()