import pickle
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from hier_auto import Hier_auto
from price.case_118 import price_case
from evaluate import MSE,MAPE
import torch.multiprocessing as mp
import time
def main():
    # mp.set_start_method('spawn', force=True)
    torch.cuda.set_device(0)  # 使用 GPU 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_ratio = 0.99
    epochs = 15

    data = pd.read_csv('../data/Offshore Wind Farm Dataset1(WT5).csv', nrows=6)
    df = data.drop('Sequence No.', axis=1)

    # 读取海风数据
    x_data = torch.tensor(np.load('../data/train_data1.npy')).to(dtype=torch.float32)
    y_data = torch.tensor(np.squeeze(np.load('../data/val_data1.npy'), axis=1))[:, 0:1].to(dtype=torch.float32)

    x_data, y_data = x_data.to(device), y_data.to(device)

    with open('../new_data/adag_dict_1.pkl', 'rb') as f: edge_index = pickle.load(f)

    feature_index = {feature: index for index, feature in enumerate(df.columns)}

    split_index = int(len(x_data) * split_ratio)
    X_train, X_test = x_data[0:split_index], x_data[split_index:]
    y_train, y_test = y_data[0:split_index], y_data[split_index:]

    x_price,y_price = x_data[-144:],y_data[-144:]
    trian_dict, test_dict = edge_index[0:split_index],edge_index[split_index:]

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=128)

    auto_model = Hier_auto(
        n_head=8,
        hidden_size=256,
        factor=2,
        dropout=0.5,
        conv_hidden_size=32,
        MovingAvg_window=3,
        activation="gelu",
        encoder_layers=1,
        decoder_layers=1,
        c_in=8,
        seq_lenth=6,
        c_out=1,
        gruop_dec=True
    )

    auto_model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(auto_model.parameters(), lr=0.0002, weight_decay=0.15)
    l1_lambda = 0.15
    # Early stopping parameters
    patience = 15
    best_mse = float('inf')
    patience_counter = 0
    for epoch in range(epochs):
        start_time = time.time()
        auto_model.train()
        total_loss = 0
        I=0
        for batch in train_dataloader:
            inputs, targets = batch
            dicts=trian_dict[I:I+len(inputs)]
            I+=len(inputs)
            optimizer.zero_grad()

            model_output = auto_model(inputs, dicts)
            model_output = model_output.permute(1, 0, 2)
            loss = loss_function(model_output[-1], targets)
            l1_norm = sum(p.abs().sum() for p in auto_model.parameters())
            loss = loss + l1_lambda * l1_norm
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Validation loss
        auto_model.eval()
        with torch.no_grad():
            prediction = auto_model(X_test, test_dict).squeeze(-1)
            val_mse = MSE(y_test, prediction)
            print(val_mse)
        # Check for early stopping
        if val_mse < best_mse:
            best_mse = val_mse
            patience_counter = 0
            # Save the best model
            torch.save(auto_model.state_dict(), '../new_data/nondec_best_model1_val.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

        end_time = time.time()  # 记录结束时间
        epoch_time = end_time - start_time  # 计算每个 epoch 的运行时间

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")

    # Load the best model before inference or further training
    auto_model.load_state_dict(torch.load('../new_data/nondec_best_model1_val.pt'), strict=False)

    windfore_price = auto_model(x_price, price_dict).squeeze(-1)
    Pwmax_value=1.5
    windfore_price = windfore_price*Pwmax_value/100
    y_price = y_price*Pwmax_value/100
    difference = torch.abs(windfore_price - y_price)
    mean = torch.mean(difference).cpu().detach().numpy()
    var = torch.var(difference).cpu().detach().numpy()
    windfore_hourly=windfore_price.T.repeat(11, 1)
    windfore_hourly = windfore_hourly.view(11, 24, 6).mean(dim=2)

    # 计算均值和方差
    Energy_price, Reserve_price, Inertia_price = price_case(windfore_hourly.cpu().detach().numpy() ,Pwmax_value,mean,var,6)
    Energy_price, Reserve_price, Inertia_price = price_case(windfore_hourly.cpu().detach().numpy(), Pwmax_value, mean,
                                                            var, 1)
    prediction = auto_model(X_test, test_dict).squeeze(-1)
    y_test, prediction = y_test.cpu().numpy(), prediction.detach().cpu().numpy()
    print("MSE:", MSE(y_test, prediction))
    print("MAPE:", MAPE(y_test, prediction))


if __name__ == '__main__':
    main()