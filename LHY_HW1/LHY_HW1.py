# Numerical Operations
import math
import numpy as np

# Reading / Writing Data
import os
import csv
import pandas as pd

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For Progress Bar
from tqdm import tqdm

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter


def same_seed(seed):
    """
    函数作用: 固定网络优化算法和随机数种子，确保结果可复现.
    """
    # 固定卷积算法为默认算法
    torch.backends.cudnn.deterministic = True
    # 禁止搜索最适合的卷积实现算法 (即固定卷积实现算法)
    torch.backends.cudnn.benchmark = False
    # 固定随机数种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    """
    函数作用: 将训练数据分为训练集和验证集.
    """
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(
        data_set,
        [train_set_size, valid_set_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    """
    函数作用: 使用训练好的模型对测试数据进行预测.
    """
    model.eval()
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


class COVID19Dataset(Dataset):
    """
    x: Features.
    y: Targets, if none, do prediction.
    """

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


class My_Model(nn.Module):
    def __init__(self, input_dim):
        # 在子类的构造函数中，通过 super() 函数调用父类的构造函数，进行必要的初始化。
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    """
    Function: Selects useful features to perform regression.
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = (
        train_data[:, :-1],
        valid_data[:, :-1],
        test_data,
    )

    # TODO: Select suitable feature columns.
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [0, 1, 2, 3, 4]

    return (
        raw_x_train[:, feat_idx],
        raw_x_valid[:, feat_idx],
        raw_x_test[:, feat_idx],
        y_train,
        y_valid,
    )


def trainer(train_loader, valid_loader, model, config, device):
    # Define your loss function, do not modify this.
    criterion = nn.MSELoss(reduction="mean")

    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"], momentum=0.9
    )

    # Writer of tensoboard.
    writer = SummaryWriter()

    # Create directory of saving models.
    if not os.path.isdir("./models"):
        os.mkdir("./models")

    n_epochs, best_loss, step, early_stop_count = config["n_epochs"], math.inf, 0, 0

    for epoch in range(n_epochs):
        # Set your model to train mode.
        model.train()
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            # Set gradient to zero.
            optimizer.zero_grad()
            # Move your data to device.
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            # Compute gradient(backpropagation).
            loss.backward()
            # Update parameters.
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}]")
            train_pbar.set_postfix({"loss": loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar("Loss/train", mean_train_loss, step)

        # Set your model to evaluation mode.
        model.eval()
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(
            f"Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}"
        )
        writer.add_scalar("Loss/valid", mean_valid_loss, step)

        # 如果当前模型的 valid loss 要比之前最小的 loss 小，则记录当前模型.
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config["save_path"])  # Save your best model
            print("Saving model with loss {:.3f}...".format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        # Early stop
        if early_stop_count >= config["early_stop"]:
            print("\nModel is not improving, so we halt the training session.")
            return


def save_pred(preds, file):
    """
    Function: Save predictions to specified file.
    """
    with open(file, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["id", "tested_positive"])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


if __name__ == "__main__":
    # 参数设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        # Seed number
        "seed": 5201314,
        # Whether to use all features
        "select_all": True,
        # validation_size = train_size * valid_ratio
        "valid_ratio": 0.2,
        # Number of epochs
        "n_epochs": 3000,
        "batch_size": 256,
        "learning_rate": 1e-5,
        # If model has not improved for this epochs, stop training.
        "early_stop": 400,
        # The path that model will be saved.
        "save_path": "./models/model.ckpt",
    }

    # Set seed for reproducibility
    same_seed(config["seed"])

    # Dataloader

    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)
    train_data, test_data = (
        pd.read_csv("../dataset/LHY_HW1/covid_train.csv").values,
        pd.read_csv("../dataset/LHY_HW1/covid_test.csv").values,
    )
    train_data, valid_data = train_valid_split(
        train_data, config["valid_ratio"], config["seed"]
    )
    # Print out the data size.
    print(
        f"""train_data size: {train_data.shape}
    valid_data size: {valid_data.shape}
    test_data size: {test_data.shape}"""
    )
    # Select features
    x_train, x_valid, x_test, y_train, y_valid = select_feat(
        train_data, valid_data, test_data, config["select_all"]
    )
    # Print out the number of features.
    print(f"number of features: {x_train.shape[1]}")

    train_dataset, valid_dataset, test_dataset = (
        COVID19Dataset(x_train, y_train),
        COVID19Dataset(x_valid, y_valid),
        COVID19Dataset(x_test),
    )
    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True
    )

    # Start training!
    # put your model and data on the same computation device.
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    trainer(train_loader, valid_loader, model, config, device)

    # Testing
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    model.load_state_dict(torch.load(config["save_path"]))
    preds = predict(test_loader, model, device)
    save_pred(preds, "pred.csv")
