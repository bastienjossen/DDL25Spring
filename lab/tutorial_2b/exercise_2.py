from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class BottomModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(BottomModel, self).__init__()
        self.local_out_dim = out_feat
        self.fc1 = nn.Linear(in_feat, out_feat)
        self.fc2 = nn.Linear(out_feat, out_feat)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.act(self.fc1(x))
        return self.dropout(self.act(self.fc2(x)))


class TopModel(nn.Module):
    def __init__(self, local_models, n_outs):
        super(TopModel, self).__init__()
        self.in_size = sum([local_models[i].local_out_dim for i in range(len(local_models))])
        self.fc1 = nn.Linear(self.in_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 2)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        concat_outs = torch.cat(x, dim=1)
        x = self.act(self.fc1(concat_outs))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.dropout(x)


class VFLNetwork(nn.Module):
    def __init__(self, local_models, n_outs):
        super(VFLNetwork, self).__init__()
        self.num_cli = None
        self.cli_features = None
        self.bottom_models = local_models
        self.top_model = TopModel(self.bottom_models, n_outs)
        self.optimizer = optim.AdamW(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def train_with_settings(self, epochs, batch_sz, n_cli, cli_features, x, y, log_loss=None):
        self.num_cli = n_cli
        self.cli_features = cli_features
        x = x.astype('float32')
        y = y.astype('float32')
        x_train = [torch.tensor(x[feats].values) for feats in cli_features]
        y_train = torch.tensor(y.values)
        num_batches = len(x) // batch_sz if len(x) % batch_sz == 0 else len(x) // batch_sz + 1
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            correct = 0.0
            total = 0.0
            for minibatch in range(num_batches):
                if minibatch == num_batches - 1:
                    x_minibatch = [x[int(minibatch * batch_sz):] for x in x_train]
                    y_minibatch = y_train[int(minibatch * batch_sz):]
                else:
                    x_minibatch = [x[int(minibatch * batch_sz):int((minibatch + 1) * batch_sz)] for x in x_train]
                    y_minibatch = y_train[int(minibatch * batch_sz):int((minibatch + 1) * batch_sz)]

                outs = self.forward(x_minibatch)
                pred = torch.argmax(outs, dim=1)
                actual = torch.argmax(y_minibatch, dim=1)
                correct += torch.sum((pred == actual))
                total += len(actual)
                loss = self.criterion(outs, y_minibatch)
                total_loss += loss
                loss.backward()
                self.optimizer.step()

            epoch_loss = total_loss.detach().numpy() / num_batches
            if log_loss:
                log_loss(epoch_loss)

    def forward(self, x):
        local_outs = [self.bottom_models[i](x[i]) for i in range(len(self.bottom_models))]
        return self.top_model(local_outs)

    def test(self, x, y):
        x = x.astype('float32')
        y = y.astype('float32')
        x_test = [torch.tensor(x[feats].values) for feats in self.cli_features]
        y_test = torch.tensor(y.values)
        with torch.no_grad():
            outs = self.forward(x_test)
            preds = torch.argmax(outs, dim=1)
            actual = torch.argmax(y_test, dim=1)
            accuracy = torch.sum((preds == actual)) / len(actual)
            loss = self.criterion(outs, y_test)
            return accuracy, loss


if __name__ == "__main__":

    def log_loss(epoch_loss):
        losses.append(epoch_loss)

    CLIENT_COUNTS = [2, 4, 6, 8]  # Experiment with different numbers of clients
    all_losses = {n_clients: [] for n_clients in CLIENT_COUNTS}  # Store losses for each client count
    torch.manual_seed(42)

    for n_clients in CLIENT_COUNTS:
        print(f"Experimenting with {n_clients} clients...")
        np.random.seed(42)
        df = pd.read_csv("lab/tutorial_2a/heart.csv")
        categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df[numerical_cols] = MinMaxScaler().fit_transform(df[numerical_cols])  
        encoded_df = pd.get_dummies(df, columns=categorical_cols)  
        X = encoded_df.drop("target", axis=1)
        Y = pd.get_dummies(encoded_df[['target']], columns=['target'])

        total_features = list(X.columns)
        encoded_df_feature_names = total_features.copy()

        base_features_per_client = len(total_features) // n_clients
        extra_features = len(total_features) % n_clients
        
        features_per_client = [base_features_per_client + (1 if i < extra_features else 0) for i in range(n_clients)]
        client_feature_names = []
        start_index = 0

        for num_feats in features_per_client:
            feat_names = total_features[start_index:start_index + num_feats]
            client_feature_names.append(feat_names)
            start_index += num_feats

        for i in range(len(client_feature_names)):
            updated_names = []
            for column_name in client_feature_names[i]:
                if column_name not in categorical_cols:
                    updated_names.append(column_name)
                else:
                    for name in encoded_df_feature_names:
                        if '_' in name and column_name in name:
                            updated_names.append(name)
            client_feature_names[i] = updated_names

        outs_per_client = 2
        bottom_models = [BottomModel(len(in_feats), outs_per_client * len(in_feats)) for in_feats in client_feature_names]
        final_out_dims = 2
        Network = VFLNetwork(bottom_models, final_out_dims)

        EPOCHS = 300
        BATCH_SIZE = 64
        TRAIN_TEST_THRESH = 0.8
        X_train = X.loc[:int(TRAIN_TEST_THRESH * len(X))]
        X_test = X.loc[int(TRAIN_TEST_THRESH * len(X)) + 1:]
        Y_train = Y.loc[:int(TRAIN_TEST_THRESH * len(Y))]
        Y_test = Y.loc[int(TRAIN_TEST_THRESH * len(Y)) + 1:]

        losses = []
        Network.train_with_settings(EPOCHS, BATCH_SIZE, n_clients, client_feature_names, X_train, Y_train, log_loss)
        all_losses[n_clients] = losses

        accuracy, loss = Network.test(X_test, Y_test)
        print(f"{n_clients} Clients - Test accuracy: {accuracy * 100:.2f}%")
        print(f"{n_clients} Clients - Test loss: {loss:.3f}")
        print(f"Features per client: {client_feature_names}")

    for n_clients, losses in all_losses.items():
        plt.plot(losses, label=f'{n_clients} Clients')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss for Different Client Counts')
    plt.legend()
    plt.show()