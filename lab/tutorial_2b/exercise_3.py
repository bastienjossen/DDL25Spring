from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ClientEncoder(nn.Module):
    """
    First part of the centralized autoencoder, maping a client's input to a latent representation.
    """
    def __init__(self, input_dim, latent_dim):
        super(ClientEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, 48)  
        self.bn1 = nn.BatchNorm1d(48)
        self.linear2 = nn.Linear(48, 32)           
        self.bn2 = nn.BatchNorm1d(32)
        self.linear3 = nn.Linear(32, 32)           
        self.bn3 = nn.BatchNorm1d(32)
        self.fc = nn.Linear(32, latent_dim)
        self.bn_fc = nn.BatchNorm1d(latent_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.linear2(x)))
        x = self.relu(self.bn3(self.linear3(x)))
        x = self.relu(self.bn_fc(self.fc(x)))
        return x

class ClientDecoder(nn.Module):
    """
    Converts synthetic latent back to the local input space.
    """
    def __init__(self, latent_dim, output_dim):
        super(ClientDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.linear2 = nn.Linear(latent_dim, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.linear3 = nn.Linear(32, 48)
        self.bn3 = nn.BatchNorm1d(48)
        self.linear4 = nn.Linear(48, output_dim)
        self.bn4 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, z):
        z = self.relu(self.bn1(self.linear1(z)))
        z = self.relu(self.bn2(self.linear2(z)))
        z = self.relu(self.bn3(self.linear3(z)))
        z = self.bn4(self.linear4(z))
        return z

class ServerVAE(nn.Module):
    """
    VAE server operating on the concatenated client latents.
    """
    def __init__(self, D_in, H=48, H2=32, latent_dim=16):
        super(ServerVAE, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.lin_bn1 = nn.BatchNorm1d(H)
        self.linear2 = nn.Linear(H, H2)
        self.lin_bn2 = nn.BatchNorm1d(H2)
        self.linear3 = nn.Linear(H2, H2)
        self.lin_bn3 = nn.BatchNorm1d(H2)
        self.fc1 = nn.Linear(H2, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H2)
        self.fc_bn4 = nn.BatchNorm1d(H2)
        self.linear4 = nn.Linear(H2, H2)
        self.lin_bn4 = nn.BatchNorm1d(H2)
        self.linear5 = nn.Linear(H2, H)
        self.lin_bn5 = nn.BatchNorm1d(H)
        self.linear6 = nn.Linear(H, D_in)
        self.lin_bn6 = nn.BatchNorm1d(D_in)
        self.relu = nn.ReLU()
        
    def encode(self, x):
        x = self.relu(self.lin_bn1(self.linear1(x)))
        x = self.relu(self.lin_bn2(self.linear2(x)))
        x = self.relu(self.lin_bn3(self.linear3(x)))
        x = self.relu(self.bn1(self.fc1(x)))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
        
    def decode(self, z):
        z = self.relu(self.fc_bn3(self.fc3(z)))
        z = self.relu(self.fc_bn4(self.fc4(z)))
        z = self.relu(self.lin_bn4(self.linear4(z)))
        z = self.relu(self.lin_bn5(self.linear5(z)))
        z = self.lin_bn6(self.linear6(z))
        return z
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class VFLVAE(nn.Module):
    """
    Combination of client encoders, server VAE, and client decoders.
    """
    def __init__(self, client_encoders, server_vae, client_decoders, client_latent_dim):
        super(VFLVAE, self).__init__()
        self.client_encoders = nn.ModuleList(client_encoders)
        self.server_vae = server_vae
        self.client_decoders = nn.ModuleList(client_decoders)
        self.client_latent_dim = client_latent_dim
        
    def forward(self, x_clients):
        client_latents = [enc(x) for enc, x in zip(self.client_encoders, x_clients)]
        concat_latent = torch.cat(client_latents, dim=1)
        recon_concat, mu, logvar = self.server_vae(concat_latent)
        reconstructed_clients = []
        start = 0
        for dec in self.client_decoders:
            end = start + self.client_latent_dim
            latent_part = recon_concat[:, start:end]
            x_recon = dec(latent_part)
            reconstructed_clients.append(x_recon)
            start = end
        return reconstructed_clients, mu, logvar, concat_latent, recon_concat

def combined_loss(x_clients, recon_clients, concat_latent, recon_concat, mu, logvar):
    mse = nn.MSELoss(reduction='sum')
    client_loss = 0.0
    for orig, recon in zip(x_clients, recon_clients):
        client_loss += mse(recon, orig)
    latent_loss = mse(recon_concat, concat_latent)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return client_loss + latent_loss + kl_loss

if __name__ == "__main__":
    df = pd.read_csv(Path("lab/tutorial_2a/heart.csv"))
    categorical = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df = pd.get_dummies(df, columns=categorical)
    X = df.drop("target", axis=1)
    y = df["target"]
    data = pd.concat([X, y], axis=1)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
    
    total_features = list(data_scaled.columns)
    D_in = len(total_features)
    n_clients = 4
    base = D_in // n_clients
    extra = D_in % n_clients
    client_feature_counts = [base + (1 if i < extra else 0) for i in range(n_clients)]
    
    client_feature_names = []
    start = 0
    for count in client_feature_counts:
        client_feature_names.append(total_features[start:start+count])
        start += count
    
    x_clients = []
    for feats in client_feature_names:
        x_clients.append(torch.tensor(data_scaled[feats].values).float())
    
    client_latent_dim = 8
    client_encoders = []
    client_decoders = []
    for feats in client_feature_names:
        input_dim = len(feats)
        client_encoders.append(ClientEncoder(input_dim, client_latent_dim))
        client_decoders.append(ClientDecoder(client_latent_dim, input_dim))
    
    server_input_dim = n_clients * client_latent_dim
    server_vae = ServerVAE(D_in=server_input_dim, H=48, H2=32, latent_dim=16)
    
    model = VFLVAE(client_encoders, server_vae, client_decoders, client_latent_dim)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1000
    losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        recon_clients, mu, logvar, concat_latent, recon_concat = model(x_clients)
        loss = combined_loss(x_clients, recon_clients, concat_latent, recon_concat, mu, logvar)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss per Epoch")
    plt.show()