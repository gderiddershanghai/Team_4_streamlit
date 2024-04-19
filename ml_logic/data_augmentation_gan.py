import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class Generator(nn.Module):
    def __init__(self, input_dim, num_classes, output_dim=50):
        super(Generator, self).__init__()
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, 128),  # Adjust input dimension to include class labels
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    def forward(self, z, labels):
        # Embed labels
        c = self.label_embedding(labels).view(labels.size(0), -1)
        # Concatenate label embeddings with noise z
        x = torch.cat([z, c], dim=1)  # Ensure correct concatenation axis
        return self.net(x)



class Discriminator(nn.Module):
    def __init__(self, input_dim=48, num_classes=2):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        # Adjust the input dimension to account for concatenated label embeddings
        self.net = nn.Sequential(
            nn.Linear(input_dim + num_classes, 512),  # input_dim now includes label embedding dimension
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        # Embed labels (already suitable for concatenation)
        c = self.label_embedding(labels).view(labels.size(0), -1)
        # Concatenate label embeddings with x along the feature dimension
        x = torch.cat([x, c], dim=1)  # This produces the correct shape for subsequent layers
        return self.net(x)



def create_dataloader(X, y, batch_size=64):
    """
    Creates a DataLoader from Pandas DataFrames for a Conditional GAN setup.

    Parameters:
    - X (pd.DataFrame): Features in a DataFrame.
    - y (pd.DataFrame or pd.Series): Labels in a DataFrame or Series.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - DataLoader: A PyTorch DataLoader containing the dataset, suitable for cGAN.
    """
    # Convert the Pandas DataFrame/Series to torch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    # Ensure labels are of type torch.long for embedding layers and other categorical operations
    y_tensor = torch.tensor(y.values, dtype=torch.long)  # Adjusted dtype here

    # In case y_tensor is a single dimension, ensure it's correctly shaped
    if y_tensor.ndim == 1:
        y_tensor = y_tensor.unsqueeze(1)

    # Create a dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader




# #####################CODE FOR IMPORTING TO NOTEBOOK###############################
# from data_augmentation import Generator, Discriminator, create_dataloader

# # # Hyperparameters
# latent_dim = 100
# output_dim = 45
# lr = 0.0002
# batch_size = 64
# epochs = 200
# # Initialize the Generator and Discriminator
# dataloader = create_dataloader(X_train, y_train)
# input_dim = 100  # Size of the noise vector
# generator = Generator(input_dim)
# discriminator = Discriminator(output_dim)

# # Optimizers
# optim_g = torch.optim.Adam(generator.parameters(), lr=lr)
# optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

# # Loss function
# criterion = nn.BCELoss()

# # For GPU acceleration, if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generator.to(device)
# discriminator.to(device)

# for epoch in range(epochs):
#     for i, (features, labels) in enumerate(dataloader):  # Correctly unpack the features and labels
#         # Ensure batch size matches current batch (last batch can be smaller)
#         current_batch_size = features.size(0)

#         # ---------------------
#         # Train Discriminator
#         # ---------------------
#         optim_d.zero_grad()

#         # Move data to device
#         real_data = features.to(device)
#         real_labels = torch.ones(current_batch_size, 1).to(device)  # Use current_batch_size

#         # Train with real data
#         real_loss = criterion(discriminator(real_data), real_labels)

#         # Generate fake data
#         noise = torch.randn(current_batch_size, latent_dim, device=device)  # Use current_batch_size and generate noise directly on the device
#         fake_data = generator(noise)
#         fake_labels = torch.zeros(current_batch_size, 1, device=device)  # Use current_batch_size and move to device

#         # Train with fake data
#         fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)

#         # Gradient backprop & optimize for discriminator
#         d_loss = real_loss + fake_loss
#         d_loss.backward()
#         optim_d.step()

#         # -----------------
#         # Train Generator
#         # -----------------
#         optim_g.zero_grad()

#         # All generated data should be considered real by the generator for loss calculation
#         g_loss = criterion(discriminator(fake_data), real_labels)  # Note: real_labels reused here to indicate fake data should be seen as real

#         # Gradient backprop & optimize for generator
#         g_loss.backward()
#         optim_g.step()

#     print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")
