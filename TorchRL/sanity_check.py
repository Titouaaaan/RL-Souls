import torch
import torch.nn as nn
import torch.optim as optim

# Check CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dummy model
model = nn.Sequential(
    nn.Linear(1000, 1000),
    nn.ReLU(),
    nn.Linear(1000, 1000)
).to(device)

# Dummy data
x = torch.randn(1024, 1000).to(device)
y = torch.randn(1024, 1000).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for i in range(10000):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f"Step {i}, Loss: {loss.item()}")

# Confirm at least one parameter is on CUDA
for name, param in model.named_parameters():
    print(f"{name} is on {param.device}")
