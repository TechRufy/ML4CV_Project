import torch
from tqdm import tqdm


def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (images, labels) in tqdm(enumerate(dataloader)):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # Compute prediction and loss
        pred = model(images)
        loss = loss_fn(pred, labels, pred)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
