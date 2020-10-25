import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import DataGeneratorPreFetch
from tensorflow.python.platform import flags


class MANN(nn.Module):
    def __init__(self, num_classes, sample_per_class, embed_size=784):
        super().__init__()
        self.num_classes = num_classes
        self.sample_per_classes = sample_per_class
        self.embed_size = embed_size
        self.lstm1 = nn.LSTM(embed_size + num_classes, 128)
        self.lstm2 = nn.LSTM(128, num_classes)

    def forward(self, input_images, raw_input_labels):
        # B, K+1, N, 784: input_images
        # B, K+1, N, N: input_labels
        B, k_plus_one, N, N = raw_input_labels.shape
        input_images = input_images.reshape(B, -1, self.embed_size)
        input_labels = raw_input_labels.copy().reshape(B, -1, N)
        input_labels[:, -self.num_classes:] = 0.
        x = torch.tensor(np.dstack((input_images, input_labels))).transpose(0, 1)  # (K+1)*N, B, N
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x  # ((K + 1)*N, B, N)


def train(num_classes, num_samples, meta_batch_size, print_every=100, n_step = 50000):
    data_generator = DataGeneratorPreFetch(
        num_classes, num_samples + 1)

    model = MANN(num_classes, num_samples + 1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # n_step = 50000
    test_accs = []

    for step in range(n_step):
        images, labels = data_generator.sample_batch('train', meta_batch_size)
        last_n_step_labels = labels[:, -1:].copy()
        last_n_step_labels = last_n_step_labels.squeeze(1).reshape(-1, num_classes)  # (B * N, N)
        target = torch.tensor(last_n_step_labels.argmax(axis=1))
        logits = model(images, labels)
        last_n_step_logits = logits[-num_classes:].\
            transpose(0, 1).contiguous().view(-1, num_classes)
        optimizer.zero_grad()
        loss = criterion(last_n_step_logits, target)
        loss.backward()
        optimizer.step()
        if step % print_every == 0:
            with torch.no_grad():
                print("*" * 5 + "Iter " + str(step) + "*" * 5)
                images, labels = data_generator.sample_batch('test', 100)
                last_n_step_labels = labels[:, -1:].copy()
                last_n_step_labels = last_n_step_labels.squeeze(1).reshape(-1, num_classes)  # (B * N, N)
                target = torch.tensor(last_n_step_labels.argmax(axis=1))
                logits = model(images, labels)
                last_n_step_logits = logits[-num_classes:].\
                    transpose(0, 1).contiguous().view(-1, num_classes)
                pred = last_n_step_logits.argmax(axis=1)
                test_loss = criterion(last_n_step_logits, target)

                print("Train Loss:", loss.item(), "Test Loss:", test_loss.item())
                test_accuracy = (1.0 * (pred == target)).mean().item()
                print("Test Accuracy", test_accuracy)
                test_accs.append(test_accuracy)

    import matplotlib.pyplot as plt
    plt.plot(range(len(test_accs)), test_accs)
    plt.xlabel("Step (x 100)")
    plt.ylabel("Test accuracy")
    plt.show()
    return model

model_k1n2 = train(num_classes=2, num_samples=1, meta_batch_size=4, print_every=100, n_step=2000)
