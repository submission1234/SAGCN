"""
训练
Contains the trainable sub-network of an audio classifier based on the VGGish
pre-trained feature extractor. Handles calling the training and evaluation.
"""
from matplotlib.pyplot import show
import torch
import torch.nn as nn
import torch.optim as optim

from utility.network_functions import train_nn, test_nn
from definitions import (AUDIO_SELFATTENTION_TRAIN_FEATURES_FILE,
                         AUDIO_SELFATTENTION_VAL_FEATURES_FILE,
                         AUDIO_SELFATTENTION_TEST_FEATURES_FILE,
                         BEST_SELFATTENTION_MODEL,
                         VECTORS_PER_AUDIO, AUDIO_SF_EMBED)


class AudioSFModel(nn.Module):
    def __init__(self):
        super(AudioSFModel, self).__init__()
        self.input_dim = AUDIO_SF_EMBED
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Dropout(0.6),

            nn.Linear(256, 10)  # 类别在这
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    run_train = True
    run_test = True            # Test on segment level
    run_mean_test = True         # Test on file level

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model = AudioSFModel()

    if run_train:
        optimizer = optim.Adam(model.parameters(), lr=0.000055)
        epochs = 60
        batch_size = 32
        train_nn(model=model, optimizer=optimizer,
                 train_features=AUDIO_SELFATTENTION_TRAIN_FEATURES_FILE,
                 val_features=AUDIO_SELFATTENTION_VAL_FEATURES_FILE,
                 best_model=BEST_SELFATTENTION_MODEL,
                 device=device,
                 epochs=epochs, batch_size=batch_size)

    if run_test:
        model.load_state_dict(torch.load(BEST_SELFATTENTION_MODEL))
        model.eval()
        test_nn(model=model,
                test_features=AUDIO_SELFATTENTION_TEST_FEATURES_FILE,
                device=device,
                verbose=True)

    if run_mean_test:
        model.load_state_dict(torch.load(BEST_SELFATTENTION_MODEL))
        model.eval()
        test_nn(model=model,
                test_features=AUDIO_SELFATTENTION_TEST_FEATURES_FILE,
                device=device,
                verbose=True,
                batch_size=VECTORS_PER_AUDIO,
                take_batch_mean=True)

    show()

"""
test: 0.609     val: 0.460      val_loss 1.4093      avg: 59.78%     test_loss: 1.2802
optimizer = optim.Adam(model.parameters(), lr=0.000055)
epochs = 60 #(30)
test: 0.608     val: 0.459     val_loss: 1.4099     avg: 59.33%     test_loss: 1.2801
optimizer = optim.Adam(model.parameters(), lr=0.000085)
epochs = 30 #(15)
test: 0.606     val: 0.466     val_loss: 1.4036
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 30 #(15)
batch_size = 32
self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )
"""
