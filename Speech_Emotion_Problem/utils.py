from python_speech_features import mfcc
from scipy.io import wavfile
import torch
import torch.nn as nn

max_feature = torch.tensor([23.94766958, 35.47760648, 32.407666,   82.12595251, 85.63384494, 62.8985791,
 61.30809941, 79.46652656, 59.85923447, 70.58265384, 43.68436301, 58.37647754,
 54.0187414 ])


min_feature = torch.tensor([-36.04365339, -58.63459727, -57.64907143, -80.63907682, -81.08503093,
 -85.38878632, -75.72724483, -76.89883101, -92.74224276, -90.46006047,
 -75.38062188, -70.24483321, -79.46963393])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(13, 64, 8, batch_first=True)
        self.relu = nn.ReLU()
        self.linear_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
        )
        self.hidden = (torch.zeros(8, 1, 64).float(), torch.zeros(8, 1, 64).float())

    def forward(self, inputs):
        _, hidden = self.lstm(inputs)
        hid = (hidden[0].clone().detach().permute(1, 0, 2)).flatten(start_dim=1)
        output = self.linear_layers(hid)
        return output


def get_model():
    model = Net()
    path = './mffcc-features-normalised-momentum.pth'
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_mfcc_features(file):
    rate, signal = wavfile.read(file)
    return mfcc(signal)


def normalise(features):
    return (features - min_feature)/(max_feature - min_feature)
