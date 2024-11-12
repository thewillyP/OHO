from itertools import cycle
import torch 
from torch.nn import functional as f
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from delayed_add_task import createExamplesIO, randomSineExampleIO, randomSparseIO
from learning import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


num_epochs = 2
batch_size = 100
hidden_size = 200

alpha_ = 1
activation_ = f.relu
learning_rate = 0.001

def getDataset(loader, numEx_tr: int, numEx_vl: int, numEx_te: int):
    return [loader(numEx_tr), cycle(loader(numEx_vl)), loader(numEx_te)]


@curry
def load_adder_task(randomExamplesIO, randomAdderIO, t1: float, t2: float, ts, numEx: int):
    seq_length = len(ts)
    xs, ys = randomExamplesIO(numEx, ts, lambda: randomAdderIO(t1, t2))
    ts_broadcasted = ts.view(1, seq_length, 1).expand(numEx, seq_length, 1)
    ys[ts_broadcasted < max(t1, t2)] = 0
    ds = TensorDataset(xs, ys)
    
    dl = DataLoader(ds, batch_size=100, shuffle=True)
    return dl

load_sinadder = load_adder_task(createExamplesIO, randomSineExampleIO)
load_sparse = lambda outT: load_adder_task(createExamplesIO, randomSparseIO(outT))
loader = load_sparse(9)(5, 1, torch.arange(0, 10))
train_loader, _, test_loader = getDataset(loader, 1000, 1000, 200)
cleanData = map(lambda pair: (pair[0].permute(1, 0, 2), pair[1])) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
# cleanData = lambda x:x


W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(2, hidden_size, 1)
alpha_ = 1
optimizer = torch.optim.Adam((W_rec_, W_in_, b_rec_, W_out_, b_out_), lr=learning_rate) 
a0 = torch.zeros(1, hidden_size, dtype=torch.float32)

state0 = VanillaRnnState( a0
                        , 0
                        , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_))

rnn = getRnn(VanillaRnnStateInterpreter(), activation_)
trainFn = trainRnn(VanillaRnnStateInterpreter(), optimizer, rnn)
trainEpochsFn = repeatRnnWithReset(VanillaRnnStateInterpreter(), trainFn)
epochs = [cleanData(train_loader) for _ in range(num_epochs)]
stateTrained = trainEpochsFn(epochs, state0)







# rnnPredictor_ = resetRnnActivation(VanillaRnnStateInterpreter(), fmapSuffix(snd, rnn), a0)

# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     for images, labels in cleanData(test_loader):
#         outputs = rnnPredictor_(images, stateTrained)
#         _, predicted = torch.max(outputs.data, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()

#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network on the 10000 test images: {acc} %')


# plot losses
# plt.plot(losses)
# plt.plot(param_norms)
# plt.show()



# %%