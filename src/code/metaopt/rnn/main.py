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
# from line_profiler import profile
from memory_profiler import profile



num_epochs = 25
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

# cleanData = map(lambda pair: (pair[0].permute(1, 0, 2), pair[1].permute(1, 0, 2))) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
cleanData = map(lambda pair: zip(pair[0].permute(1, 0, 2), pair[1].permute(1, 0, 2))) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
# cleanData = lambda x:x
epochs = [cleanData(train_loader) for _ in range(num_epochs)]

# x = next(iter(cleanData(train_loader)))
# x = list(x)
# print(len(x))
# x = x[0]
# print(x[0].shape, x[1].shape)
# quit()


W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(2, hidden_size, 1)
alpha_ = 1
optimizer = torch.optim.Adam((W_rec_, W_in_, b_rec_, W_out_, b_out_), lr=learning_rate) 
a0 = torch.zeros(1, hidden_size, dtype=torch.float32)

state0 = VanillaRnnState( a0
                        , 0
                        , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_)
                        , 0)


# because loss is on the per step basis, we have to combine loss at the online level

@profile
def run():
    predictionStep = liftA2(fmapSuffix, predictTrans, activationTrans(activation_))
    lossStep = liftA2(fuse, predictionStep, lossTrans(f.mse_loss))
    loss = compose2(lossStep, foldr)
    paramStep = liftA2(fmapSuffix, parameterTrans(optimizer), loss)
    trainFn = liftA2(apply, repeatRnnWithReset, paramStep)
    trainEpochsFn = liftA2(apply, repeatRnnWithReset, trainFn)(VanillaRnnStateInterpreter())
    stateTrained = trainEpochsFn(epochs, state0)

run()

# rnnPredictor_ = liftA2(apply, repeatRnnWithReset, loss)

# rnnPredictor_ = resetRnnActivation(VanillaRnnStateInterpreter(), fmapSuffix(snd, rnn), a0)
# # print average test mse loss

# avgloss = 0
# with torch.no_grad():
#     for images, labels in cleanData(test_loader):
#         outputs = rnnPredictor_(images, stateTrained)
#         avgloss += outputs

#     print(f'Average test loss: {avgloss / len(test_loader)}')


# plot losses
# plt.plot(losses)
# plt.plot(param_norms)
# plt.show()



# %%