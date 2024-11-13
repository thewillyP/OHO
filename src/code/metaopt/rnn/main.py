from itertools import cycle
import time
import torch 
from torch.nn import functional as f
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from delayed_add_task import createExamplesIO, randomSineExampleIO, randomSparseIO
from learning import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from line_profiler import profile
import wandb
# from memory_profiler import profile


# wandb.init(
#         # set the wandb project where this run will be logged
#         project="my-awesome-project",

#         # track hyperparameters and run metadata
#         config={
#             "tr_loss": 0.0,
#             "vl_loss": 0.0,
#             "te_loss": 0.0,
#             "eta": args.lr,
#             "l2": args.lambda_l2,
#             "dFdlr": 0.0,
#             "dFdl2": 0.0,
#             "grad_norm": 0.0,
#             "grad_norm_vl": 0.0,
#             "gang": 0.0,
#             "param_norm": 0.0,
#             "grad_corr_mean": 0.0,
#             "grad_corr_std": 0.0,
#             "Hv_lr": 0.0    
#         }
#     )


num_epochs = 100
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
                        , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_))


# because loss is on the per step basis, we have to combine loss at the online level

@profile
def run():
    predictionStep = liftA2(fmapSuffix, predictTrans, activationTrans(activation_))
    lossStep = liftA2(fuse, predictionStep, lossTrans(f.mse_loss))
    loss = compose2(lossStep, foldr)
    paramStep = liftA2(fmapSuffix, parameterTrans(optimizer), loss)
    trainFn = liftA2(apply, repeatRnnWithReset, paramStep)

    # state = state0
    # for ds in epochs:
    #     state = VanillaRnnStateInterpreter().putActivation(a0, state)
    #     state = VanillaRnnStateInterpreter().putLoss(0, state)
    #     state = trainFn(VanillaRnnStateInterpreter())(ds, state)
    trainEpochsFn = liftA2(apply, repeatRnnWithReset, trainFn)(VanillaRnnStateInterpreter())
    stateTrained = trainEpochsFn(epochs, state0)

    stateTrained_ = VanillaRnnStateInterpreter().putActivation(a0, stateTrained)
    stateTrained_ = VanillaRnnStateInterpreter().putLoss(0, stateTrained_)
    # x = compose2(predictionStep(VanillaRnnStateInterpreter()), snd)

    rnnPredictor_ = liftA2(apply, repeatRnnWithReset, loss) 
    with torch.no_grad():
        avgloss = rnnPredictor_(VanillaRnnStateInterpreter())(cleanData(test_loader), stateTrained_).loss / len(test_loader)
        print(f'Average test loss: {avgloss}')



start = time.time()
run()
print(time.time() - start)


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







# # Hyper-parameters 
# # input_size = 784 # 28x28
# num_classes = 10
# num_epochs = 2
# batch_size = 100

# input_size = 28
# sequence_length = 28
# hidden_size = 128
# num_layers = 1

# alpha_ = 1
# activation_ = f.tanh
# learning_rate = 0.001




# # MNIST dataset 
# train_dataset = torchvision.datasets.MNIST(root='./data', 
#                                         train=True, 
#                                         transform=transforms.ToTensor(),  
#                                         download=True)

# test_dataset = torchvision.datasets.MNIST(root='./data', 
#                                         train=False, 
#                                         transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
#                                         batch_size=batch_size, 
#                                         shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
#                                         batch_size=batch_size, 
#                                         shuffle=False)
# cleanData = map(lambda pair: (pair[0].squeeze(1).permute(1, 0, 2), pair[1])) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
# epochs = [cleanData(train_loader) for _ in range(num_epochs)]



# W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
# alpha_ = 1
# optimizer = torch.optim.Adam((W_rec_, W_in_, b_rec_, W_out_, b_out_), lr=learning_rate) 
# a0 = torch.zeros(1, hidden_size, dtype=torch.float32)

# state0 = VanillaRnnState( a0
#                         , 0
#                         , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_)
#                         , 0)

# evolveStep = compose2(activationTrans(activation_), foldr)
# predictionStep = liftA2(fmapSuffix, predictTrans, evolveStep)
# lossStep = liftA2(fuse, predictionStep, lossTrans(f.cross_entropy))
# paramStep = liftA2(fmapSuffix, parameterTrans(optimizer), lossStep)
# trainFn = liftA2(apply, repeatRnnWithReset, paramStep)
# trainEpochsFn = liftA2(apply, repeatRnnWithReset, trainFn)(VanillaRnnStateInterpreter())
# stateTrained = trainEpochsFn(epochs, state0)