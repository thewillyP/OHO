#%%
from itertools import cycle
import time
import torch 
from torch.nn import functional as f
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from delayed_add_task import getDataLoaderIO, DatasetType, randomUniform, sparseIO, waveIO, waveArbitraryUniform, sparseUniformConstOutT, visualizeOutput
from learning import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from line_profiler import profile
import wandb
from matplotlib.ticker import MaxNLocator
from pyrsistent import pdeque, PDeque

torch.manual_seed(0)

# wandb.init(
#         # set the wandb project where this run will be logged
#         project="my-test-project",

#         # track hyperparameters and run metadata
#         config={
#             "tr_loss": 0.0,
#             "te_loss": 0.0, 
#         }
#     )

step = 0
def parameterTrans(opt, lossFn):
    def parameterTrans_(t: Union[HasParameter[MODEL, PARAM], HasLoss[MODEL, torch.Tensor]]) -> Callable[[MODEL], MODEL]:
        def parameterTrans__(env: MODEL) -> MODEL:
            global step
            opt.zero_grad() 
            loss = t.getLoss(env)
            loss.backward()
            W_rec, W_in, b_rec, W_out, b_out, _ = t.getParameter(env)
            torch.nn.utils.clip_grad_norm_((W_rec, W_in, b_rec, W_out, b_out), 1)
            opt.step()  # will actuall physically spooky mutate the param so no update needed. 
            step += 1
            with torch.no_grad():
                if step % 1 == 0:
                    wandb.log({"tr_loss": loss.item()})
                    wandb.log({"te_loss": lossFn(env)})
                    print(f"Step {step}, Loss {loss.item()}")
            return env
        return parameterTrans__
    return parameterTrans_


def predictAccum(t: Union[
    HasActivation[MODEL, torch.Tensor]
    , HasParameter[MODEL, PARAM]
    , HasPrediction[MODEL, PDeque[torch.Tensor]]]) -> Callable[[MODEL], MODEL]:
    def predictTrans_(env: MODEL) -> MODEL:
        a = t.getActivation(env)
        _, _, _, W_out, b_out, _ = t.getParameter(env)
        pred = f.linear(a, W_out, b_out)
        predAccum = t.getPrediction(env).append(pred)
        env_ = t.putPrediction(predAccum, env)
        return env_
    return predictTrans_


def getRandFn(datasetType: DatasetType):
    match datasetType:
        case DatasetType.Random:
            return lambda: (randomUniform, randomUniform)
        case DatasetType.Sparse:
            return lambda: sparseIO(sparseUniformConstOutT(8))
        case DatasetType.Wave:
            return lambda: (waveIO(waveArbitraryUniform), waveIO(waveArbitraryUniform))
        case _:
            raise Exception("Invalid dataset type")
        

num_epochs = 100
batch_size = 200
hidden_size = 200
numExamples = 10
t1 = 2 
t2 = 2 
ts = torch.arange(10)

alpha_ = 1
activation_ = f.relu
learning_rate = 0.01

train_loader = getDataLoaderIO(getRandFn(DatasetType.Random), t1, t2, ts, numExamples, batch_size)
test_loader = getDataLoaderIO(getRandFn(DatasetType.Random), t1, t2, ts, numExamples, batch_size)


visualizeOutput((y for batch in train_loader for y in batch[1]))

#%%

# cleanData = map(lambda pair: (pair[0].permute(1, 0, 2), pair[1].permute(1, 0, 2))) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
cleanData = map(lambda pair: zip(pair[0].permute(1, 0, 2), pair[1].permute(1, 0, 2))) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]
epochs = [cleanData(train_loader) for _ in range(num_epochs)]



W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(2, hidden_size, 1)
alpha_ = 1
optimizer = torch.optim.SGD((W_rec_, W_in_, b_rec_, W_out_, b_out_), lr=learning_rate) 
a0 = torch.zeros(1, hidden_size, dtype=torch.float32)

state0 = VanillaRnnStatePred( a0
                        , 0
                        , (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_)
                        , 0)



# because loss is on the per step basis, we have to combine loss at the online level
predictionStep = liftA2(fmapSuffix, predictTrans, activationTrans(activation_))
lossStep = liftA2(fuse, predictionStep, lossTrans(f.mse_loss))
loss = compose2(lossStep, foldr)

# I have to define a whole new prediction function because there is literally two different ways to go about it.
def modePrediction(env: VanillaRnnStatePred[torch.Tensor, torch.Tensor, PARAM, PDeque[torch.Tensor]]
                , xs: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    pStep = liftA2(fmapSuffix, predictAccum, activationTrans(activation_))
    # If I really want to be exact about it, I would switch out my interpreter here to disallow learning. 
    predictor = pStep(VanillaRnnStatePredInterpreter())
    prediction = foldr(predictor)(xs, env)
    return VanillaRnnStatePredInterpreter().getPrediction(prediction)



def avgLossFn(env):
    env = VanillaRnnStatePredInterpreter().putActivation(a0, env)
    env = VanillaRnnStatePredInterpreter().putLoss(0, env)
    rnnPredictor_ = liftA2(apply, repeatRnnWithReset_, loss) 
    return rnnPredictor_(VanillaRnnStatePredInterpreter())(cleanData(test_loader), env).loss / len(test_loader)


paramStep = liftA2(fmapSuffix, parameterTrans(optimizer, avgLossFn), loss)
trainFn = liftA2(apply, repeatRnnWithReset, paramStep)
trainEpochsFn = liftA2(apply, repeatRnnWithReset, trainFn)(VanillaRnnStatePredInterpreter())
start = time.time()
stateTrained = trainEpochsFn(epochs, state0)
print(time.time() - start)

#%%
RNG = np.random.RandomState(0)
def plotIO2(model, env):
    with torch.no_grad():
        data, labels = next(iter(test_loader))
        xs = data[0]
        ys = labels[0]
        # print(ys)
        # print(env.activation)

        predicts = model(env, xs)
        predicts = torch.tensor(list(predicts))
        # print(predicts)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.plot(torch.arange(0, 10), ys, torch.arange(0, 10), predicts.flatten().detach().numpy(), marker='o')
        # plt.savefig('../../figs/mnist/loss_lr.png', format='png')
        plt.show()

posteriorState = VanillaRnnStatePred(a0, 0, stateTrained.parameter, pdeque([]))

plotIO2(modePrediction, posteriorState)

"""
The problem with wanting to extract predictions out of a fold is that it really is a side effect.
Side effect = monad. 
Although general monadic functions are possible in python at the cost of no typing, and python has no higher kinded types,
I really don't want to recode monad stuff. Just use built in IO monad or use a god state monad.
I already have a pseudo-state monad set up so just use that. 
If I start scanning, then when I fold them, I have to deal with the problem of
1) composing scans with scans with folds? That's too hard to think about. Just use foldr and be at peace.
2) Now I have to deal with the side effect of scans as well as trying to fold the stuff underneath simulatenously? Just use a state! 

It's fine that I need to expand my state monad everytime I need to add a new state. 
I can always switch out how I combine states with natural functions so we gucci. 

"""


# x = compose2(predictionStep(VanillaRnnStateInterpreter()), snd)

# with torch.no_grad():
#     avgloss = rnnPredictor_(VanillaRnnStateInterpreter())(cleanData(test_loader), stateTrained_).loss / len(test_loader)
#     print(f'Average test loss: {avgloss}')






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
# %%
