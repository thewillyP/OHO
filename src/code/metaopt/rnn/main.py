#%%
import torch 
from torch.nn import functional as f
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from learning import *



# Hyper-parameters 
# input_size = 784 # 28x28
num_classes = 10
num_epochs = 2
batch_size = 100

input_size = 28
sequence_length = 28
hidden_size = 128
num_layers = 1

alpha_ = 1
activation_ = f.tanh
learning_rate = 0.001


# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=True, 
                                        transform=transforms.ToTensor(),  
                                        download=True)

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                        train=False, 
                                        transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=False)
cleanData = map(lambda pair: (pair[0].squeeze(1).permute(1, 0, 2), pair[1])) # origin shape: [N, 1, 28, 28] -> resized: [28, N, 28]



W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(input_size, hidden_size, num_classes)
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





#%%

rnnPredictor_ = resetRnnActivation(VanillaRnnStateInterpreter(), fmapSuffix(snd, rnn), a0)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in cleanData(test_loader):
        outputs = rnnPredictor_(images, stateTrained)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')


# plot losses
plt.plot(losses)
# plt.plot(param_norms)
# plt.show()



# %%
