
import os, sys, math, argparse, time
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

from metaopt.optimizer import SGD_Multi_LR, SGD_Quotient_LR
from itertools import product, cycle
import itertools
import pickle
from functorch import jacrev

from mlp import * 
from metaopt.util import *
from metaopt.util_ml import *
from delayed_add_task import *
from toolz.curried import take_nth, take

TRAIN=0
VALID=1
TEST =2

#torch.manual_seed(3)
ifold=0
RNG = np.random.RandomState(ifold)

"""parsing and configuration"""
def parse_args():  # IO
    desc = "Pytorch implementation of DVAE collections"
    parser = argparse.ArgumentParser(description=desc)


    # parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fmnist'],
    #                     help='The name of dataset')
    # parser.add_argument('--num_epoch', type=int, default=100, help='The number of epochs to run')
    # parser.add_argument('--batch_size', type=int, default=100, help='The size of batch')
    # parser.add_argument('--batch_size_vl', type=int, default=100, help='The size of validation batch')
    # parser.add_argument('--result_dir', type=str, default='results',
    #                     help='Directory name to save the generated images')
    # parser.add_argument('--log_dir', type=str, default='logs',
    #                     help='Directory name to save training logs')
    # parser.add_argument('--model_type', type=str, default='mlp',help="'mlp' | 'amlp'")
    # parser.add_argument('--opt_type', type=str, default='sgd', help="'sgd' | 'sgld'")
    # parser.add_argument('--xdim', type=float, default=784)
    # parser.add_argument('--hdim', type=float, default=128)
    # parser.add_argument('--ydim', type=float, default=10)
    # parser.add_argument('--num_hlayers', type=int, default=3)
    # parser.add_argument('--lr', type=float, default=1e-3)
    # parser.add_argument('--mlr', type=float, default=1e-4)
    # parser.add_argument('--lambda_l1', type=float, default=1e-4)
    # parser.add_argument('--lambda_l2', type=float, default=1e-4)
    # parser.add_argument('--update_freq', type=int, default=1)
    # parser.add_argument('--reset_freq', type=int, default=-0)
    # parser.add_argument('--beta1', type=float, default=0.9)
    # parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--valid_size', type=int, default=10000)
    # parser.add_argument('--checkpoint_freq', type=int, default=10)
    # parser.add_argument('--is_cuda', type=int, default=0)
    # parser.add_argument('--save', type=int, default=0)
    # parser.add_argument('--save_dir', type=str, default='/scratch/ji641/imj/')
    # parser.add_argument('--task', type=str, default='sin')
    # parser.add_argument('--t1', type=float, default=1)
    # parser.add_argument('--t2', type=float, default=1)
    # parser.add_argument('--outT', type=float, default=10)
    # parser.add_argument('--seq', type=int, default=10)
    # parser.add_argument('--numTr', type=int, default=1000)
    # parser.add_argument('--numVl', type=int, default=1000)
    # parser.add_argument('--numTe', type=int, default=200)
    # parser.add_argument('--oho', type=int, default=1)
    

    return check_args(parser.parse_args())



def load_mnist(args):

    ## Initialize Dataset
    dataset = datasets.MNIST('data/mnist', train=True, download=True,
                                        transform=transforms.Compose(
                                                [transforms.ToTensor()]))
    train_set, valid_set = torch.utils.data.random_split(dataset,[60000 - args.valid_size, args.valid_size])


    data_loader_tr = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    data_loader_vl = DataLoader(valid_set, batch_size=args.batch_size_vl, shuffle=True)
    data_loader_te = DataLoader(datasets.MNIST('data/mnist', train=False, download=True,
                                                        transform=transforms.Compose(
                                                                [transforms.ToTensor()])),
                                                batch_size=args.batch_size, shuffle=True)

    data_loader_vl = cycle(data_loader_vl)
    dataset = [data_loader_tr, data_loader_vl, data_loader_te]
    return dataset

def getDataset(loader, numEx_tr: int, numEx_vl: int, numEx_te: int):
    return [loader(numEx_tr), cycle(loader(numEx_vl)), loader(numEx_te)]

@curry
def load_adder_task(randomExamplesIO, randomAdderIO, t1: float, t2: float, ts, args, numEx: int):
    seq_length = len(ts)
    xs, ys = randomExamplesIO(numEx, ts, lambda: randomAdderIO(t1, t2))
    ts_broadcasted = ts.view(1, seq_length, 1).expand(numEx, seq_length, 1)
    ys[ts_broadcasted < max(t1, t2)] = 0
    ds = TensorDataset(xs, ys)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    return dl

load_sinadder = load_adder_task(createExamplesIO, randomSineExampleIO)
load_sparse = lambda outT: load_adder_task(createExamplesIO, randomSparseIO(outT))
def load_random(seq_length, t1, t2):
    ts = torch.arange(0, seq_length)
    return load_adder_task(createExamplesIO2, generate_random_lists, t1, t2, ts)

def main(filename, args, ifold=0, trial=0, quotient=None, device='cuda', is_cuda=1):  # iscuda not even used

    if args.task == 'sin':
        loader = load_sinadder(args.t1, args.t2, torch.arange(0, args.seq), args)
    elif args.task == 'sparse':
        loader = load_sparse(args.outT)(args.t1, args.t2, torch.arange(0, args.seq), args)
    else:
        loader = load_random(args.seq, args.t1, args.t2)(args)
    dataset = getDataset(loader, args.numTr, args.numVl, args.numTe) 

    # dataset = getDataset(load_sinadder(1, 1, torch.arange(0, 20), args), 1000, 1000, 200) 

    ## Initialize Model and Optimizer
    # hdims = [args.xdim] + [args.hdim]*args.num_hlayers + [args.ydim]
    hdims = []
    num_layers = 0  # args.num_hlayers + 2
    if args.model_type == 'amlp':
        model = AMLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = SGD_Multi_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == 'qmlp':
        model = QMLP(num_layers, hdims, args.lr, args.lambda_l2, quotient=quotient, is_cuda=is_cuda)
        optimizer = SGD_Quotient_LR(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2, quotient=quotient)
    elif args.model_type == 'mlp_drop':
        model = MLP_Drop(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    elif args.model_type == 'bptt':
        model = BPTTRNN(2, 300, 1, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    else:
        model = MLP(num_layers, hdims, args.lr, args.lambda_l2, is_cuda=is_cuda)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.lambda_l2)
    
    if args.is_cuda:
        model = model.to(device)
        cudnn.benchmark = True

    # print('Model Type: %s Opt Type: %s Update Freq %d Reset Freq %d' \
    #         % (args.model_type, args.opt_type, args.update_freq, args.reset_freq))
 
    os.makedirs('%s/exp/mnist/' % args.save_dir, exist_ok=True)
    os.makedirs('%s/exp/mnist/mlr%f_lr%f_l2%f/' % (args.save_dir, args.mlr, args.lr, args.lambda_l2), exist_ok=True)
    fdir = '%s/exp/mnist/mlr%f_lr%f_l2%f/%s_%depoch_%dvlbz_%s_%dupdatefreq_%dresetfreq_fold%d/' \
            % (args.save_dir, args.mlr, args.lr, args.lambda_l2, args.model_type, args.num_epoch, args.batch_size_vl, args.opt_type, args.update_freq, args.reset_freq, ifold)
    if quotient is not None:
        fdir = fdir.rstrip('/') + '_quotient%d/' % quotient

    os.makedirs(fdir, exist_ok=True)
    os.makedirs(fdir+'/checkpoint/', exist_ok=True)
    args.fdir = fdir
    # print(args.fdir)
    ## Train 
    Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, tr_epoch, vl_epoch, te_epoch,\
                                tr_acc_list, te_acc_list, tr_loss_list, vl_loss_list, te_loss_list,\
                                tr_corr_mean_list, tr_corr_std_list \
                                = train(args, dataset, model, optimizer, is_cuda=is_cuda)

    # if args.save:


    # os.makedirs(fil, exist_ok=True)

    script_dir = os.path.dirname(__file__)  # Get the directory of the current script
    filename = os.path.join(script_dir, filename)  # Build the relative path

    # np.savez(filename+'.npz'
    #         , Wn=Wn_list
    #         , lr=lr_list
    #         , l2=l2_list
    #         , gang_list=gang_list
    #         , dFdlr_list=dFdlr_list
    #         , dFdl2_list=dFdl2_list
    #         , tr_loss=tr_loss_list
    #         , vl_loss=vl_loss_list
    #         , te_loss=te_loss_list)


    # np.save(filename+'Wn', Wn_list)
    # np.save(filename+'lr', lr_list)
    # np.save(filename+'l2', l2_list)
    # np.save(filename+'gang_list', gang_list)
    # np.save(filename+'dFdlr_list', dFdlr_list)
    # np.save(filename+'dFdl2_list', dFdl2_list)
    # np.save(filename+'tr_epoch', tr_epoch)
    # np.save(filename+'vl_epoch', vl_epoch)
    # np.save(filename+'te_epoch', te_epoch)
    # np.save(filename+'tr_loss', tr_loss_list)
    # np.save(filename+'vl_loss', vl_loss_list)
    # np.save(filename+'te_loss', te_loss_list)
    # np.save(filename+'tr_acc', tr_acc_list)
    # np.save(filename+'te_acc', te_acc_list)
    # np.save(filename+'tr_grad_corr_mean', tr_corr_mean_list)
    # np.save(filename+'tr_grad_corr_std', tr_corr_std_list)

    # print('Final test loss %f' % te_loss_list[-1])
    # print(type(te_loss_list[-1]))

    return model


def train(args, dataset, model, optimizer, saveF=0, is_cuda=1):

    counter = 0
    lr_list, l2_list = [], []
    dFdlr_list, dFdl2_list, Wn_list, gang_list = [], [], [], []
    tr_epoch, tr_loss_list, tr_acc_list = [], [], []
    vl_epoch, vl_loss_list, vl_acc_list = [], [], []
    te_epoch, te_loss_list, te_acc_list = [], [], []
    tr_corr_mean_list, tr_corr_std_list = [], []

    gradient_list = []

    try:
        
        optimizer = update_optimizer_hyperparams(args, model, optimizer)

        start_time0 = time.time()
        for epoch in range(args.num_epoch+1):
            # if epoch % 1 == 0:
            #     te_losses, te_accs = [], []
            #     for batch_idx, (data, target) in enumerate(dataset[TEST]):  # data = (batch, channel=1, 28, 28)
            #         data, target = to_torch_variable(data, target, is_cuda, floatTensorF=1)
            #         _, loss, accuracy, _, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
            #         te_losses.append(loss)
            #         te_accs.append(accuracy)
            #     te_epoch.append(epoch)
            #     te_loss_list.append(np.mean(te_losses))
            #     te_acc_list.append(np.mean(te_accs))
        
                # print('Valid Epoch: %d, Loss %f Acc %f' % 
                #     (epoch, np.mean(te_losses), np.mean(te_accs)))
                

            grad_list = []
            start_time = time.time()
            for batch_idx, (data, target) in enumerate(dataset[TRAIN]):
                data, target = to_torch_variable(data, target, is_cuda)
                opt_type = args.opt_type
                if epoch > args.num_epoch * 0.1 and args.opt_type == 'sgld':
                    opt_type = args.opt_type
                else:
                    opt_type = 'sgd'
                    model, loss, accuracy, output, noise, grad_vec = feval(data, target, model, optimizer, \
                                        is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
                    tr_epoch.append(counter)
                    tr_loss_list.append(loss)
                    tr_acc_list.append(accuracy)
                    grad_list.append(grad_vec)

                if args.reset_freq > 0 and counter % args.reset_freq == 0:
                    model.reset_jacob() 

                """ meta update only uses most recent gradient on the update freq. feval resets gradient everytime its called. so meta_update will not use the sum of gradients """
                if args.oho == 1 and counter % args.update_freq == 0 and args.mlr != 0.0 and counter >= 0:
                    data_vl, target_vl = next(dataset[VALID])
                    data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)
                    model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise, is_cuda=is_cuda)
                    vl_epoch.append(counter)
                    vl_loss_list.append(loss_vl.item())
                
                
                te_losses, te_accs = [], []
                for batch_idx, (data, target) in enumerate(dataset[TEST]):  # data = (batch, channel=1, 28, 28)
                    data, target = to_torch_variable(data, target, is_cuda, floatTensorF=1)
                    _, loss, accuracy, _, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
                    te_losses.append(loss)
                    te_accs.append(accuracy)
                te_epoch.append(epoch)
                te_loss_list.append(np.mean(te_losses))
                te_acc_list.append(np.mean(te_accs))

                counter += 1  
            #grad_list = np.asarray(grad_list)   
            # corr_mean, corr_std = compute_correlation(grad_list, normF=1)
            # tr_corr_mean_list.append(corr_mean)
            # tr_corr_std_list.append(corr_std)
            grad_list = np.asarray(grad_list)

            end_time = time.time()
            # if epoch == 0: print('Single epoch timing %f' % ((end_time-start_time) / 60))

            # if epoch % args.checkpoint_freq == 0:
            #     os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
            #     save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 


            fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f, Grad Corr %f %f'
            if np.isnan(tr_loss_list[-1]):
                    break
            print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
                            np.mean(vl_loss_list[-100:]), \
                            np.mean(tr_acc_list[-100:]), \
                            str(model.eta), str(model.lambda_l2), \
                            model.dFdlr_norm, model.dFdl2_norm,\
                            model.grad_norm,  model.grad_norm_vl, \
                            model.grad_angle, model.param_norm, 0, 0))

            Wn_list.append(model.param_norm)
            dFdlr_list.append(model.dFdlr_norm)
            dFdl2_list.append(model.dFdl2_norm)
            if args.model_type == 'amlp':
                lr_list.append(model.eta.copy())
                l2_list.append(model.lambda_l2.copy())
            else:
                lr_list.append(model.eta)
                l2_list.append(model.lambda_l2)
            gang_list.append(model.grad_angle)

            gradient_list.append(model.grad_norm)
    except:
        pass

    # print(gradient_list)
    # print(len(tr_loss_list), len(vl_loss_list), len(te_loss_list))
    # assert(len(tr_loss_list) == len(vl_loss_list) == len(te_loss_list))

    Wn_list = np.asarray(Wn_list)
    l2_list = np.asarray(l2_list)
    lr_list = np.asarray(lr_list)
    dFdlr_list = np.asarray(dFdlr_list)
    dFdl2_list = np.asarray(dFdl2_list)
    tr_epoch = np.asarray(tr_epoch)
    vl_epoch = np.asarray(vl_epoch)
    te_epoch = np.asarray(te_epoch)
    tr_acc_list = np.asarray(tr_acc_list)
    te_acc_list = np.asarray(te_acc_list)
    tr_loss_list = np.asarray(tr_loss_list)
    vl_loss_list = np.asarray(vl_loss_list)
    te_loss_list = np.asarray(te_loss_list)
    gang_list = np.asarray(gang_list)
    tr_corr_mean_list = np.asarray(tr_corr_mean_list)
    tr_corr_std_list = np.asarray(tr_corr_std_list)

    return Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, \
                tr_epoch, vl_epoch, te_epoch, tr_acc_list, te_acc_list, \
                tr_loss_list, vl_loss_list, te_loss_list, tr_corr_mean_list, tr_corr_std_list


def feval(data, target, model, optimizer, mode='eval', is_cuda=0, opt_type='sgd', N=50000):

    if mode == 'eval':
        model.eval()
        with torch.no_grad():
            output = model(data)
    else:
        model.train()
        optimizer.zero_grad()
        output = model(data)
            
    # Compute Loss
    loss = F.mse_loss(output, target)
    accuracy = torch.sqrt(loss)
    # pred = output.argmax(dim=1, keepdim=True).flatten()  # get the index of the max log-probability
    # accuracy = pred.eq(target).float().mean()

    grad_vec = []
    noise = None
    if 'train' in mode:
        loss.backward()  # check how getting bacthed, try gigureout out how gradient modificat


        for i,param in enumerate(model.parameters()):
            if opt_type == 'sgld':
                noise = torch.randn(size=param.shape)
                if type(model.eta) == type(np.array([])):
                    eps = np.sqrt(model.eta[i]*2/ N) * noise  if model.eta[i] > 0 else 0 * noise
                else:
                    eps = np.sqrt(model.eta*2/ N) * noise  if model.eta > 0 else 0 * noise
                eps = to_torch_variable(eps, is_cuda=is_cuda)
                param.grad.data = param.grad.data + eps.data
            grad_vec.append(param.grad.data.cpu().numpy().flatten())

        if 'SGD_Quotient_LR' in str(optimizer):
            optimizer.mlp_step()
        else:
            optimizer.step()
        grad_vec = np.hstack(grad_vec) 
        grad_vec = grad_vec / norm_np(grad_vec)

    elif 'grad' in mode:
        loss.backward()
    
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient: {param.grad.norm().item()}")  # Norm gives a sense of gradient magnitude
            else:
                print(f"{name} has no gradient.")
        quit()

    return model, loss.item(), accuracy.item(), output, noise, grad_vec


def meta_update(args, data_vl, target_vl, data_tr, target_tr, model, optimizer, noise=None, is_cuda=1):
   
    #Compute Hessian Vector Product
    param_shapes = model.param_shapes
    dFdlr = unflatten_array(model.dFdlr, model.param_cumsum, param_shapes)
    Hv_lr  = compute_HessianVectorProd_MSE(model, dFdlr, data_tr, target_tr, is_cuda=is_cuda)

    dFdl2 = unflatten_array(model.dFdl2, model.param_cumsum, param_shapes)
    Hv_l2  = compute_HessianVectorProd_MSE(model, dFdl2, data_tr, target_tr, is_cuda=is_cuda)

    model, loss_valid, grad_valid = get_grad_valid(model, data_vl, target_vl, is_cuda)
    #model, loss_valid, grad_valid = get_grad_valid(model, data_tr, target_tr, is_cuda)

    #Compute angle between tr and vl grad
    grad = flatten_array(get_grads(model.parameters(), is_cuda)).data
    
    param = flatten_array(model.parameters())#.data.cpu().numpy()
    model.grad_norm = norm(grad)
    model.param_norm = norm(param)
    grad_vl = flatten_array(grad_valid)
    model.grad_angle = torch.dot(grad / model.grad_norm, grad_vl / model.grad_norm_vl).item()

    #Update hyper-parameters   
    model.update_dFdlr(Hv_lr, param, grad, is_cuda, noise=noise)
    model.update_eta(args.mlr, val_grad=grad_valid)
    param = flatten_array_w_0bias(model.parameters()).data
    model.update_dFdlambda_l2(Hv_l2, param, grad, is_cuda)
    model.update_lambda(args.mlr*0.01, val_grad=grad_valid)

    #Update optimizer with new eta
    optimizer = update_optimizer_hyperparams(args, model, optimizer)

    return model, loss_valid, optimizer


def get_grad_valid(model, data, target, is_cuda):

    val_model = deepcopy(model)
    val_model.train()
       
    output = val_model(data)
    loss = F.mse_loss(output, target)
    loss.backward()
    grads = get_grads(val_model.parameters(), is_cuda)
    model.grad_norm_vl = norm(flatten_array(grads))
    
    return model, loss, grads


def update_optimizer_hyperparams(args, model, optimizer):

    optimizer.param_groups[0]['lr'] = np.copy(model.eta)
    optimizer.param_groups[0]['weight_decay'] = model.lambda_l2

    return optimizer


if __name__ == '__main__':

    # args = parse_args()
    class Arg:
        pass 
    args = Arg()
    args.is_cuda = 0
    args.mlr = 0.00001
    args.lr = 0.001
    args.lambda_l2 = 0.
    args.opt_type = "sgd"
    args.update_freq = 1
    args.save = 1
    args.model_type = 'bptt'
    args.num_epoch = 500
    args.save_dir = "results"
    args.batch_size = 100
    args.reset_freq = 0 
    args.batch_size_vl = 1
    args.task = 'sin'  # oho can't solve random case
    args.t1 = 1
    args.t2 = 1
    args.outT = 9
    args.seq = 10
    args.oho = 1
    args.numTr = 1000
    args.numVl = 1000
    args.numTe = 200

    # filenames = [f'exp3/trial{i}' for i in range(24)]


    is_cuda = args.is_cuda

    # for filename in filenames:
    model = main("./test", args, ifold=ifold, is_cuda=is_cuda)

    # print(model.b_rec_, model.b_out_)

    # from matplotlib.ticker import MaxNLocator

    # def plotIO1(model):
    #     t1: int = 1
    #     t2: int = 1
    #     seq_length = 20
    #     ts = torch.arange(0, seq_length)

    #     def randomSineWaveIO():
    #         amplitude = RNG.uniform(-1, 1)  # Random amplitude between 0.5 and 2
    #         frequency = RNG.uniform(0, 20)   # Random frequency between 1 and 10 Hz
    #         phase_shift = RNG.uniform(0, 2 * np.pi)
    #         bias = RNG.uniform(-1, 1)
    #         sine_wave = lambda t: amplitude * torch.sin(frequency * t + phase_shift) + bias
    #         return sine_wave

    #     def randomSineExampleIO(t1: float, t2: float):
    #         x1 = randomSineWaveIO()
    #         x2 = randomSineWaveIO()
    #         y = createDelayedAdder(t1, t2, x1, x2)
    #         return x1, x2, y

    #     genRndomSineExampleIO = lambda: randomSineExampleIO(t1, t2)
    #     x1, x2, y = genRndomSineExampleIO()
    #     xs, ys = createExamples(ts, x1, x2, y)
    #     ys[ts < max(t1, t2)] = 0
    #     predicts = model(xs.unsqueeze(0))
    #     print(predicts.shape, ys.shape)
    #     plt.plot(ts.detach().numpy(), ys.flatten().detach().numpy(), ts.detach().numpy(), predicts.flatten().detach().numpy())
    #     plt.show()
    #     # plt.savefig('../../figs/mnist/loss_lr.png', format='png')
    
    # def plotIO2(model):

    #     t1: float = 1
    #     t2: float = 1
    #     outT: float = 9

    #     seq_length = 20
    #     ts = torch.arange(0, seq_length)

    #     def randomSineExampleIO(t1: float, t2: float):
    #         a_ = RNG.uniform(-2, 2)
    #         b_ = RNG.uniform(-2, 2)
    #         t1d = 1 #RNG.uniform(0, 2)
    #         t2d = 1 #RNG.uniform(0, 2)
    #         x1, x2, y = createAddMemoryTask(t1, t2, a_, b_, t1d, t2d, outT)
    #         return x1, x2, y


    #     genRndomSineExampleIO = lambda: randomSineExampleIO(t1, t2)
    #     x1, x2, y = genRndomSineExampleIO()
    #     xs, ys = createExamples(ts, x1, x2, y)
    #     ys[ts < max(t1, t2)] = 0
    #     # print(xs)
    #     # print(ys)
    #     predicts = model(xs.unsqueeze(0).permute(1,0,2))
    #     predicts = predicts.permute(1,0,2)
    #     # print(predicts)
    #     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    #     plt.plot(ts.detach().numpy(), ys.flatten().detach().numpy(), ts.detach().numpy(), predicts.flatten().detach().numpy(), marker='o')
    #     # plt.savefig('../../figs/mnist/loss_lr.png', format='png')
    #     plt.show()
    
    # def plotIO3(model):

    #     t1: float = 1
    #     t2: float = 1

    #     seq_length = 20
    #     ts = torch.arange(0, seq_length)

    #     def generate_random_lists(ts):
    #         #` Generate random lists for x1(t) and x2(t)
    #         T = len(ts)
    #         x1 = RNG.randn(T)  # Random values for x1
    #         x2 = RNG.randn(T)  # Random values for x2
    #         x1 = torch.from_numpy(x1).to(torch.float32)
    #         x2 = torch.from_numpy(x2).to(torch.float32)
            
    #         # Initialize y(t) with zeros
    #         y = torch.zeros(T).to(torch.float32)
            
    #         # Calculate y(t) = x1(t - t1) + x2(t - t2)
    #         for t in ts:
    #             if t >= max(t1, t2):
    #                 y[t] = x1[t - t1] + x2[t - t2]
            
    #         return torch.stack([x1, x2], dim=1), y

    #     xs, ys = generate_random_lists(ts)
    #     predicts = model(xs.unsqueeze(0))
    #     plt.plot(ts.detach().numpy(), ys.flatten().detach().numpy(), ts.detach().numpy(), predicts.flatten().detach().numpy(), marker='o')
    #     # plt.savefig('../../figs/mnist/loss_lr.png', format='png')
    #     plt.show()

    # plotIO2(model)

#     for i,param in enumerate(model.parameters()):
#         print(param)

# T = TypeVar('T') 
# X = TypeVar('X')
# Y = TypeVar('Y')

# # def ohoUpdate(state: T, modelOutput: X) -> T:
# #     hyperparameters, influenceMatrix = state
# #     data_vl, target_vl, gradFn = modelOutput  # gradFn should already be curried with "data_tr, target_tr"
# #     hessian(gradFn, )

# P = TypeVar('Parameter') 
# Z = TypeVar('Z')
# E = TypeVar('E')

# def getModelLossFn(lossFn: Callable[[Y, Y], Z]
#                     , model: Callable[[X, P], Y]
#                     , input_tr, target_tr):  # just need to map this on input stream
#     outFn = compose(lossFn(target_tr), model(input_tr))
#     return outFn


# # jacrev(outFn)
# def updateParam(paramUpdateStrategy, state, jacobianFn):
#     hp, p = state
#     jacobian = jacobianFn(p)
#     p_ = paramUpdateStrategy(p, hp, jacobian)
#     return p_


# def main():
#     dataset = datasets.MNIST('data/mnist', train=True, download=True,
#                                         transform=transforms.Compose(
#                                                 [transforms.ToTensor()]))
#     train_set, valid_set = torch.utils.data.random_split(dataset,[60000 - 10000, 10000])


#     data_loader_tr = DataLoader(train_set, batch_size=100, shuffle=True)
#     data_loader_vl = DataLoader(valid_set, batch_size=100, shuffle=True)
#     data_loader_te = DataLoader(datasets.MNIST('data/mnist', train=False, download=True,
#                                                         transform=transforms.Compose(
#                                                                 [transforms.ToTensor()])),
#                                                 batch_size=100, shuffle=True)
#     data_loader_vl = cycle_efficient(data_loader_vl)

#     W_rec_, W_in_, b_rec_, W_out_, b_out_ = initializeParametersIO(n_in, n_h, n_out)
#     self.rnn = compose(drop(1)
#                     ,  scan(rnnTransition(W_in_, W_rec_, b_rec_, f.relu, 1)))
#     # self.fc = linear_(self.W_out_,self.b_out_)
#     self.initH = lambda x: torch.zeros(x.size(1), n_h).to('cpu' if is_cuda==0 else 'gpu') 

#     datastream = zip(data_loader_tr, data_loader_vl)
#     getModelLossFn(F.mse_loss, )


# def learn(config, getLossFn: Callable[[X, Y], Callable[[P], Z]], getModel: Callable[[T, Callable[[P], E]], P], datastream):
#     # getModel is getModelLossFn(f, g)
#     # getLossFn = updateParam(h)

#     scan()

#     """
#     learning rate list
#     lambda list
#     influence matrix list
#     parameter norm = Wn_list
#     gradient angle list = gang_list
#     training epoch list?
#     training loss list
#     training prediction/rmse list?
#     validation epoch list?
#     validation loss list
#     validation pred
#     same three for test as well 
#     idk what compute correlation is for    
#     """
    

#     for epoch in range(args.num_epoch+1):
#         # if epoch % 100 == 0:
#             # te_losses, te_accs = [], []
#             # for batch_idx, (data, target) in enumerate(dataset[TEST]):  # data = (batch, channel=1, 28, 28)
#             #     data, target = to_torch_variable(data, target, is_cuda, floatTensorF=1)
#             #     data = data.permute(1, 0, 2)
#             #     target = target.permute(1, 0, 2)
#             #     _, loss, accuracy, _, _, _ = feval(data, target, model, optimizer, mode='eval', is_cuda=is_cuda)
#             #     te_losses.append(loss)
#             #     te_accs.append(accuracy)
#             # te_epoch.append(epoch)
#             # te_loss_list.append(np.mean(te_losses))
#             # te_acc_list.append(np.mean(te_accs))
    
#             # print('Valid Epoch: %d, Loss %f Acc %f' % 
#             #     (epoch, np.mean(te_losses), np.mean(te_accs)))
            
#         for batch_idx, (data, target) in enumerate(dataset[TRAIN]):
#             opt_type = 'sgd'
#             model, loss, accuracy, output, noise, grad_vec = feval(data, target, model, optimizer, \
#                                 is_cuda=is_cuda, mode='meta-train', opt_type=opt_type)
#             tr_epoch.append(counter)
#             tr_loss_list.append(loss)
#             tr_acc_list.append(accuracy)
#             grad_list.append(grad_vec)

#             if args.reset_freq > 0 and counter % args.reset_freq == 0:
#                 model.reset_jacob() 

#             """ meta update only uses most recent gradient on the update freq. feval resets gradient everytime its called. so meta_update will not use the sum of gradients """
#             if args.oho == 1 and counter % args.update_freq == 0 and args.mlr != 0.0:
#                 data_vl, target_vl = next(dataset[VALID])
#                 data_vl = data_vl.permute(1, 0, 2)
#                 target_vl = target_vl.permute(1, 0, 2)
#                 data_vl, target_vl = to_torch_variable(data_vl, target_vl, is_cuda)
#                 model, loss_vl, optimizer = meta_update(args, data_vl, target_vl, data, target, model, optimizer, noise, is_cuda=is_cuda)
#                 vl_epoch.append(counter)
#                 vl_loss_list.append(loss_vl.item())

#             counter += 1  
#         #grad_list = np.asarray(grad_list)   
#         corr_mean, corr_std = compute_correlation(grad_list, normF=1)
#         tr_corr_mean_list.append(corr_mean)
#         tr_corr_std_list.append(corr_std)
#         grad_list = np.asarray(grad_list)

#         end_time = time.time()
#         if epoch == 0: print('Single epoch timing %f' % ((end_time-start_time) / 60))

#         # if epoch % args.checkpoint_freq == 0:
#         #     os.makedirs(args.fdir+ '/checkpoint/', exist_ok=True)
#         #     save(model, args.fdir+ '/checkpoint/epoch%d' % epoch) 


#         fprint = 'Train Epoch: %d, Tr Loss %f Vl loss %f Acc %f Eta %s, L2 %s, |dFdlr| %.2f |dFdl2| %.2f |G| %.4f |G_vl| %.4f Gang %.3f |W| %.2f, Grad Corr %f %f'
#         if np.isnan(np.mean(tr_loss_list[-100:])):
#                 print('STOP')
#                 print(tr_loss_list[-100:])
#                 return
#         print(fprint % (epoch, np.mean(tr_loss_list[-100:]), \
#                         np.mean(vl_loss_list[-100:]), \
#                         np.mean(tr_acc_list[-100:]), \
#                         str(model.eta), str(model.lambda_l2), \
#                         model.dFdlr_norm, model.dFdl2_norm,\
#                         model.grad_norm,  model.grad_norm_vl, \
#                         model.grad_angle, model.param_norm, corr_mean, corr_std))

#         Wn_list.append(model.param_norm)
#         dFdlr_list.append(model.dFdlr_norm)
#         dFdl2_list.append(model.dFdl2_norm)
#         if args.model_type == 'amlp':
#             lr_list.append(model.eta.copy())
#             l2_list.append(model.lambda_l2.copy())
#         else:
#             lr_list.append(model.eta)
#             l2_list.append(model.lambda_l2)
#         gang_list.append(model.grad_angle)

#     Wn_list = np.asarray(Wn_list)
#     l2_list = np.asarray(l2_list)
#     lr_list = np.asarray(lr_list)
#     dFdlr_list = np.asarray(dFdlr_list)
#     dFdl2_list = np.asarray(dFdl2_list)
#     tr_epoch = np.asarray(tr_epoch)
#     vl_epoch = np.asarray(vl_epoch)
#     te_epoch = np.asarray(te_epoch)
#     tr_acc_list = np.asarray(tr_acc_list)
#     te_acc_list = np.asarray(te_acc_list)
#     tr_loss_list = np.asarray(tr_loss_list)
#     vl_loss_list = np.asarray(vl_loss_list)
#     te_loss_list = np.asarray(te_loss_list)
#     gang_list = np.asarray(gang_list)
#     tr_corr_mean_list = np.asarray(tr_corr_mean_list)
#     tr_corr_std_list = np.asarray(tr_corr_std_list)

#     return Wn_list, l2_list, lr_list, dFdlr_list, dFdl2_list, gang_list, \
#                 tr_epoch, vl_epoch, te_epoch, tr_acc_list, te_acc_list, \
#                 tr_loss_list, vl_loss_list, te_loss_list, tr_corr_mean_list, tr_corr_std_list




# """
# jac :: data -> theta -> theta 
# if I have
# jac data_tr, then I can compute 

# f :: data -> weights -> output
# f' = f data_tr
# then I can call j = jac(f) to create the jac
# and do
# f(model)
# or f(model + r*v)


# """