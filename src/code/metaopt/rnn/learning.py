from objectalgebra import *
from typing import Callable, TypeVar, Iterator, Union


T = TypeVar('T')
E = TypeVar('E')
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')
D = TypeVar('D')
X = TypeVar('X')
Y = TypeVar('Y')
H = TypeVar('H')
P = TypeVar('P')
L = TypeVar('L')
HP = TypeVar('HP')

ENV = TypeVar('ENV')

def offlineRnnPredict(
        t: Union[HasActivation[ENV, A], HasParameter[ENV, P]]
        , actvT: Callable[[Union[HasActivation[ENV, A], HasParameter[ENV, P]]], Callable[[X, ENV], ENV]]
        , predictT: Callable[[Union[HasParameter[ENV, P], HasActivation[A, T]]], Callable[[ENV], tuple[ENV, T]]]) -> Callable[[Iterator[X], ENV], tuple[ENV, T]]:
    actvStep = actvT(t)
    predictStep = predictT(t)
    offline = foldr(actvStep)
    return fmapSuffix(predictStep, offline)

def learnStep(t: Union[HasParameter[ENV, P], HasLoss[ENV, L]]
            , prediction: Callable[[X, ENV], tuple[ENV, T]]
            , lossT: Callable[[Union[HasLoss[ENV, L]]], Callable[[Y, tuple[ENV, T]], ENV]]
            , paramT: Callable[[Union[HasParameter[ENV, P], HasLoss[ENV, L]]], Callable[[ENV], ENV]]) -> Callable[[tuple[X, Y], ENV], ENV]:
    lossStep = lossT(t)
    paramStep = paramT(t)
    lossFn = fuse(prediction, lossStep)
    return fmapSuffix(paramStep, lossFn)

def resetRnnActivation(t: Union[HasActivation[ENV, A]]
                , resetee:  Callable[[X, ENV], ENV]
                , actv0: A) -> Callable[[X, ENV], ENV]:
    def reset_(env: ENV) -> ENV:
        return t.putActivation(actv0, env)
    return fmapPrefix(reset_, resetee)

def resetLoss(t: Union[HasLoss[ENV, L]]  # TODO: generalize this
                , resetee:  Callable[[X, ENV], T]
                , loss0: A) -> Callable[[X, ENV], T]:
    def reset_(env: ENV) -> ENV:
        return t.putLoss(loss0, env)
    return fmapPrefix(reset_, resetee)


def repeatRnnWithReset(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]
                , repeatee: Callable[[X, ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
    def repeat_(xs: Iterator[X], env: ENV) -> ENV:
        a0 = t.getActivation(env)
        l0 = t.getLoss(env)
        resetter = resetRnnActivation(t, repeatee, a0)
        resetter = resetLoss(t, resetter, l0)
        return foldr(resetter)(xs, env)
    return repeat_





PARAM =  tuple[torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , float]

MODEL = VanillaRnnState[torch.Tensor, torch.Tensor, PARAM]


def rnnTrans(activation: Callable) -> Callable[[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor], torch.Tensor]:
    def rnnTrans_(x: torch.Tensor, param: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], h: torch.Tensor) -> torch.Tensor:
        W_rec, W_in, b_rec, alpha = param
        return (1-alpha ) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))
    return rnnTrans_
# (W_rec_, W_in_, b_rec_, W_out_, b_out_, alpha_))
def activationTrans(activationFn: Callable[[torch.Tensor], torch.Tensor]):
    def activationTrans_(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
        def activationTrans__(x: torch.Tensor, env: MODEL) -> MODEL:
            a = t.getActivation(env)
            W_rec, W_in, b_rec, _, _, alpha = t.getParameter(env)

            # norms = []
            # for param in t.getParameter(env)[:-1]:
            #     norms.append(param.norm().item())  # Get the norm as a scalar
            # avg_param_norm = sum(norms) / len(norms)
            # print(avg_param_norm)

            a_ = rnnTrans(activationFn)(x, (W_rec, W_in, b_rec, alpha), a)
            return t.putActivation(a_, env)
        return activationTrans__
    return activationTrans_


# def activationLayersTrans(activationFn: Callable[[torch.Tensor], torch.Tensor]):
#     def activationTrans_(t: Union[HasActivation[MODEL, List[torch.Tensor]], HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
#         def activationTrans__(x: torch.Tensor, env: MODEL) -> MODEL:
#             as_ = t.getActivation(env)
#             W_rec, W_in, b_rec, _, _, alpha = t.getParameter(env)
#             def scanner(prevActv: torch.Tensor, nextActv: torch.Tensor) -> torch.Tensor:  # i'm folding over nextActv
#                 return rnnTrans(activationFn)(prevActv, (W_in, W_rec, b_rec, alpha), nextActv)
#             as__ = list(scan0(scanner, x, as_))
#             return t.putActivation(as__, env)
#         return activationTrans__
#     return activationTrans_



# doing multiple layers is just a fold over it

def predictTrans(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[MODEL], tuple[MODEL, torch.Tensor]]:
    def predictTrans_(env: MODEL) -> tuple[MODEL, torch.Tensor]:
        a = t.getActivation(env)
        _, _, _, W_out, b_out, _ = t.getParameter(env)
        return env, f.linear(a, W_out, b_out)
    return predictTrans_

def lossTrans(t: Union[HasLoss[MODEL, torch.Tensor]]) -> Callable[[torch.Tensor, tuple[MODEL, torch.Tensor]], MODEL]:
    def lossTrans_(y: torch.Tensor, pair: tuple[MODEL, torch.Tensor]) -> MODEL:
        env, prediction = pair
        loss = f.cross_entropy(prediction, y) + t.getLoss(env)
        return t.putLoss(loss, env)
    return lossTrans_

step = 0
losses = []
param_norms = []
def parameterTrans(opt):
    def parameterTrans_(t: Union[HasParameter[MODEL, PARAM], HasLoss[MODEL, torch.Tensor]]) -> Callable[[MODEL], MODEL]:
        def parameterTrans__(env: MODEL) -> MODEL:
            global step, losses, param_norms
            opt.zero_grad() 
            loss = t.getLoss(env)

            # make_dot(loss, params=dict(t.getParameter(env).named_parameters())).render("working", format="png")
            loss.backward()
            opt.step()  # will actuall physically spooky mutate the param so no update needed. 
            if (step+1) % 100 == 0:
                # make_dot(loss, params=dict({i: a for i, a in enumerate(t.getParameter(env)[:-1])})).render("graph1", format="png")
                # quit()
                print (f'Step [{step+1}], Loss: {loss.item():.4f}')
            losses.append(loss.item())
            step += 1

            # norms = []
            # for param in t.getParameter(env)[:-1]:
            #     norms.append(param.norm().item())  # Get the norm as a scalar
            # avg_param_norm = sum(norms) / len(norms)
            # param_norms.append(avg_param_norm)
            return env
        return parameterTrans__
    return parameterTrans_

def getRnn(t: VanillaRnnStateInterpreter, activationFn) -> Callable[[Iterator[torch.Tensor], MODEL], tuple[MODEL, torch.Tensor]]:
    return offlineRnnPredict(t, activationTrans(activationFn), predictTrans)

def trainRnn(t: VanillaRnnStateInterpreter
                , optimizer
                , rnn: Callable[[X, MODEL], tuple[MODEL, torch.Tensor]]) -> Callable[[Iterator[tuple[X, torch.Tensor]], MODEL], MODEL]:
    learn = learnStep(t, rnn, lossTrans, parameterTrans(optimizer))
    return repeatRnnWithReset(t, learn)


def pyTorchRnn(t: Union[HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], tuple[MODEL, torch.Tensor]]:
    def predictTrans_(x: torch.Tensor, env: MODEL) -> tuple[MODEL, torch.Tensor]:
        model = t.getParameter(env)
        return env, model(x)
    return predictTrans_