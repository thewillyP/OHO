from objectalgebra import *
from typing import Callable, TypeVar, Iterator, Union
from line_profiler import profile

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

# def learnStep(t: Union[HasParameter[ENV, P], HasLoss[ENV, L]]
#             , prediction: Callable[[X, ENV], tuple[ENV, T]]
#             , lossT: Callable[[Union[HasLoss[ENV, L]]], Callable[[Y, tuple[ENV, T]], ENV]]
#             , paramT: Callable[[Union[HasParameter[ENV, P], HasLoss[ENV, L]]], Callable[[ENV], ENV]]) -> Callable[[tuple[X, Y], ENV], ENV]:
#     lossStep = lossT(t)
#     paramStep = paramT(t)
#     lossFn = fuse(prediction, lossStep)
#     return fmapSuffix(paramStep, lossFn)

def learnStep(prediction: Callable[[Union[HasParameter[ENV, P], HasActivation[ENV, A]]], Callable[[X, ENV], tuple[ENV, T]]]
            , lossT: Callable[[Union[HasLoss[ENV, L]]], Callable[[Y, tuple[ENV, T]], ENV]]
            , paramT: Callable[[Union[HasParameter[ENV, P], HasLoss[ENV, L]]], Callable[[ENV], ENV]]) -> Callable[[Union[HasParameter[ENV, P], HasLoss[ENV, L]]], Callable[[tuple[X, Y], ENV], ENV]]:
    lossStep = liftA2(fuse, prediction, lossT)
    return liftA2(fmapSuffix, paramT, lossStep)





def resetRnnActivation(t: Union[HasActivation[ENV, A]]
                , resetee:  Callable[[X, ENV], B]
                , actv0: A) -> Callable[[X, ENV], B]:
    def reset_(env: ENV) -> ENV:
        return t.putActivation(actv0, env)
    return fmapPrefix(reset_, resetee)

def resetRnnActivation(t: Union[HasActivation[ENV, A]]) -> Callable[[Callable[[X, ENV], B], A], Callable[[X, ENV], B]]:
    def resetRnnActivation_(resetee: Callable[[X, ENV], B], actv0: A) -> Callable[[X, ENV], B]:
        def reset_(env: ENV) -> ENV:
            return t.putActivation(actv0, env)
        return fmapPrefix(reset_, resetee)
    return resetRnnActivation_


# lambda t: lambda env: t.putActivation(actv0, env)

# def resetThis(resetter: Callable[[ENV], ENV], resetee: Callable[]) -> Callable[[X, ENV], B]:
#     return fmapPrefix(resetter, resetee)

# def resetLoss(t: Union[HasLoss[ENV, L]]  # TODO: generalize this
#                 , resetee:  Callable[[X, ENV], T]
#                 , loss0: A) -> Callable[[X, ENV], T]:
#     def reset_(env: ENV) -> ENV:
#         return t.putLoss(loss0, env)
#     return fmapPrefix(reset_, resetee)

# rewrite resetLoss in same form as repeatRnnWithReset
def resetLoss(t: Union[HasLoss[ENV, L]]) -> Callable[[Callable[[X, ENV], T], L], Callable[[X, ENV], T]]:
    def resetLoss_(resetee: Callable[[X, ENV], T], loss0: L) -> Callable[[X, ENV], T]:
        def resetLoss__(env: ENV) -> ENV:
            return t.putLoss(loss0, env)
        return fmapPrefix(resetLoss__, resetee)
    return resetLoss_


# def repeatRnnWithReset(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]
#                 , repeatee: Callable[[X, ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
#     def repeat_(xs: Iterator[X], env: ENV) -> ENV:
#         a0 = t.getActivation(env)
#         l0 = t.getLoss(env)
#         resetter = resetRnnActivation(t, repeatee, a0)
#         resetter = resetLoss(t, resetter, l0)
#         return foldr(resetter)(xs, env)
#     return repeat_

@profile
def repeatRnnWithReset(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]) -> Callable[[Callable[[X, ENV], ENV]], Callable[[Iterator[X], ENV], ENV]]:
    def repeat_(repeatee: Callable[[X, ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
        def repeat__(xs: Iterator[X], env: ENV) -> ENV:
            a0 = t.getActivation(env)
            l0 = t.getLoss(env)
            resetter = resetRnnActivation(t)(repeatee, a0)
            resetter = resetLoss(t)(resetter, l0)
            env = t.putPrediction(0, env)
            return foldr(resetter)(xs, env)
        return repeat__
    return repeat_


# def repeatRnnWithReset(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]) -> Callable[[Callable[[X, ENV], ENV]], Callable[[Iterator[X], ENV], ENV]]:
#     def repeat_(repeatee: Callable[[X, ENV], ENV]
#                 , resetters: list[Callable[[]]]) -> Callable[[Iterator[X], ENV], ENV]:
#         def repeat__(xs: Iterator[X], env: ENV) -> ENV:
#             a0 = t.getActivation(env)
#             l0 = t.getLoss(env)
#             resetter = resetRnnActivation(t)(repeatee, a0)
#             resetter = resetLoss(t)(resetter, l0)
#             return foldr(resetter)(xs, env)
#         return repeat__
#     return repeat_

# to add oho, literally just call fuse with the oho function 




PARAM =  tuple[torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , float]

MODEL = VanillaRnnState[torch.Tensor, torch.Tensor, PARAM, torch.Tensor]

@profile
def rnnTrans(activation: Callable, x: torch.Tensor, param: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], h) -> torch.Tensor:
    W_rec, W_in, b_rec, alpha = param
    return (1-alpha ) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))

@profile
def activationTrans(activationFn: Callable[[torch.Tensor], torch.Tensor]):
    def activationTrans_(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
        def activationTrans__(x: torch.Tensor, env: MODEL) -> MODEL:
            a = t.getActivation(env)
            W_rec, W_in, b_rec, _, _, alpha = t.getParameter(env)
            a_ = rnnTrans(activationFn, x, (W_rec, W_in, b_rec, alpha), a)
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
@profile
def predictTrans(t: Union[
    HasActivation[MODEL, torch.Tensor]
    , HasParameter[MODEL, PARAM]
    , HasPrediction[MODEL, torch.Tensor]]) -> Callable[[MODEL], MODEL]:
    def predictTrans_(env: MODEL) -> MODEL:
        a = t.getActivation(env)
        _, _, _, W_out, b_out, _ = t.getParameter(env)
        pred = f.linear(a, W_out, b_out)
        return t.putPrediction(pred, env)
    return predictTrans_
@profile
def lossTrans(criterion: Callable):
    def lossTrans_(t: Union[HasLoss[MODEL, torch.Tensor], HasPrediction[MODEL, torch.Tensor]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
        def lossTrans__(y: torch.Tensor, env: MODEL) -> MODEL:
            prediction = t.getPrediction(env)
            loss = criterion(prediction, y) + t.getLoss(env)
            return t.putLoss(loss, env)
        return lossTrans__
    return lossTrans_

step = 0
@profile
def parameterTrans(opt):
    def parameterTrans_(t: Union[HasParameter[MODEL, PARAM], HasLoss[MODEL, torch.Tensor]]) -> Callable[[MODEL], MODEL]:
        def parameterTrans__(env: MODEL) -> MODEL:
            global step
            opt.zero_grad() 
            loss = t.getLoss(env)
            loss.backward()
            opt.step()  # will actuall physically spooky mutate the param so no update needed. 
            step += 1
            if step % 100 == 0:
                print(f'Step [{step}] Loss: {loss.item()}')
            return env
        return parameterTrans__
    return parameterTrans_

@profile
def getRnn(t: VanillaRnnStateInterpreter, activationFn) -> Callable[[Iterator[torch.Tensor], MODEL], tuple[MODEL, torch.Tensor]]:
    return offlineRnnPredict(t, activationTrans(activationFn), predictTrans)

# def trainRnn(t: VanillaRnnStateInterpreter
#                 , optimizer
#                 , rnn: Callable[[X, MODEL], tuple[MODEL, torch.Tensor]]
#                 , criterion: Callable) -> Callable[[Iterator[tuple[X, torch.Tensor]], MODEL], MODEL]:
#     learn = learnStep(t, rnn, lossTrans(criterion), parameterTrans(optimizer))
#     return repeatRnnWithReset(t, learn)


def trainRnn( optimizer
            , rnn: Callable[[VanillaRnnStateInterpreter], Callable[[X, MODEL], tuple[MODEL, torch.Tensor]]]
            , criterion: Callable) -> Callable[[VanillaRnnStateInterpreter], Callable[[Iterator[tuple[X, torch.Tensor]], MODEL], MODEL]]:
    learn = learnStep(rnn, lossTrans(criterion), parameterTrans(optimizer))
    return compose2(repeatRnnWithReset, lambda repeat: repeat(learn))

def pyTorchRnn(t: Union[HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], tuple[MODEL, torch.Tensor]]:
    def predictTrans_(x: torch.Tensor, env: MODEL) -> tuple[MODEL, torch.Tensor]:
        model = t.getParameter(env)
        return env, model(x)
    return predictTrans_

def oho(t: Union[HasParameter[MODEL, PARAM], HasHyperParameter[MODEL, HP]]) -> Callable[[MODEL], MODEL]:
    def oho_(env: MODEL) -> MODEL:
        return t.putHyperParameter(t.getHyperParameter(env), env)
    return oho_

