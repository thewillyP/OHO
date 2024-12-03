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


# rewrite resetLoss in same form as repeatRnnWithReset
def resetLoss(t: Union[HasLoss[ENV, L]]) -> Callable[[Callable[[X, ENV], T], L], Callable[[X, ENV], T]]:
    def resetLoss_(resetee: Callable[[X, ENV], T], loss0: L) -> Callable[[X, ENV], T]:
        def resetLoss__(env: ENV) -> ENV:
            return t.putLoss(loss0, env)
        return fmapPrefix(resetLoss__, resetee)
    return resetLoss_

def repeatRnnWithReset(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]) -> Callable[[Callable[[X, ENV], ENV]], Callable[[Iterator[X], ENV], ENV]]:
    def repeat_(repeatee: Callable[[X, ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
        def repeat__(xs: Iterator[X], env: ENV) -> ENV:
            a0 = t.getActivation(env)
            l0 = t.getLoss(env)
            resetter = resetRnnActivation(t)(repeatee, a0)
            resetter = resetLoss(t)(resetter, l0)
            return foldr(resetter)(xs, env)
        return repeat__
    return repeat_


def repeatRnnWithReset_(t: Union[HasActivation[ENV, A], HasParameter[ENV, P], HasLoss[ENV, L]]) -> Callable[[Callable[[X, ENV], ENV]], Callable[[Iterator[X], ENV], ENV]]:
    def repeat_(repeatee: Callable[[X, ENV], ENV]) -> Callable[[Iterator[X], ENV], ENV]:
        def repeat__(xs: Iterator[X], env: ENV) -> ENV:
            a0 = t.getActivation(env)
            resetter = resetRnnActivation(t)(repeatee, a0)
            return foldr(resetter)(xs, env)
        return repeat__
    return repeat_


PARAM =  tuple[torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , torch.Tensor
            , float]

MODEL = VanillaRnnState[torch.Tensor, torch.Tensor, PARAM]

def rnnTrans(activation: Callable, x: torch.Tensor, param: tuple[torch.Tensor, torch.Tensor, torch.Tensor, float], h) -> torch.Tensor:
    W_rec, W_in, b_rec, alpha = param
    return (1-alpha ) * h + alpha * activation(f.linear(x, W_in, None) + f.linear(h, W_rec, b_rec))

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

# def predictTrans(t: Union[HasActivation[MODEL, torch.Tensor], HasParameter[MODEL, PARAM]]) -> Callable[[MODEL], tuple[MODEL, torch.Tensor]]:
#     def predictTrans_(env: MODEL) -> tuple[MODEL, torch.Tensor]:
#         a = t.getActivation(env)
#         _, _, _, W_out, b_out, _ = t.getParameter(env)
#         return env, f.linear(a, W_out, b_out)
#     return predictTrans_


# def lossTrans(criterion: Callable):
#     def lossTrans_(t: Union[HasLoss[MODEL, torch.Tensor]]) -> Callable[[torch.Tensor, tuple[MODEL, torch.Tensor]], MODEL]:
#         def lossTrans__(y: torch.Tensor, pair: tuple[MODEL, torch.Tensor]) -> MODEL:
#             env, prediction = pair
#             loss = criterion(prediction, y) + t.getLoss(env)
#             return t.putLoss(loss, env)
#         return lossTrans__
#     return lossTrans_


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

def lossTrans(criterion: Callable):
    def lossTrans_(t: Union[HasLoss[MODEL, torch.Tensor], HasPrediction[MODEL, torch.Tensor]]) -> Callable[[torch.Tensor, MODEL], MODEL]:
        def lossTrans__(y: torch.Tensor, env: MODEL) -> MODEL:
            prediction = t.getPrediction(env)
            loss = criterion(prediction, y) + t.getLoss(env)
            return t.putLoss(loss, env)
        return lossTrans__
    return lossTrans_


def pyTorchRnn(t: Union[HasParameter[MODEL, PARAM]]) -> Callable[[torch.Tensor, MODEL], tuple[MODEL, torch.Tensor]]:
    def predictTrans_(x: torch.Tensor, env: MODEL) -> tuple[MODEL, torch.Tensor]:
        model = t.getParameter(env)
        return env, model(x)
    return predictTrans_

def oho(t: Union[HasParameter[MODEL, PARAM], HasHyperParameter[MODEL, HP]]) -> Callable[[MODEL], MODEL]:
    def oho_(env: MODEL) -> MODEL:
        return t.putHyperParameter(t.getHyperParameter(env), env)
    return oho_

