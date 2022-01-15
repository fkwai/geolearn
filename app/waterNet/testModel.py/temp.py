
import torch
import torch.nn.functional as F

nh = 16
ns = 5
p = torch.randn(10, ns, nh*12).cuda()

actLst = ['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid',
          'exp', 'relu', 'relu', 'relu',
          'hardsigmoid', 'hardsigmoid', 'exp', 'skip']
outLst = list()
for k, act in enumerate(actLst):
    if act == 'skip':
        outLst.append(p[..., nh*k:nh*(k+1)])
    else:
        if hasattr(torch, act):
            ff = getattr(torch, act)
        elif hasattr(F, act):
            ff = getattr(F, act)
        else:
            Exception('can not find activate func')
        outLst.append(ff(p[..., nh*k:nh*(k+1)]))
