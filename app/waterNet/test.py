import torch

v = torch.tensor([0., 0., 0.], requires_grad=True)
h = v.register_hook(lambda x: print('grad', x))  # double the gradient
v.backward(torch.tensor([1., 2., 3.]))
v.grad
