import torch
import torch.nn as nn

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss(reduction='none')
loss_reduction = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = torch.randn(2, 3, requires_grad=True)
print(input)
# each element in target has to have 0 <= value < C
target = torch.tensor([1, 2])
print(m(input))
output = loss(m(input), target)
print(output)
output = loss_reduction(m(input), target)
print(output)
