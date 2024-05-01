import torch
import torch.nn as nn
import torch.distributed as dist    
import accelerate

# class Net(nn.Module):
#     def forward(self, x):
#         return x + 1
    
# net = Net()
# net = nn.DataParallel(net)

# print(net(torch.tensor([[1.0], [2.0], [3.0], [4.0]])))
# torch.distributed.init_process_group(backend='nccl')
# x = torch.ones(2, 2, requires_grad=True)
# x = torch.distributed.all_reduce(x, 0)
# print(x)
from torch.distributed.elastic.multiprocessing import errors

accelerator = accelerate.Accelerator()

@errors.record
def main():
    print(accelerator.device)
    x = torch.ones(2, requires_grad=True).to(accelerator.device)
    print("hello, world")
    print(x)
    # y  = torch.ones(2, requires_grad=True).to('cuda:1')
    # print(y)

    x = accelerator.reduce(x, reduction='sum')
    print(x)

try:
    main()
except Exception as e:
    if accelerator.is_main_process:
        breakpoint()
finally:
    accelerator.wait_for_everyone()