import torch

################
# Select device:
#   - CPU: "cpu"
#   - GPU: "gpu"
################
DEV_SEL = "cpu"

if torch.cuda.is_available() and DEV_SEL is not "cpu":
    device = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
elif DEV_SEL == "cpu" or torch.cuda.is_available() is False:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
