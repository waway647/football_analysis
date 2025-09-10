#check cuda if available
import torch
def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    return device
if __name__ == "__main__":
    check_cuda()