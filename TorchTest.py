"""
main.py: calls different functions of yolo
@author:  Bernhard Pandion
"""
import torch
def main():
    print("Starting TORCH/CUDA test")
    print("########################")
    # print(torch.cuda)
    x = torch.rand(5, 3)
    print(x)
    print("Cuda is available: "+ str(torch.cuda.is_available()))
    for i in range(torch.cuda.device_count()):
        print("Torch-name:"+torch.cuda.get_device_properties(i).name)
    

if __name__ == "__main__":
    main()
