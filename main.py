import os
import torch
import time

def test_pytorch():
    print("Testing PyTorch GPU capabilities...")
    if torch.cuda.is_available():
        device = torch.device("cuda") #"cpu")
        cpu_count = os.cpu_count()
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
      
        n_gpu = torch.cuda.device_count()
        print(f"number of cuda gpus is {n_gpu}, cpus {cpu_count}")

        for _ in range(0, 10):
            start_time = time.time()

            size = 100000000
            et = torch.rand(size, device=device)
            #print(et)
            ew = torch.fft.fft(et)
            #print(ew)

            elapsed_time = time.time() - start_time
            print(f"PyTorch fft time: {elapsed_time:.8f} seconds")
    else:
        print("CUDA is not available on this system.")

if __name__ == "__main__":
    test_pytorch()
