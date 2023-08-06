"""GPU utilities"""
import torch

def select_device():
    """Select leats busy GPU or CPU if no GPU available"""
    if torch.cuda.is_available():
        selected_gpu_memory = 1e8
        for n_gpu in range(torch.cuda.device_count()):
            current_gpu_memory = torch.cuda.memory_usage(n_gpu)
            print(f"GPU {n_gpu} MEM {current_gpu_memory}")
            if current_gpu_memory < selected_gpu_memory:
                selected_gpu_memory = current_gpu_memory
                selected_gpu = n_gpu
        print(f"Selected GPU number {selected_gpu}")
        return torch.device("cuda", selected_gpu)
    else:
        return torch.device("cpu")
