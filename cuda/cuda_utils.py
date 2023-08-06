"""GPU utilities"""
import torch

def select_device():
    """Select leats busy GPU or CPU if no GPU available"""
    if torch.cuda.is_available():
        selected_gpu_memory = 1e8
        for n_gpu in range(torch.cuda.device_count()):
            try:
                current_gpu_memory = torch.cuda.mem_get_info(n_gpu)[0]
                print(f"GPU {n_gpu} MEM {current_gpu_memory}")
                if current_gpu_memory < selected_gpu_memory:
                    selected_gpu_memory = current_gpu_memory
                    selected_gpu = n_gpu
            except RuntimeError as e:
                print(f"CUDA:{n_gpu} returned {str(e)}")
        print(f"Selected GPU number {selected_gpu}")
        return torch.device("cuda", selected_gpu)
    else:
        return torch.device("cpu")
