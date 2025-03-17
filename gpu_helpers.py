import torch
import gc

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        # Get the current device
        device = torch.cuda.current_device()
        
        # Get allocated memory in bytes
        allocated = torch.cuda.memory_allocated(device)
        
        # Get cached memory in bytes (allocated + cached = total reserved)
        reserved = torch.cuda.memory_reserved(device)
        
        # Convert to more readable format (MB)
        allocated_mb = allocated / 1024 / 1024
        reserved_mb = reserved / 1024 / 1024
        
        return {
            "device": device,
            "allocated_bytes": allocated,
            "allocated_mb": allocated_mb,
            "reserved_bytes": reserved,
            "reserved_mb": reserved_mb
        }
    else:
        return {}

def find_cuda_objects():
    cuda_objects = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                cuda_objects.append(obj)
            elif isinstance(obj, torch.nn.Module) and any(p.is_cuda for p in obj.parameters()):
                cuda_objects.append(obj)
        except:
            pass  # Avoid issues with objects that can't be inspected
    return cuda_objects

def find_cuda_tensors_with_grads():
    tensors_with_grads = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.is_cuda and obj.grad is not None:
                tensors_with_grads.append(obj)
        except:
            pass  # Ignore inaccessible objects
    return tensors_with_grads