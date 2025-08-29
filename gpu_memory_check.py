import torch
import gc
import psutil

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        print("=== GPU信息 ===")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # 当前GPU内存使用情况
        current_device = torch.cuda.current_device()
        print(f"\n=== GPU {current_device} 内存使用情况 ===")
        
        # 获取内存信息（字节）
        total_memory = torch.cuda.get_device_properties(current_device).total_memory
        allocated_memory = torch.cuda.memory_allocated(current_device)
        cached_memory = torch.cuda.memory_reserved(current_device)
        
        # 转换为GB
        total_gb = total_memory / 1024**3
        allocated_gb = allocated_memory / 1024**3
        cached_gb = cached_memory / 1024**3
        free_gb = total_gb - allocated_gb
        
        print(f"总内存: {total_gb:.2f} GB")
        print(f"已分配: {allocated_gb:.2f} GB")
        print(f"已缓存: {cached_gb:.2f} GB")
        print(f"可用内存: {free_gb:.2f} GB")
        
        return free_gb
    else:
        print("CUDA不可用，使用CPU")
        return None

def check_system_memory():
    """检查系统内存"""
    memory = psutil.virtual_memory()
    print(f"\n=== 系统内存 ===")
    print(f"总内存: {memory.total / 1024**3:.2f} GB")
    print(f"可用内存: {memory.available / 1024**3:.2f} GB")
    print(f"使用率: {memory.percent}%")
    return memory.available / 1024**3

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU缓存已清理")

if __name__ == "__main__":
    check_gpu_memory()
    check_system_memory()
    clear_gpu_cache()
