import os
from accelerate import Accelerator
from typing import Optional

class AcceleratorManager:
    _instance: Optional[Accelerator] = None
    
    @classmethod
    def initialize(cls, **kwargs):
        if cls._instance is None:
            cls._instance = Accelerator(**kwargs)
            
            if cls._instance.is_main_process:
                print(f"Distributed training setup:")
                print(f"- Number of processes: {cls._instance.num_processes}")
                print(f"- Mixed precision: {cls._instance.mixed_precision}")
                print(f"- Gradient accumulation steps: {cls._instance.gradient_accumulation_steps}")
                
    @classmethod
    def get_accelerator(cls) -> Accelerator:
        if cls._instance is None:
            raise RuntimeError(
                "Accelerator not initialized. Call AcceleratorManager.initialize() first."
            )
        return cls._instance
    
    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None


def get_accelerator() -> Accelerator:
    return AcceleratorManager.get_accelerator()

def is_main_process() -> bool:
    return get_accelerator().is_main_process

def get_local_rank() -> int:
    return get_accelerator().local_process_index

def get_world_size() -> int:
    return get_accelerator().num_processes

def synchronize():
    get_accelerator().wait_for_everyone()