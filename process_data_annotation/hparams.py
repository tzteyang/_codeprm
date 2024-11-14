from dataclasses import dataclass
from typing import (
    Dict, List, Union, Optional, Tuple
)


@dataclass
class DataParams:
    data_file: str
    output_dir: str
    logs_dir: str

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(
            data_file=config_dict.get('data_file'),
            output_dir=config_dict.get('output_dir', None),
            logs_dir=config_dict.get('logs_dir', None)
        )


@dataclass
class ModelParams:
    model_type: str
    model_name: str
    temperature: float
    max_tokens: int
    max_completion_tokens: int

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(
            model_type=config_dict.get('model_type'),
            model_name=config_dict.get('model_name'),
            temperature=config_dict.get('temperature', 1.0),
            max_tokens=config_dict.get('max_tokens', 512),
            max_completion_tokens=config_dict.get('max_completion_tokens', 2048)
        )


@dataclass
class SearchAlgorithmParams:
    c_puct: float
    alpha: float
    beta: float 
    length_scale: int
    num_rollouts: int
    max_search_count: int 
    rollout_budget: int

    @classmethod
    def from_dict(cls, config_dict: Dict):
        return cls(
            c_puct=config_dict.get('c_puct', 0.125),
            alpha=config_dict.get('alpha', 0.5),
            beta=config_dict.get('beta', 0.9),
            length_scale=config_dict.get('length_scale', 500),
            num_rollouts=config_dict.get('num_rollouts', 16),
            max_search_count=config_dict.get('max_search_count', 8),
            rollout_budget=config_dict.get('rollout_budget', 200)
        )