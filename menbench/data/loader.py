import os
import json
from pathlib import Path
from dataclasses import dataclass, field

from typing import Dict, Any, List, Tuple, Optional

@dataclass
class Record:
    gt_label: Tuple[int, int, int]
    pred_label: Optional[Tuple[int, int, int]]  # None if unparsable
    latency: float

    @classmethod
    def from_json_line(cls, raw_data: Dict[str, Any]):
        gt_label = raw_data.get("gt_label", None)
        pred_label = raw_data.get("pred_label", None)
        latency = raw_data.get("latency", None)
        return cls(gt_label = gt_label, pred_label=pred_label, latency=latency)

@dataclass
class Obs:
    images: Dict[str, Any] = field(default_factory=dict)
    imu: List[float] = field(default_factory=list)
    past_actions: List[Any] = field(default_factory=list)

@dataclass
class MetaData:
    timestamp: float
    is_switch_point: bool
    split_id: str
    global_idx: int

@dataclass
class Sample:
    metadata: MetaData
    obs: Obs
    label: Tuple[int, int, int]

    @classmethod
    def from_json_line(cls, raw_data: Dict[str, Any]):
        meta_dict = raw_data.get("metadata", {})
        obs_dict = raw_data.get("obs", {})

        return cls(metadata=MetaData(
            timestamp=meta_dict["timestamp"],
            is_switch_point=meta_dict["is_switch_point"],
            split_id=meta_dict["split_id"],
            global_idx=meta_dict["global_idx"]
        ), obs=Obs(
            images=obs_dict.get("images", {}),
            imu=obs_dict.get("imu", []),
            past_actions=obs_dict.get("past_actions", [])
        ),
        label=raw_data.get("label", [0, 0, 0]))
    
    def get_stamp(self)->int:
        return self.metadata.global_idx

@dataclass
class Result:
    sample: Sample
    record: Record

    @classmethod
    def from_json_line(cls, raw_data: Dict[str, Any]):
        return cls(sample=Sample.from_json_line(raw_data=raw_data), record=Record.from_json_line(raw_data=raw_data))
    def get_stamp(self)->int:
        return self.sample.get_stamp()

def load_existing_results(res_file:str) -> dict[int, Result]:
        results = {}
        if os.path.exists(res_file):
            with open(res_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    item = Result.from_json_line(json.loads(line))
                    key = item.get_stamp()
                    results[key] = item
        return results