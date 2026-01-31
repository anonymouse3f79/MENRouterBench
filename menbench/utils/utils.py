import re
import yaml
import os
import glob
import json
import random
from typing import Tuple, Dict, Optional

from menbench.utils.paths import get_task1_saving_path
from menbench.utils.logger_utils import logger
from menbench.data.loader import Record, Sample, Result, load_existing_results

def compute_s_overall(pred: Optional[Tuple[int, int, int]],
                      gt: Tuple[int, int, int],
                      deltas: Tuple[int, int, int]) -> float:
    """
    Instance-level overall consistency s_overall(t) used for τ-thresholding in Task II.
    Matches the equations we aligned in Task I:
      c_dir(t) = mean_k f_dir
      c_num(t) = mean_k [ I(f_dir=1) * (1 - |p-g|/Delta_k) ]
      s_overall(t) = (c_dir(t) + c_num(t))/2
    If pred is None (unparsable), return 0.
    """
    if pred is None:
        return 0.0
    cdir = 0.0
    cnum = 0.0
    for k in range(3):
        p = pred[k]
        g = gt[k]
        d = deltas[k] if deltas[k] > 0 else 1
        fd = 1 if (p * g > 0) or (p == 0 and g == 0) else 0
        cdir += fd
        if fd == 1:
            # cnum += (1.0 - (abs(p - g) / float(d)))
            cnum += (p == g)
        else:
            cnum += 0.0
    cdir /= 3.0
    cnum /= 3.0
    return 0.5 * (cdir + cnum)

def align_results(configs:dict) -> Tuple[list[int], Dict[str, Dict[int, Result]]]:
    tau = float(configs["tau"])
    if not (0.0 < tau <= 1.0):
        raise ValueError("--tau must be in (0,1].")

    models = configs["models"]
    if configs["min_model"] not in models:
        raise ValueError(f"--min_model {configs['min_model']} must be included in --models")
    if configs["max_model"] not in models:
        raise ValueError(f"--max_model {configs['max_model']} must be included in --models")

    # Load per-model maps
    per_model: Dict[str, Dict[int, Result]] = {}
    for m in models:
        target_file = get_task1_saving_path(model=m,
                                results_dir=configs["results_dir"],
                                w_name="subset_" + configs["wk"],
                                no_imu=configs["no_imu"])
        per_model[m] = load_existing_results(target_file)
        logger.info(f"[LOAD] {m}: ({len(per_model[m])} records)")

    # Align by global_idx intersection across all models
    keys = None
    for m in models:
        ks = set(per_model[m].keys())
        keys = ks if keys is None else (keys & ks)
    keys = sorted(list(keys)) if keys is not None else []

    if not keys:
        raise RuntimeError("No common global_idx across the provided models.")

    # Optional decision-point filter: use is_switch_point from any model (should match across models)
    if configs["switch_only"]:
        keys = [k for k in keys if per_model[models[0]][k].sample.metadata.is_switch_point]
        if not keys:
            raise RuntimeError("After --switch_only filtering, no instances remain.")

    logger.info(f"[ALIGN] common instances = {len(keys)} (switch_only={configs['switch_only']})")

    return keys, per_model


def load_configs(base_config_path:str, input_args:dict):
    assert os.path.exists(base_config_path), "Base config does not exist. Supposed to be at configs/base.yaml"
    with open(base_config_path, "r") as f:
        yaml_config = yaml.safe_load(f)
    final_config = {**yaml_config, **{k: v for k, v in input_args.items() if v is not None or k not in yaml_config}}
    final_config["w_name"] = os.path.basename(final_config.get("subset_path", "").rstrip('/'))
    return final_config

def parse_answer(text:str):
    try:
        match = re.search(r"\[Action Output\]:\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", text)
        if match:
            return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
        return None
    except Exception:
        return None
    
def sample_dataset(subset_path:str, existing_res:dict[str:Result], num_samples:int) \
    -> Tuple[list[Sample], list[Sample]]:
    search_pattern = os.path.join(subset_path, "test", "*.jsonl")
    test_files = sorted(glob.glob(search_pattern))
    
    if not test_files:
        logger.warning("[!] Error: No test files found.")
        return None, None

    all_samples_in_testset = []
    for f in test_files:
        with open(f, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                if line.strip():
                    all_samples_in_testset.append(Sample.from_json_line(json.loads(line)))
    all_samples_in_testset.sort(key=lambda x: (x.metadata.split_id, x.metadata.timestamp))
    target_eval_set = random.sample(all_samples_in_testset, min(len(all_samples_in_testset), num_samples))
    
    samples_to_process = []
    for sample in target_eval_set:
        key = sample.get_stamp()
        if key not in existing_res:
            samples_to_process.append(sample)

    return target_eval_set, samples_to_process