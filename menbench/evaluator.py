import os
import json
import random
import argparse
import time
from tqdm import tqdm
import math
from typing import Dict, Tuple, List
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from menbench.viz import report_task1
from menbench.data import load_existing_results

from menbench.utils import logger
from menbench.utils.paths import get_task1_saving_path, get_task2_saving_path
from menbench.utils import load_configs, parse_answer, sample_dataset, compute_s_overall, align_results

from menbench.server import AGENT, ROUTER, RouterServer
from menbench.data.loader import Sample
from menbench.data import Result

BASE_TASK1_CONFIG = "configs/base_task1.yaml"
BASE_TASK2_CONFIG = "configs/base_task2.yaml"

class Task1Evaluator:
    def __init__(self, args:dict):
        self.args = load_configs(BASE_TASK1_CONFIG, args)
        if not os.path.exists(self.args["results_dir"]): os.makedirs(self.args["results_dir"])
        
        random.seed(self.args["seed"])
        self.args["res_file"] = get_task1_saving_path(**self.args)
        self.agent = AGENT.get(self.args["agent_server"])(self.args)
        
        # 1. 加载已有的结果，用于去重
        self.existing_res = load_existing_results(self.args["res_file"])
        logger.info(f"[*] Loaded {len(self.existing_res)} existing results from {self.args['res_file']}")

    def call_agent(self, sample:Sample) -> dict:
        start_time = time.perf_counter()  # 开始计时
        try:
            response = self.agent.run(sample)
            latency = time.perf_counter() - start_time  # 结束计时
            
            if response.status_code == 200:
                raw_reply = response.json()['choices'][0]['message']['content']
                return {
                    "obs": asdict(sample.obs),
                    "metadata": asdict(sample.metadata), 
                    "gt_label": sample.label, 
                    "pred_label": parse_answer(raw_reply), 
                    "raw_reply": raw_reply,
                    "latency": round(latency, 4)  # 保存时延
                }
        except Exception as e:
            pass
        return {
                    "obs": asdict(sample.obs),
                    "metadata": asdict(sample.metadata), 
                    "gt_label": sample.label, 
                    "pred_label": None, 
                    "raw_reply": None,
                    "latency": -1  # 保存时延
                }

    def run(self) -> None:
        target_eval_set, samples_to_process = sample_dataset(self.args["subset_path"], self.existing_res, self.args["num_samples"])
        
        logger.info(f"[*] Target Evaluation Set Size: {len(target_eval_set)}")
        logger.info(f"[*] Already completed: {len(target_eval_set) - len(samples_to_process)}")
        logger.info(f"[*] Remaining to process: {len(samples_to_process)}")

        if not samples_to_process:
            logger.info("[*] All targeted samples have been evaluated.")
            self.report()
            return

        with ThreadPoolExecutor(max_workers=self.args["concurrency"]) as executor:
            futures = [executor.submit(self.call_agent, s) for s in samples_to_process]
            for future in tqdm(as_completed(futures), total=len(futures), desc="VLM Inference"):
                res = future.result()
                if res:
                    with open(self.args["res_file"], 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(res) + '\n')

        self.report()

    def report(self) -> None:
        final_results = load_existing_results(self.args["res_file"])

        if not final_results:
            logger.warning("[!] No results found to report.")
            return
        
        report_task1(final_results, self.args)

class Task2Evaluator:
    def __init__(self, args:dict):
        self.args = load_configs(BASE_TASK2_CONFIG, args)
        self.router_servers = [ROUTER.get(router_cls)(self.args) for router_cls in self.args["router_servers"]]
        self.compared_to_router_server = ROUTER.get(self.args["compared_to_router_server"])(self.args)


    def call_router(self, router_instance:RouterServer, result:dict[str,Result], oks:dict[str, int]) \
        -> Tuple[float, float] | None:
        return router_instance.run(result, oks)

    def Determine_Lmin_Lmax(self, lat, sol_idx):
        all_lats_sol = []
        for m in self.args["models"]:
            all_lats_sol.extend(lat[m][i] for i in sol_idx)
        Lmin = float(self.args["Lmin"]) if self.args["Lmin"] is not None else float(min(all_lats_sol))
        Lmax = float(self.args["Lmax"]) if self.args["Lmax"] is not None else float(max(all_lats_sol))
        if Lmax <= Lmin:
            raise RuntimeError(f"Invalid budget interval: Lmin={Lmin}, Lmax={Lmax}")
        logger.info(f"[BUDGET] Lmin={Lmin:.6f}, Lmax={Lmax:.6f}")
        return Lmin, Lmax
    
    def compute_weighted_auc(self, events: List[Tuple[float, float]], Lmin: float, Lmax: float, denom: int) -> float:
        """
        Compute AUC = ∫_{Lmin}^{Lmax} Acc(L) dL for a right-continuous step function:
        Acc(L) = (sum_{i: lat_i <= L} w_i) / denom.
        events: list of (latency, weight) with weight >= 0.
        denom: number of instances in D_sol(τ).

        Returns AUC (not normalized).
        """
        if denom <= 0:
            return 0.0
        if Lmax <= Lmin:
            return 0.0

        events_sorted = sorted(events, key=lambda x: x[0])

        # Initialize cumulative weight with events at/below Lmin
        cum = 0.0
        idx = 0
        n = len(events_sorted)
        while idx < n and events_sorted[idx][0] <= Lmin:
            cum += events_sorted[idx][1]
            idx += 1

        area = 0.0
        cur = Lmin

        while idx < n and events_sorted[idx][0] < Lmax:
            nxt = events_sorted[idx][0]
            if nxt > cur:
                area += (cum / float(denom)) * (nxt - cur)
                cur = nxt
            # absorb all events at this latency
            while idx < n and events_sorted[idx][0] == nxt:
                cum += events_sorted[idx][1]
                idx += 1

        if cur < Lmax:
            area += (cum / float(denom)) * (Lmax - cur)

        return area

    def prepare(self,
                keys:list[int],
                per_model:Dict[str, Dict[int, Result]]):
        # Infer per-axis Delta_k from gt/pred values across the aligned set and models
        axis_min = [math.inf, math.inf, math.inf]
        axis_max = [-math.inf, -math.inf, -math.inf]
        for k in keys:
            for m in self.args["models"]:
                r = per_model[m][k] # Result instance
                for ax in range(3):
                    axis_min[ax] = min(axis_min[ax], r.record.gt_label[ax])
                    axis_max[ax] = max(axis_max[ax], r.record.gt_label[ax])
                    if r.record.pred_label is not None:
                        axis_min[ax] = min(axis_min[ax], r.record.pred_label[ax])
                        axis_max[ax] = max(axis_max[ax], r.record.pred_label[ax])

        deltas = tuple(int(axis_max[i] - axis_min[i]) if axis_max[i] > axis_min[i] else 1 for i in range(3))
        logger.info(f"[DELTA] inferred deltas (yaw,lon,lat) = {deltas} from ranges {list(zip(axis_min, axis_max))}")

        lat: Dict[str, List[float]] = {m: [] for m in self.args["models"]}
        ok: Dict[str, List[int]] = {m: [] for m in self.args["models"]}
        for k in keys:
            for m in self.args["models"]:
                r = per_model[m][k] # Result
                s = compute_s_overall(r.record.pred_label, r.record.gt_label, deltas)
                lat[m].append(r.record.latency)
                ok[m].append(1 if s >= self.args["tau"] else 0)

        # Build D_sol(τ): indices i where exists model with ok==1
        sol_idx: List[int] = []
        for i in range(len(keys)):
            if any(ok[m][i] == 1 for m in self.args["models"]):
                sol_idx.append(i)

        logger.info(f"[SOLVABLE] |D|={len(keys)}, |D_sol(tau)|={len(sol_idx)}, rho_sol(tau)={len(sol_idx) / float(len(keys)):.4f} (tau={self.args['tau']})")
        if len(sol_idx) == 0:
            raise RuntimeError("D_sol(tau) is empty; consider lowering --tau.")
        
        return lat, ok, sol_idx


    def run(self):
        keys, per_model = align_results(self.args)
        
        # Precompute per-instance, per-model: latency, tau-correct indicator
        lat, ok, sol_idx = self.prepare(keys,
                                        per_model)
        
        Lmin, Lmax = self.Determine_Lmin_Lmax(lat, sol_idx)


        server_events: Dict[str, List[Tuple[float, float]]] = {}
        for router_server in self.router_servers:
            for i in sol_idx:
                ev = self.call_router(router_server, 
                                    {m: per_model[m][keys[i]] for m in self.args["models"]}, # Result
                                    {m: ok[m][i] for m in self.args["models"]}, # oks
                                    )
                if ev:
                    if not server_events.get(type(router_server).__name__, None):
                        server_events[type(router_server).__name__] = []
                    server_events[type(router_server).__name__].append(ev)

        compared_server_events: Dict[str, List[Tuple[float, float]]] = {}
        for i in sol_idx:
            ev = self.call_router(self.compared_to_router_server, 
                                    {m: per_model[m][keys[i]] for m in self.args["models"]}, # Result
                                    {m: ok[m][i] for m in self.args["models"]}, # oks
                                    )
            if ev:
                if not compared_server_events.get(type(self.compared_to_router_server).__name__, None):
                    compared_server_events[type(self.compared_to_router_server).__name__] = []
                compared_server_events[type(self.compared_to_router_server).__name__].append(ev)
        
        assert len(compared_server_events[type(self.compared_to_router_server).__name__]) == len(sol_idx), f"Events should match solvable count: {compared_server_events[type(router_server).__name__]}"


        auc_scores : dict[str, float] = {}
        for router_server in self.router_servers:
            auc_scores[type(router_server).__name__] = \
                self.compute_weighted_auc(server_events[type(router_server).__name__], Lmin=Lmin, Lmax=Lmax, denom=len(sol_idx))
            
        compared_to_auc_scores : dict[str, float] = {}
        compared_to_auc_scores[type(self.compared_to_router_server).__name__] = \
            self.compute_weighted_auc(compared_server_events[type(self.compared_to_router_server).__name__], Lmin=Lmin, Lmax=Lmax, denom=len(sol_idx))


        self.report(
            auc_scores,
            compared_to_auc_scores,
            server_events,
            compared_server_events,
            Lmin,
            Lmax,
            sol_idx,
            keys
        )

    def report(self,
               auc_scores,
               compared_to_auc_scores,
               server_events,
               compared_server_events,
               Lmin,
               Lmax,
               sol_idx,
               keys
               ):
        ###### report
        def safe_ratio(a, b):
            return 0.0 if b <= 0 else max(0.0, min(1.0, a / b))

        save_curves = get_task2_saving_path(**self.args)

        report_content = "\n===== Task II (AUC-normalized) Scores ====="

        for router_server in self.router_servers:
            report_content += f"\n{type(router_server).__name__} AUC\t: {auc_scores[type(router_server).__name__]:.6f} \tScore: {safe_ratio(auc_scores[type(router_server).__name__], compared_to_auc_scores[type(self.compared_to_router_server).__name__]):.6f}"
        # logger.info("Compared to chosen routers:")
        report_content += f"\n{type(self.compared_to_router_server).__name__} AUC\t: {compared_to_auc_scores[type(self.compared_to_router_server).__name__]:.6f}"


        if "stdio" in self.args["report_to"]:
            logger.info(report_content)
        if "txt" in self.args["report_to"]:
            with open(save_curves.replace(".json", "_fine_report.txt"), 'w', encoding='utf-8') as f:
                f.write(report_content)
        # Use a compact grid: union of event latencies + endpoints (clipped to [Lmin,Lmax])
        def grid_from_events(ev: List[Tuple[float, float]]) -> List[float]:
            xs = [Lmin, Lmax]
            xs += [x for (x, _) in ev if Lmin <= x <= Lmax]
            xs = sorted(set(xs))
            return xs

        all_event = []
        for router_server in self.router_servers:
            all_event = all_event + server_events[type(router_server).__name__]

        grid = sorted(set(
            grid_from_events(all_event) 
        ))

        # Evaluate right-continuous Acc(L) on grid
        def eval_acc(ev: List[Tuple[float, float]], grid_: List[float]) -> List[float]:
            evs = sorted(ev, key=lambda x: x[0])
            acc = []
            cum = 0.0
            j = 0
            for L in grid_:
                while j < len(evs) and evs[j][0] <= L:
                    cum += evs[j][1]
                    j += 1
                acc.append(cum / float(len(sol_idx)))
            return acc

        out = {
            "meta": {
                "wk": self.args["wk"],
                "imu": self.args["no_imu"],
                "tau": self.args["tau"],
                "switch_only": bool(self.args["switch_only"]),
                "models": self.args["models"],
                "min_model": self.args["min_model"],
                "max_model": self.args["max_model"],
                "n_all_aligned": len(keys),
                "n_sol": len(sol_idx),
                "rho_sol": len(sol_idx) / float(len(keys)),
                "Lmin": Lmin,
                "Lmax": Lmax,
                "auc": {type(router_server).__name__: auc_scores[type(router_server).__name__] for router_server in self.router_servers},
                "compared_to_auc": {type(self.compared_to_router_server).__name__: compared_to_auc_scores[type(self.compared_to_router_server).__name__]}
            },
            "grid_L": grid,
            "acc": {type(router_server).__name__: eval_acc(server_events[type(router_server).__name__], grid) for router_server in self.router_servers},
            "compared_to_acc": {type(self.compared_to_router_server).__name__: eval_acc(compared_server_events[type(self.compared_to_router_server).__name__], grid)},
            
        }
        os.makedirs(os.path.dirname(save_curves), exist_ok=True)
        with open(save_curves, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info(f"\n[SAVED] curves -> {save_curves}")
    

def main_task1():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_path", type=str, required=True)
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--model",       type=str, required=True)

    parser.add_argument("--results_dir", type=str, help="Directory containing Task I jsonl outputs (default: results_task1)")
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--concurrency", type=int)
    
    parser.add_argument("--no_imu", action="store_true")
    
    args = parser.parse_args()
    Task1Evaluator(vars(args)).run()

def main_task2():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                    help="Directory containing Task I jsonl outputs (default: results_task1)")
    parser.add_argument("--wk", type=str, required=True, choices=["w3", "w4", "w5"],
                    help="Window K setting (wk): one of {w3,w4,w5}")
    parser.add_argument("--no_imu", type=bool, required=False, default=False,
                    help="IMU setting: imu or noimu")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                    help="Model names comprising the model zoo group for routing evaluation")
    parser.add_argument("--min_model", type=str, required=True,
                    help="Model name used by the 'min-model routing' baseline (user prior)")
    parser.add_argument("--max_model", type=str, required=True,
                    help="Model name used by the 'max-model routing' baseline (user prior)")
    parser.add_argument("--tau", type=float, default=0.5,
                    help="Threshold tau for τ-correctness (default: 0.5)")
    parser.add_argument("--switch_only", action="store_true",
                    help="If set, restrict to metadata.is_switch_point == True (recommended)")
    parser.add_argument("--Lmin", type=float, default=0.0,
                    help="Latency budget lower bound. Default: min latency over aligned instances across the group.")
    parser.add_argument("--Lmax", type=float, default=30,
                    help="Latency budget upper bound. Default: max latency over aligned instances across the group.")
    parser.add_argument("--seed", type=int, default=0,
                    help="Unused (random policy is computed in expectation), kept for reproducibility hooks.")
    parser.add_argument("--save_curves", type=str,
                    help="Optional path to save curves as JSON (oracle/min/max/random).")
    parser.add_argument("--group_name", type=str, default="test")
    
    args = parser.parse_args()
    Task2Evaluator(vars(args)).run()

if __name__ == "__main__":
    # python -m menbench.evaluator --subset_path configs/subset_w3/ --model google/gemini-2.5-pro --api_key your-api-key
    main_task2()