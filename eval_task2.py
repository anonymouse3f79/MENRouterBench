import argparse
from menbench.evaluator import Task2Evaluator

def main():
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
    main()