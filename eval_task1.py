from menbench.evaluator import Task1Evaluator
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_path", type=str, required=True)
    parser.add_argument("--api_key",     type=str, required=True)
    parser.add_argument("--model",       type=str, required=True)

    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--image_root", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--concurrency", type=int)
    
    parser.add_argument("--no_imu", action="store_true")
    
    args = parser.parse_args()
    Task1Evaluator(vars(args)).run()

if __name__ == "__main__":
    main()