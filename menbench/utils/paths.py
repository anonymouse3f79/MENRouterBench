import os

def get_task1_saving_path(model:str, 
                            results_dir: str,
                            w_name: str,
                            no_imu: bool,
                            **kwargs):
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"results_dir not found: {results_dir}")
    model_safe_name = model.replace("/", "_").replace(":", "_")
    return os.path.join(results_dir, 
                 f"{w_name}_{model_safe_name}{'_no_imu' if no_imu else ''}_raw.jsonl")

def get_task2_saving_path(wk,
                          no_imu,
                          save_curves,
                          group_name,
                          tau,
                          **kwargs):
    return os.path.join(save_curves, 
                        f"{wk}_{'noimu' if no_imu else 'imu'}_{group_name}_tau{tau}.json")