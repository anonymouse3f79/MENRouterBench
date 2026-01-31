import os
import re
import csv
import glob

def parse_report(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    model = re.search(r"Model: (.*?) \|", content).group(1)
    subset = re.search(r"Subset: (.*)", content).group(1)
    
    no_imu = "Yes" if "_noimu" in file_path else "No"
    
    inst_rate = re.search(r"Instruction Following Rate: ([\d\.]+)%", content).group(1)
    latency = re.search(r"Average End-to-End Latency: ([\d\.]+) s", content).group(1)

    macro_dirs = re.findall(r"\[Macro Directional\]: ([\d\.]+)", content)
    macro_nums = re.findall(r"\[Macro Numerical\]: ([\d\.]+)", content)
    overall_scores = re.findall(r"\[Overall Consistency Score\]: ([\d\.]+)", content)

    data = {
        "Model": model,
        "Subset": subset,
        "No_IMU": no_imu,
        "Inst_Following_%": inst_rate,
        "Avg_Latency_s": latency,
        # Overall
        "All_Macro_Dir": macro_dirs[0] if len(macro_dirs) > 0 else "N/A",
        "All_Macro_Num": macro_nums[0] if len(macro_nums) > 0 else "N/A",
        "All_Overall_Score": overall_scores[0] if len(overall_scores) > 0 else "N/A",
        # T_dp
        "DP_Macro_Dir": macro_dirs[1] if len(macro_dirs) > 1 else "N/A",
        "DP_Macro_Num": macro_nums[1] if len(macro_nums) > 1 else "N/A",
        "DP_Overall_Score": overall_scores[1] if len(overall_scores) > 1 else "N/A",
    }
    return data

def main(input_dir, output_file):
    report_files = glob.glob(os.path.join(input_dir, "*_fine_report.txt"))
    
    if not report_files:
        print(f"No report files found in {input_dir}")
        return

    results = []
    for f in report_files:
        try:
            results.append(parse_report(f))
        except Exception as e:
            print(f"Error parsing {f}: {e}")

    headers = [
        "Model", "Subset", "No_IMU", "Inst_Following_%", "Avg_Latency_s",
        "All_Macro_Dir", "All_Macro_Num", "All_Overall_Score",
        "DP_Macro_Dir", "DP_Macro_Num", "DP_Overall_Score"
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)

    print(f"Successfully summarized {len(results)} reports to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./results_task1", help="Directory containing report files")
    parser.add_argument("--output", type=str, default="benchmark_task1_summary.csv", help="Output CSV file name")
    args = parser.parse_args()

    main(args.input_dir, args.output)