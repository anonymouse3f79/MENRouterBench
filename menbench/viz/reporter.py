from menbench.viz.radar import draw_radar_chart
from menbench.data import Result
from menbench.utils import logger

def calculate_score(parsed_results:list[Result]):
    def init_stats():
        return {
            'total': 0,
            'dir_correct': {'yaw': 0, 'lon': 0, 'lat': 0},
            'num_correct': {'yaw': 0, 'lon': 0, 'lat': 0},
            'num_denom': {'yaw': 0, 'lon': 0, 'lat': 0}
        }

    stats_all = init_stats()
    stats_dp = init_stats()
    axes = ['yaw', 'lon', 'lat']

    for item in parsed_results:
        gt, pred = item.record.gt_label, item.record.pred_label
        is_dp = item.sample.metadata.is_switch_point

        def is_dir_match(p, g):
            if g > 0 and p > 0: return True
            if g < 0 and p < 0: return True
            if g == 0 and p == 0: return True
            return False

        current_metrics = {'dir': {}, 'num': {}}
        for i, axis in enumerate(axes):
            g_val, p_val = gt[i], pred[i]
            dir_match = is_dir_match(p_val, g_val)
            current_metrics['dir'][axis] = dir_match
            if dir_match:
                current_metrics['num'][axis] = (p_val == g_val)
            else:
                current_metrics['num'][axis] = None

        def update_stats(s, metrics):
            s['total'] += 1
            for ax in axes:
                if metrics['dir'][ax]:
                    s['dir_correct'][ax] += 1
                    s['num_denom'][ax] += 1
                    if metrics['num'][ax]:
                        s['num_correct'][ax] += 1

        update_stats(stats_all, current_metrics)
        if is_dp:
            update_stats(stats_dp, current_metrics)

    def calculate_final_scores(s):
        if s['total'] == 0: return [0]*9
        acc_dir_yaw = s['dir_correct']['yaw'] / s['total']
        acc_dir_lon = s['dir_correct']['lon'] / s['total']
        acc_dir_lat = s['dir_correct']['lat'] / s['total']
        macro_dir = (acc_dir_yaw + acc_dir_lon + acc_dir_lat) / 3
        acc_num_yaw = s['num_correct']['yaw'] / s['num_denom']['yaw'] if s['num_denom']['yaw'] > 0 else 0
        acc_num_lon = s['num_correct']['lon'] / s['num_denom']['lon'] if s['num_denom']['lon'] > 0 else 0
        acc_num_lat = s['num_correct']['lat'] / s['num_denom']['lat'] if s['num_denom']['lat'] > 0 else 0
        macro_num = (acc_num_yaw + acc_num_lon + acc_num_lat) / 3
        overall = (macro_dir + macro_num) / 2
        return (acc_dir_yaw, acc_dir_lon, acc_dir_lat, macro_dir, 
                acc_num_yaw, acc_num_lon, acc_num_lat, macro_num, overall)

    return calculate_final_scores(stats_all), calculate_final_scores(stats_dp), stats_all['total'], stats_dp['total']

def report_task1(final_results:dict[str, Result], configs):
    total_queried = len(final_results)
    # 统计平均时延
    latency_list = [item.record.latency for key, item in final_results.items() if item.record.latency]

    total_latency = sum(latency_list)
    max_latency = max(latency_list)
    avg_latency = total_latency / total_queried if total_queried > 0 else 0

    parsed_results = [item for key, item in final_results.items() if item.record.pred_label is not None]
    total_parsed = len(parsed_results)
    inst_following_rate = total_parsed / total_queried if total_queried > 0 else 0

    scores_all, scores_dp, len_all, len_dp = calculate_score(parsed_results)

    report_content = f"--- MENRouterBench Task I Fine-grained Report ---\n"
    report_content += f"Model: {configs['model']} | Subset: {configs['w_name']}\n"
    report_content += f"Instruction Following Rate: {inst_following_rate:.2%} ({total_parsed}/{total_queried})\n"
    report_content += f"Max End-to-End Latency: {max_latency:.4f} s\n"
    report_content += f"Average End-to-End Latency: {avg_latency:.4f} s\n"
    
    def format_stats(title, sc, total_cnt):
        return f"""
{title} (Valid Samples: {total_cnt})
--------------------------------------------------
Directional Consistency:
Yaw: {sc[0]:.4f} | Lon: {sc[1]:.4f} | Lat: {sc[2]:.4f}
[Macro Directional]: {sc[3]:.4f}
Numerical Consistency (Conditional):
Yaw: {sc[4]:.4f} | Lon: {sc[5]:.4f} | Lat: {sc[6]:.4f}
[Macro Numerical]: {sc[7]:.4f}
>>> [Overall Consistency Score]: {sc[8]:.4f}
--------------------------------------------------
"""
    report_content += format_stats("OVERALL STATISTICS", scores_all, len_all)
    report_content += format_stats("DECISION POINT (T_dp) STATISTICS", scores_dp, len_dp)


    if "radar" in configs["report_to"]:
        scores_all = list(scores_all)
        scores_all.append(avg_latency)
        info = [{
            "model": configs["model"],
            "scores_all": scores_all,
        }]
        draw_radar_chart(model_stats=info, save_path=configs["res_file"].replace("_raw.jsonl", "_radar_chart.png"), max_latency=configs["max_latency"])
    if "stdio" in configs["report_to"]:
        logger.info("\n" + report_content)
    if "txt" in configs["report_to"]:
        with open(configs["res_file"].replace("_raw.jsonl", "_fine_report.txt"), 'w', encoding='utf-8') as f:
            f.write(report_content)