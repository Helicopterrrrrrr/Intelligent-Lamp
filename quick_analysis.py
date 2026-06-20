"""Quick analysis of evaluation results."""
import json
from pathlib import Path

def analyze_results(result_dir, label):
    summary_file = Path(result_dir) / "summary.json"
    if not summary_file.exists():
        print(f"{label}: 结果文件不存在")
        return

    with open(summary_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    total_videos = len(results)
    total_frames = sum(r['frames'] for r in results)

    # 统计在位识别
    seated_frames = 0
    absent_frames = 0
    for r in results:
        presence = r['presence_counts']
        seated_frames += presence.get('seated', 0)
        absent_frames += presence.get('absent', 0)

    print(f"\n总体统计:")
    print(f"  视频数: {total_videos}")
    print(f"  总帧数: {total_frames}")
    print(f"  在位帧数: {seated_frames} ({seated_frames/total_frames*100:.1f}%)")
    print(f"  离位帧数: {absent_frames} ({absent_frames/total_frames*100:.1f}%)")

    # 统计异常检测
    print(f"\n异常姿态检测:")
    abnormal_count = 0
    for r in results:
        raw_posture = r['raw_posture_counts']
        if any(k != 'normal' for k in raw_posture.keys()):
            abnormal_count += 1
            abnormal_frames = sum(v for k, v in raw_posture.items() if k != 'normal')
            print(f"  {r['video']}: {abnormal_frames} 帧异常")

    print(f"\n检测到异常的视频数: {abnormal_count}/{total_videos}")

# 分析旧系统
analyze_results("output/batch_eval/public_videos", "旧系统（固定阈值）")

# 分析新系统
analyze_results("output/batch_eval/new_normalized", "新系统（肩宽归一化）")

print(f"\n{'='*60}")
