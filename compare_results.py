"""Compare old vs new evaluation results."""
import json
import pandas as pd
from pathlib import Path

old_results = pd.read_csv("output/batch_eval/public_videos/summary.csv")
new_results = pd.read_csv("output/batch_eval/new_normalized/summary.csv")

print("=" * 80)
print("评测结果对比：旧系统 vs 新系统（肩宽归一化）")
print("=" * 80)
print()

print("关键指标对比：")
print("-" * 80)

# 在位识别率对比
old_seated_rate = []
new_seated_rate = []

for idx, row in old_results.iterrows():
    presence = eval(row['presence_counts'])
    total = row['frames']
    seated = presence.get('seated', 0)
    old_seated_rate.append(seated / total * 100)

for idx, row in new_results.iterrows():
    presence = eval(row['presence_counts'])
    total = row['frames']
    seated = presence.get('seated', 0)
    new_seated_rate.append(seated / total * 100)

print(f"平均在位识别率:")
print(f"  旧系统: {sum(old_seated_rate)/len(old_seated_rate):.1f}%")
print(f"  新系统: {sum(new_seated_rate)/len(new_seated_rate):.1f}%")
print()

# 异常姿态检测对比
print("异常姿态检测对比（期望有异常的视频）:")
print("-" * 80)

abnormal_videos = [
    ("boy_leaning_on_desk.mp4", ["leaning_left", "leaning_right"]),
    ("man_neck_pain_laptop.mp4", ["head_down", "leaning_left", "leaning_right"]),
    ("student_resting_on_desk.mp4", ["head_down", "too_close"]),
    ("student_sleeping_on_desk.mp4", ["head_down", "too_close"]),
    ("tired_boy_home_study.mp4", ["head_down", "too_close"]),
    ("focused_student_dim_desk.mp4", ["too_close"]),
]

for video_name, expected_anomalies in abnormal_videos:
    old_row = old_results[old_results['file'] == video_name].iloc[0]
    new_row = new_results[new_results['file'] == video_name].iloc[0]

    old_raw_posture = eval(old_row['raw_posture_counts'])
    new_raw_posture = eval(new_row['raw_posture_counts'])

    old_presence = eval(old_row['presence_counts'])
    new_presence = eval(new_row['presence_counts'])

    print(f"\n{video_name}")
    print(f"  期望检测: {', '.join(expected_anomalies)}")
    print(f"  旧系统:")
    print(f"    在位: {old_presence.get('seated', 0)}/{old_row['frames']} 帧")
    print(f"    姿态: {old_raw_posture}")
    print(f"  新系统:")
    print(f"    在位: {new_presence.get('seated', 0)}/{new_row['frames']} 帧")
    print(f"    姿态: {new_raw_posture}")

print()
print("=" * 80)
print("详细数据已保存到 CSV 文件，可用 Excel 打开查看")
print("=" * 80)
