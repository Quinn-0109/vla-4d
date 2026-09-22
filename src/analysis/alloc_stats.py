"""把 patch→slot 分配转换成可审计的帧贡献统计；纯 Python，便于离线测试。"""
from __future__ import annotations

from collections import Counter, defaultdict


def summarize_assignment(assign, frame_pad_mask, source_steps, patches_per_frame: int = 256) -> dict:
    """汇总一个样本、一个环境 step 的 token 来源。

    ``assign`` 按帧连续排列，值为输出 slot，``-1`` 表示未参与或被丢弃。
    ``frame_pad_mask`` 中 True 表示真实帧，False 表示 episode 开头的重复补帧。
    """
    assign = [int(x) for x in assign]
    mask = [bool(x) for x in frame_pad_mask]
    source_steps = [int(x) for x in source_steps]
    if patches_per_frame <= 0:
        raise ValueError("patches_per_frame 必须为正")
    if not mask or len(source_steps) != len(mask):
        raise ValueError("frame_pad_mask 与 source_steps 长度必须相同且非空")
    if len(assign) != len(mask) * patches_per_frame:
        raise ValueError("assign 长度必须等于 K * patches_per_frame")
    if any(x < -1 for x in assign):
        raise ValueError("assign 只能是 -1 或非负 slot")
    valid_frames = [i for i, ok in enumerate(mask) if ok]
    if not valid_frames:
        raise ValueError("至少要有一个真实帧")
    latest = max(valid_frames)

    by_slot = defaultdict(list)
    padding_assigned = valid_assigned = 0
    for patch_index, slot in enumerate(assign):
        if slot < 0:
            continue
        frame = patch_index // patches_per_frame
        by_slot[slot].append(frame)
        if mask[frame]:
            valid_assigned += 1
        else:
            padding_assigned += 1

    per_frame = []
    slot_weight_sum = Counter()
    slots = []
    mixed = latest_exclusive = 0
    frame_spans, step_spans = [], []
    latest_weight_sum = 0.0
    for slot in sorted(by_slot):
        frames = by_slot[slot]
        counts = Counter(frames)
        total = len(frames)
        valid_sources = sorted(f for f in counts if mask[f])
        source_rows = []
        for frame in sorted(counts):
            weight = counts[frame] / total
            slot_weight_sum[frame] += weight
            source_rows.append({
                "frame_index": frame,
                "source_step": source_steps[frame],
                "is_real": mask[frame],
                "patches": counts[frame],
                "weight": weight,
            })
        latest_weight = counts.get(latest, 0) / total
        latest_weight_sum += latest_weight
        is_mixed = len(valid_sources) > 1
        is_latest_exclusive = valid_sources == [latest] and not any(not mask[f] for f in counts)
        mixed += int(is_mixed)
        latest_exclusive += int(is_latest_exclusive)
        frame_span = max(valid_sources) - min(valid_sources) if valid_sources else 0
        valid_steps = [source_steps[f] for f in valid_sources]
        step_span = max(valid_steps) - min(valid_steps) if valid_steps else 0
        frame_spans.append(frame_span)
        step_spans.append(step_span)
        slots.append({
            "slot": slot,
            "patches": total,
            "sources": source_rows,
            "mixed_real_frames": is_mixed,
            "latest_frame_weight": latest_weight,
            "frame_span": frame_span,
            "step_span": step_span,
        })

    for frame in range(len(mask)):
        start, end = frame * patches_per_frame, (frame + 1) * patches_per_frame
        assigned = sum(x >= 0 for x in assign[start:end])
        per_frame.append({
            "frame_index": frame,
            "source_step": source_steps[frame],
            "is_real": mask[frame],
            "input_patches": patches_per_frame,
            "assigned_patches": assigned,
            "patch_retention": assigned / patches_per_frame,
            "slot_weight_sum": float(slot_weight_sum[frame]),
        })

    used = len(by_slot)
    return {
        "schema_version": 1,
        "patches_per_frame": patches_per_frame,
        "frames": len(mask),
        "real_frames": len(valid_frames),
        "latest_real_frame": latest,
        "slots_used": used,
        "valid_input_patches": len(valid_frames) * patches_per_frame,
        "valid_assigned_patches": valid_assigned,
        "valid_dropped_patches": len(valid_frames) * patches_per_frame - valid_assigned,
        "padding_assigned_patches": padding_assigned,
        "mixed_frame_slots": mixed,
        "mixed_frame_slot_fraction": mixed / used if used else 0.0,
        "latest_frame_exclusive_slots": latest_exclusive,
        "latest_frame_slot_weight_sum": latest_weight_sum,
        "latest_frame_mean_slot_weight": latest_weight_sum / used if used else 0.0,
        "mean_frame_span": sum(frame_spans) / used if used else 0.0,
        "max_frame_span": max(frame_spans, default=0),
        "mean_step_span": sum(step_spans) / used if used else 0.0,
        "max_step_span": max(step_spans, default=0),
        "per_frame": per_frame,
        "slots": slots,
    }
