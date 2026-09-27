# eval_longbench.py
import json
import os
from datasets import load_from_disk
from langchain_core.messages import HumanMessage

# 从 agent 里导入编译好的图
from graph import g


BUCKETS = ["8K", "16K", "32K", "64K", "128K", "256K"]


def run_one_sample(sample: dict, redis_key: str) -> dict:
    """
    跑单条样本
    """
    init_state = {
        "messages": [HumanMessage(content="请总结")], 
        "raw_context": sample["context"],              # parse 节点直接吃这段文本
        "document_status": "idle",
        "redis_key":redis_key,
        # 显式清空，防止 reducer 累加历史（如果 map_results 用了 operator.add）
        "map_results": [],
    }

    final_state = g.invoke(init_state)

    return {
        "pred": final_state.get("extracted_summary", ""),
        "answer": sample["answer"],
        "question": sample.get("question", ""),
        "token_length": sample.get("token_length", ""),
        "document_status": final_state.get("document_status", ""),
    }


def eval_bucket(bucket_name: str, ds, max_samples: int | None = None):
    """
    评测一个桶，返回结果列表，并保存到磁盘
    """
    n = len(ds) if max_samples is None else min(len(ds), max_samples)
    if n == 0:
        print(f"⚠️  桶 {bucket_name} 为空，跳过。")
        return []

    print(f"\n===== 评测 {bucket_name} ({n} 条) =====")
    results = []

    for i in range(n):
        sample = ds[i]
        redis_key = bucket_name + "[" + i + "]"
        try:
            res = run_one_sample(sample, redis_key)
            results.append(res)
            print(f"  [{i+1}/{n}] ✅ 完成 "
                  f"(status={res['document_status']}, "
                  f"pred_len={len(res['pred'])})")
        except Exception as e:
            print(f"  [{i+1}/{n}] ❌ 失败: {e}")
            results.append({
                "pred": "",
                "answer": sample.get("answer", ""),
                "question": sample.get("question", ""),
                "token_length": sample.get("token_length", ""),
                "document_status": "error",
                "error": str(e),
            })

    out_path = f"./preds_{bucket_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存: {out_path}")

    return results


def main():
    dataset_dict = load_from_disk("./longbench_pro_T4.1_buckets")

    all_results = {}
    for bucket in BUCKETS:
        ds = dataset_dict[bucket]
        # max_samples 调试时可以先设小一点，比如 3；正式跑传 None
        results = eval_bucket(bucket, ds, max_samples=3)
        all_results[bucket] = results

    # 汇总统计
    print("\n===== 汇总 =====")
    for bucket in BUCKETS:
        rs = all_results.get(bucket, [])
        ok = sum(1 for r in rs if r["document_status"] == "completed")
        print(f"{bucket:>5}: {len(rs):>4} 条, 成功 {ok} 条")

    # 全部结果合并保存
    with open("./preds_all.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("💾 全部结果已保存: ./preds_all.json")


if __name__ == "__main__":
    main()
