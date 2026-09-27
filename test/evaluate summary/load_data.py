from datasets import load_dataset, Dataset, DatasetDict

# 加载数据集
dataset = load_dataset("caskcsg/LongBench-Pro", split="test")
print("原始数据量:", len(dataset))

# 过滤 secondary_task
TARGET_TASK = "T4.1 Global-Coverage Constrained Summary"
filtered = dataset.filter(lambda x: x["secondary_task"] == TARGET_TASK)
print(f"过滤后数据量 ({TARGET_TASK}):", len(filtered))

# 按 token_length 分成 6 个子数据集
BUCKETS = ["8k", "16k", "32k", "64k", "128k", "256k"]

dataset_dict = DatasetDict({
    b: filtered.filter(lambda x, b=b: x["token_length"] == b)
    for b in BUCKETS
})

# 查看每个桶的数量
for b in BUCKETS:
    print(f"{b}: {len(dataset_dict[b])} 条")

# 8k: 10 条
# 16k: 10 条
# 32k: 10 条
# 64k: 10 条
# 128k: 10 条
# 256k: 10 条

# 保存到磁盘
dataset_dict.save_to_disk("./longbench_pro_T4.1_buckets")