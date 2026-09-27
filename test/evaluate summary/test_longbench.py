from datasets import load_dataset

dataset = load_dataset("caskcsg/LongBench-Pro", split="test")

# # 查看 primary_task 的所有独特值
# print(sorted(dataset.unique("primary_task")))

# # 看每个主任务有多少样本
# from collections import Counter
# print(Counter(dataset["primary_task"]))

# # 看 T4 下有哪些 subtask
# t4 = dataset.filter(lambda x: x["primary_task"] == "T4. Summarization & Synthesis")
# print(sorted(t4.unique("secondary_task")))

# # 看 T4 下 contextual_requirement 的分布
# from collections import Counter
# print(Counter(t4["contextual_requirement"]))

# 筛选 T4.1
t4_1 = dataset.filter(lambda x: x["secondary_task"] == "T4.1 Global-Coverage Constrained Summary")

# 确认 token_length 的独特值
print("=== token_length 独特值 ===")
print(sorted(t4_1.unique("token_length")))
print()

# 看第一条样本
sample = t4_1[0]
ctx = sample["context"]
qn = sample["question_nonthinking"]

print(f"context : {ctx}")
print(f"question_nonthinking : {qn}")