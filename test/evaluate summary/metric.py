from deepeval.scorer import Scorer
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase):
        self.score = self.scorer.rouge_score(
            prediction=test_case.actual_output,
            target=test_case.expected_output,
            score_type="rouge1"  # 可选 "rouge1" / "rouge2" / "rougeL"
        )
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"


from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ROUGE
rouge_metric = RougeMetric(threshold=0.5)
# 摘要质量（LLM judge）
summ_metric = SummarizationMetric(threshold=0.5)

for r in results:
    test_case = LLMTestCase(
        input=r["context"],
        actual_output=r["pred"],
        expected_output=r["answer"]  # ROUGE 需要
    )
    
    rouge_metric.measure(test_case)
    summ_metric.measure(test_case)
    
    print(f"ROUGE: {rouge_metric.score}, Summary: {summ_metric.score}")