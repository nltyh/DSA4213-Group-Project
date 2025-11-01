import evaluate
import nltk
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize

class ModelEvaluator:
    def __init__(self, df, response_col="response", golden_col="golden_response", query_col="user_query"):
        self.df = df.copy()
        self.response_col = response_col
        self.golden_col = golden_col
        self.query_col = query_col

        self.bertscore_metric = evaluate.load("bertscore")
    
    def bertEval(self, lang="en"):
        bertscore_results = self.bertscore_metric.compute(
            predictions=self.df[self.response_col].astype(str).tolist(),
            references=self.df[self.golden_col].astype(str).tolist(),
            lang=lang
        )
        self.df["BERTScore_F1"] = bertscore_results["f1"]
        avg_score = self.df["BERTScore_F1"].mean()
        return self.df[[self.query_col, "BERTScore_F1"]], avg_score

    def meteorEval(self):
        def compute_meteor(hypothesis: str, reference: str) -> float:
            hyp_tokens = word_tokenize(hypothesis)
            ref_tokens = word_tokenize(reference)
            return single_meteor_score(ref_tokens, hyp_tokens)

        self.df["METEOR"] = self.df.apply(
            lambda row: compute_meteor(str(row[self.response_col]), str(row[self.golden_col])),
            axis=1
        )
        avg_score = self.df["METEOR"].mean()
        return self.df[[self.query_col, "METEOR"]], avg_score