import dspy  
from utils.chunkers import extract_output, extract_reasoning

class ScanReport(dspy.Module):

    def __init__(self, hint, ChunkerSignature, PredictRelevance, PredictCategory, Translator):
        super().__init__()

        self.ChunkerSignature = ChunkerSignature
        self.PredictRelevance = PredictRelevance
        self.PredictCategory = PredictCategory
        self.Translator = Translator
        self.hint = hint

        # preselect relevant snippets
        self.preselection = dspy.Predict(self.ChunkerSignature)
        # given a snippet, predict a list of relevant categories using a CoT
        self.predict = dspy.ChainOfThoughtWithHint(self.PredictCategory)
        # given an annual report snippet, rate relevance to information extraction
        self.relevance = dspy.ChainOfThought(self.PredictRelevance)
        # reduce the number of extracted infos
        self.translator = dspy.Predict(self.Translator)

    def forward(self, context):
        
        answers = []
        reasoning = []
        
        preselection = self.preselection(context=context)

        # for each chunk in the preselection
        for snippet in [item.replace('"', '') for item in extract_output(preselection.output).split('", "')]:
            # use the LM to predict relevant products
            chunk_categories = self.predict(context=[snippet], hint=self.hint)

            chunk_relevance = self.relevance(context=[snippet])

            entry = {
                "quote": snippet,
                "relevance_score": chunk_relevance.output, 
                "categories": [item.strip() for item in chunk_categories.output.split(',')],
                "reasoning_categories": self.translator(context = extract_reasoning(chunk_categories.rationale)).output,
                "reasoning_relevance": self.translator(context = extract_reasoning(chunk_relevance.rationale)).output 
                }
            
            answers.append(entry)

        return dspy.Prediction(context=context, answer=answers)