import dspy    

def init_signatures(type_of_documents, number_of_items_in_output, objective, class_of_categories, categories):

    HINT = "Valid {class_of_categories} are:"
    hint = f"{HINT} {', '.join(categories)}."

    class ChunkerSignature(dspy.Signature):
        __doc__ = f"""Given a {type_of_documents}, determine {number_of_items_in_output} most relevant snippets (2-3 sentences) to {objective}. Do not include the context in the output."""
        context = dspy.InputField()
        output = dspy.OutputField(desc="comma-separated quotes")

    class PredictCategory(dspy.Signature):
        print(categories)
        __doc__ = f"""Given a snippet from a {type_of_documents}, identify which of the {class_of_categories} ({categories}) the snippet is relevant to. If snippet is not relevant for any {class_of_categories}, say 'other'."""
        #  __doc__ = f"""Given a snippet from a {type_of_documents}, identify which of the {class_of_categories} ({', '.join(categories)}) the snippet is relevant to. If snippet is not relevantv for any {class_of_categories}, say 'other'."""
       
        context = dspy.InputField()
        output = dspy.OutputField(desc="comma-separated {class_of_categories}", format=lambda x: ', '.join(x) if isinstance(x, list) else x)

    class PredictRelevance(dspy.Signature):
        __doc__ = f"""Given a snippet from a {type_of_documents}, determine a score between 0 and 100 of how relevant the snippet is to {objective}. A score of 100 denotes high relevance, and a score 0 denotes irrelevance."""
        context = dspy.InputField()
        output = dspy.OutputField(desc="number between 0 and 100")

    class Translator(dspy.Signature):
        __doc__ = f"""Do not include the context and introduction in the output. Translate to German."""
        context = dspy.InputField()
        output = dspy.OutputField(desc="German")

    return hint, ChunkerSignature, PredictRelevance, PredictCategory, Translator


    class SummaryReasoning(dspy.Signature):
        __doc__ = f"""Given a resoning text, summarize it. Translate to German."""
        context = dspy.InputField()
        output = dspy.OutputField(desc="German")






