import pandas as pd
import dspy
from utils.chunkers import extract_output, extract_reasoning

def flatten_dict(res: list):
    result_dict = {}
    for i, elem in enumerate(res):
        result_dict[i] = elem

    return result_dict
    
def dict_to_df(result_dict: dict):
    df = pd.DataFrame.from_dict(result_dict)
    df = df.transpose()

    return df

class SummaryReasoning(dspy.Signature):
    __doc__ = f"""Given a resoning text, summarize and make a meningful conclusion. Do not include the context in the output. Remove introductions and comments."""
    context = dspy.InputField()
    output = dspy.OutputField(desc="German")
    
# function to aggregate results per category
def aggregate_per_category(result_list: list, categories) -> dict:
    """This function aggregates the output results per category. For the relevance score the maximum is taken.
    input:
    df : pd.DataFrame
        Dictionary in json format
    categories
        a list of the categories specified by the user
    output:
    df_agg:
        pd.DataFrame with the aggregated results per category
    """

    result_dict = flatten_dict(result_list) # flatten
    df = dict_to_df(result_dict) # convert to df

    # initialize df_agg
    df_agg=pd.DataFrame(columns = df.columns)

    # Loop through categpries
    for item in categories:
        matched = False
        quote = ""
        relevance_score = 0
        reasoning_categories = ""
        reasoning_relevance = ""
        n=0
        for i in range(len(df)):
            # add results if there are some matching
            if item in df.loc[i].categories:
                n += 1
                matched = True
                relevance_score = max(relevance_score, int(df.loc[i].relevance_score))
                quote = f"{quote}# Quote {n}: {str(df.loc[i].quote)}"
                reasoning_categories = df.loc[i].reasoning_categories
                reasoning_relevance = df.loc[i].reasoning_relevance

        if matched:
            new_row = {'quote': quote, 'relevance_score': relevance_score, 'categories': item, 'reasoning_categories': reasoning_categories, 'reasoning_relevance': reasoning_relevance}
            df_agg = pd.concat([df_agg, pd.DataFrame([new_row])], ignore_index=True)

    # sort values descending by relevance_score
    df_agg = df_agg.sort_values(by=['relevance_score'], ascending=False)


    summary_reasoning_categories = []
    summary_reasoning_relevance = []

    summarize_reasoning = dspy.Predict(SummaryReasoning)
    for i in range(len(df_agg)):
        summary_reasoning_categories.append(extract_output(summarize_reasoning(context = df_agg.loc[i]['reasoning_categories']).output))
        summary_reasoning_relevance.append(extract_output(summarize_reasoning(context = df_agg.loc[i]['reasoning_relevance']).output))
    
    df_agg['reasoning_categories'] = summary_reasoning_categories
    df_agg['reasoning_relevance'] = summary_reasoning_relevance
    
    dict_agg = df_agg.transpose().to_dict() # convert to dict
    
    return dict_agg
