import re
import dspy


def substring_metric(expected: dspy.Example, pred: dspy.Prediction, trace=None) -> int:
    """Validation metric based on string comparison and regex. 

    Parameters
    ----------
    expected : dspy.Example
        Expected/example (target) data
    pred: dspy.Prediction
        Predicted data
    trace
        If None a score betwen 0 and 1 is returned, else True or False

    Returns
    -------
    int/boolean
        int: score between 0 and 1 if trace=None
        boolean: if trace!=None  
    """

    final_score = 0.0

    ## gather quotes and categories
    pred_quotes = [text_preprocessing(item['quote']) for item in pred.answer]
    pred_relevance = [item['relevance_score'] for item in pred.answer]
    pred_categories = [item['categories'] for item in pred.answer]
    
    expected_quotes = ''.join([text_preprocessing(item['quote']) for item in expected.answer]) # as we already store all quotes in one string
    expected_relevance = float([item['relevance_score'] for item in expected.answer][0])
    expected_count_quote = float([item['count_quote'] for item in expected.answer][0])
    expected_categories = [item['categories'] for item in expected.answer]

    quote_match_score = 0.0
    categories_match_score = 0.0

    # if there are quotes in the target data
    if expected_quotes != '':
        quote_match_res = quote_match(expected_quotes, expected_count_quote, pred_quotes, pred_relevance)
        quote_match_score = quote_match_res[0]
        quote_match_indexes = quote_match_res[1]

        if len(quote_match_indexes)>0:
            categories_match_score, categories_match_size = categories_match(expected_categories,pred_categories, quote_match_indexes) 
        else: # we assume that expected quotes are exist, but was not matched with the any predicted quotes (llm results)
            # print('The quotes match was not found, but the example quote(s) exist(s)')
            # look for every predicted category
            categories_match_score, categories_match_size = categories_match(expected_categories,pred_categories, range(0,len(pred_categories)))
    # else:
        # print('The example quote is not found in the target data. No validation for that example.')
    
    # compute final score 
    final_score = (quote_match_score+categories_match_score)/2
    final_score = min(1.0, final_score)

    # add trace for not None (boolean)
    if trace != None:
        final_score = (quote_match_score > 0.3) & (categories_match_score > 0.3)

    return final_score


# LLM as Judge Metric 
# Define metric as softer measure for the reasoning

def lm_metric(expected: dspy.Example, pred: dspy.Prediction, trace=None) -> int:
    """Validation metric based on LLM as a Judge for quotes and 
    string comparison for categories. 

    Parameters
    ----------
    expected : dspy.Example
        Expected/example (target) data
    pred: dspy.Prediction
        Predicted data
    trace
        If None a score betwen 0 and 1 is returned, else True or False

    Returns
    -------
    int/boolean
        int: score between 0 and 1 if trace=None
        boolean: if trace!=None  
    """
    
    ## gather quotes and categories
    pred_quotes = [item['quote'] for item in pred.answer]
    pred_relevance = [item['relevance_score'] for item in pred.answer]
    pred_categories = [item['categories'] for item in pred.answer]
    
    expected_quotes = ''.join([item['quote'] for item in expected.answer]) # as we already store all quotes in one string
    expected_relevance = float([item['relevance_score'] for item in expected.answer][0])
    expected_count_quote = float([item['count_quote'] for item in expected.answer][0])
    expected_categories = [item['categories'] for item in expected.answer]

    # initialize scores
    final_score = 0.0
    quote_match_score = 0.0
    categories_match_score = 0.0

    # if there are quotes in the target data
    if expected_quotes != '':

        # quotes
        quote_match_res = quote_match_lm(expected_quotes, expected_count_quote, pred_quotes, pred_relevance)
        quote_match_score = quote_match_res[0]
        quote_match_indexes = quote_match_res[1]

        # categories
        if len(quote_match_indexes)>0:
            categories_match_score, categories_match_size = categories_match(expected_categories,pred_categories, quote_match_indexes) 
        else: # we assume that expected quotes are exist, but was not matched with the any predicted quotes (llm results)
            # print('The quotes match was not found, but the example quote(s) exist(s)')
            # look for every predicted category
            categories_match_score, categories_match_size = categories_match(expected_categories,pred_categories, range(0,len(pred_categories)))
        # if len(quote_match_indexes)>0:
        #     categories_match_score = categories_match_lm(expected_categories, pred_categories, quote_match_indexes) 

    final_score = (quote_match_score+categories_match_score)/2

    # add trace for not None (boolean)
    if trace != None:
        final_score = (quote_match_score > 0.3) & (categories_match_score > 0.3)

    return final_score


def text_preprocessing(text: str) -> str:
    """Method preprocessed the text: remove punctuation marks, spaces and capital letters

    Parameters
    ----------
    text : str
        The text needed to be preprocessed

    Returns
    -------
    str
        Preprocessed string (no punctuation marks, only lower case, no spaces)
    """
    text = re.sub(r"\W+", "", text)
    text = text.lower()
    text = text.replace("[^a-zA-Z]", "")
    return text


def quote_match(
    expected_quotes: str,
    expected_count_quote: int,
    pred_quotes: list,
    pred_relevance: list,
) -> (int, list):
    """Check the existence of predicted quote in expected quote or vice versa.
    If match exists, then calculates the match-score based on the relevance and amount of predicted quotes:
    sum(pred_relevance)/number_of_quotes

    Parameters
    ----------
    expected_quotes : str
        Expected/example (target) quotes
    expected_count_quote : int
        Number of expected/example (target) quotes
    pred_quotes : list(str)
        Predicted list of quotes (by LLM). The default size is 5
    pred_relevance : list(float)
        Predicted list of relevance for each quote (by LLM)

    Returns
    -------
    int
        quote_match_score calculated as sum of relevance for every pred_quote existing in the expected_quote
        divided by number of all expected quote and 100
    list
        array of indexes of predicted quotes which were matched  with expected quote
    """
    # if we have a target quote for the specific example
    quote_match_score = 0.0
    pred_match_index = []

    for i, quote in enumerate(pred_quotes):
        try:
            pred_score = float(pred_relevance[i]) / 100
        except:
            pred_score = 0.1
        if (quote in expected_quotes) | (expected_quotes in quote):
            quote_match_score += pred_score
            pred_match_index.append(i)

    quote_match_score = quote_match_score / expected_count_quote

    return quote_match_score, pred_match_index


def categories_match(
    expected_categories: list, pred_categories: list, quote_match_indexes: list
) -> (int, list):
    """Check the existence of predicted categories in expected categories.
    If match exists, then calculates the match-score based on the number of matched categories
    and overall number of expected categories:
    sum(count_categories)/expected_categories_size

    Parameters
    ----------
    expected_categories : list(str)
        Expected/example (target) categories
    pred_categories : list(str)
        Predicted list of categories (by LLM)
    quote_match_indexes : list(int)
        The indexes of pred quotes matched with expected quotes

    Returns
    -------
    int
        categories_match_score calculated as sum of amount of pred categories exists in expected categories list
        divided by number of all expected categories
    list
        how many of predicted categories matched with expected categories for every quote
    """
    # extract categories for matched quotes
    pred_categories_matched_list = [
        pred_categories[index] for index in quote_match_indexes
    ]
    pred_categories_matched = [
        text_preprocessing(item) for row in pred_categories_matched_list for item in row
    ]  # flatten the array

    expected_categories = [
        item for row in expected_categories for item in row
    ]  # flatten the array
    expected_categories_size = 0
    count_categories = []

    # as expected categories are list of lists
    for category_list in expected_categories:
        # remove brackets and string marks for list in list
        category_list = re.sub(r'["]', "", category_list).split(",")
        # count the size of current list of categories
        expected_categories_size += len(category_list)
        # preprocess the elements in list
        category_list = [text_preprocessing(item) for item in category_list]
        # sum 1s if there is a match in expected and predicted categories
        match_score = sum(
            1 for item in category_list if item in pred_categories_matched
        )
        count_categories.append(match_score)

    categories_match_score = sum(count_categories) / expected_categories_size

    return categories_match_score, count_categories


class AssessReasoning(dspy.Signature):
    """Assess whether the for the given quote the reasoning for the relevance score (0 is not relevant, 100 is most relevant) is convincing and persuasive.
    Answer with a score between 0 and 100.
    A score of 100 denotes a good reasoning, and a score 0 denotes a bad reasoning. Be critical!
    """

    quote = dspy.InputField()
    reasoning = dspy.InputField()
    relevance = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="number between 0 and 100")


def relevance_reasoning_lm(
    pred_quotes: list, pred_reasoning_relevance: list, pred_relevance: list, lm_gpt
):
    """Assess the reasoning of the relevance with a LLM.

    Parameters
    ----------
    pred_quotes : list
        Predicted list of quotes (by LLM). The default size is 5
    pred_reasoning_relevance : list
        Predicted list of relevance reasoning for each quote (by LLM)
    pred_relevance : list
        Predicted list of relevance for each quote (by LLM)
    lm_gpt
        initialized LLM

    Returns
    -------
    num
        reasoning score
    """

    score = 0.0

    for i, reasoning in enumerate(pred_reasoning_relevance):
        quote = pred_quotes[i]
        relevance = pred_relevance[i]
        with dspy.context():
            assess_reason = dspy.Predict(AssessReasoning)
            contained = assess_reason(
                quote=quote, reasoning=reasoning, relevance=relevance
            )
            try:
                assess = float(contained.assessment_answer) / 100
            except:
                assess = 0.5
            score += assess
            # if 'true' in contained.assessment_answer.lower():
            #     score += 1

    score = score / len(pred_reasoning_relevance)

    return score


class AssessQuote(dspy.Signature):
    """Assess whether the information given in the predicted quote is a relavant part of the expected quote."""

    predicted_quote = dspy.InputField()
    expected_quote = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="true or false")


def quote_match_lm(
    expected_quotes: str,
    expected_count_quote: int,
    pred_quotes: list,
    pred_relevance: list,
) -> (int, list):
    """Assess whether the information of the predicted quote matches the expected quote with a LLM.
    If match exists, then calculates the match-score based on the relevance and amount of predicted quotes:
    sum(pred_relevance)/number_of_quotes

    Parameters
    ----------
    expected_quotes : str
        Expected/example (target) quotes
    expected_count_quote : int
        Number of expected/example (target) quotes
    pred_quotes : list(str)
        Predicted list of quotes (by LLM). The default size is 5
    pred_relevance : list(float)
        Predicted list of relevance for each quote (by LLM)
    lm_gpt
        initialized LLM

    Returns
    -------
    int
        quote_match_score calculated as sum of relevance for every pred_quote existing in the expected_quote
        divided by number of all expected quote and 100
    list
        array of indexes of predicted quotes which were matched with expected quote
    """
    # if we have a target quote for the specific example
    quote_match_score = 0.0
    pred_match_index = []
    # print(expected_quotes)

    for i, quote in enumerate(pred_quotes):
        try:
            pred_score = float(pred_relevance[i]) / 100
        except:
            pred_score = 0.1
        with dspy.context():
            assess_quote = dspy.Predict(AssessQuote)
            contained = assess_quote(
                predicted_quote=quote, expected_quote=expected_quotes
            )
            if "true" in contained.assessment_answer.lower():
                quote_match_score += pred_score
                pred_match_index.append(i)
                # print(contained.assessment_answer)
                # print(quote)

    quote_match_score = min(1.0, quote_match_score / expected_count_quote)

    return quote_match_score, pred_match_index


class AssessCategories(dspy.Signature):
    __doc__ = """Assess whether the predicted categories are contained in the example categories."""

    predicted_categories = dspy.InputField()
    example_categories = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="true or false")


def categories_match_lm(
    expected_categories: list, pred_categories: list, quote_match_indexes: list, lm_gpt
) -> (int, list):
    """Check the existence of predicted categories in expected categories with a LLM.
    If match exists, then calculates the match-score based on the number of matched categories
    and overall number of expected categories:
    sum(count_categories)/expected_categories_size

    Parameters
    ----------
    expected_categories : list(str)
        Expected/example (target) categories
    pred_categories : list(str)
        Predicted list of categories (by LLM)
    quote_match_indexes : list(int)
        The indexes of pred quotes matched with expected quotes
    lm_gpt
        initialized LLM

    Returns
    -------
    int
        categories_match_score calculated as sum of amount of pred categories exists in expected categories list
        divided by number of expected categories
    """
    # print(expected_categories)
    # print(pred_categories)
    # print(quote_match_indexes)
    # extract categories for matched quotes
    pred_categories_matched_list = [
        pred_categories[index] for index in quote_match_indexes
    ]
    pred_categories_matched = [
        item for row in pred_categories_matched_list for item in row
    ]  # flatten the array
    # print(pred_categories_matched)
    expected_categories = [
        item for row in expected_categories for item in row
    ]  # flatten the array
    # print(expected_categories)
    expected_categories_size = len(expected_categories)
    count_categories = []

    # as expected categories are list of lists
    for categories_str in expected_categories:
        # sum 1s if there is a match in expected and predicted categories
        for item in pred_categories_matched:
            # print(item)
            match_score = 0
            assess_categories = dspy.Predict(AssessCategories)
            contained = assess_categories(
                predicted_categories=item, example_categories=categories_str
            )
            # print("cats")
            # print(contained.assessment_answer)
            if "true" in contained.assessment_answer.lower():
                match_score += 1
        count_categories.append(match_score)
    # print(count_categories)
    # categories_match_score = sum(count_categories)/max(1, expected_categories_size)
    categories_match_score = sum(count_categories) / max(1, expected_categories_size)
    return categories_match_score


def full_lm_metric(expected: dspy.Example, pred: dspy.Prediction, trace=None) -> int:
    """Validation metric based purely on LLM as a Judge for 
     quotes, categories and reasoning. 

    Parameters
    ----------
    expected : dspy.Example
        Expected/example (target) data
    pred : dspy.Prediction
        Predicted data
    trace
        If None a score betwen 0 and 1 is returned, else True or False

    Returns
    -------
    int/boolean
        int: score between 0 and 1 if trace=None
        boolean: if trace!=None  
    """

    ## gather quotes and categories
    pred_quotes = [item['quote'] for item in pred.answer]
    pred_relevance = [item['relevance_score'] for item in pred.answer]
    pred_categories = [item['categories'] for item in pred.answer]
    pred_reasoning_relevance = [item['reasoning_relevance'] for item in pred.answer]

    expected_quotes = ''.join([item['quote'] for item in expected.answer]) # as we already store all quotes in one string
    expected_relevance = float([item['relevance_score'] for item in expected.answer][0])
    expected_count_quote = float([item['count_quote'] for item in expected.answer][0])
    expected_categories = [item['categories'] for item in expected.answer]

    # initialize scores
    final_score = 0.0
    quote_match_score = 0.0
    categories_match_score = 0.0

    reasoning_relevance_score = relevance_reasoning_lm(pred_quotes, pred_reasoning_relevance, pred_relevance, lm_gpt)

    # if there are quotes in the target data
    if expected_quotes != '':

        quote_match_res = quote_match_lm(expected_quotes, expected_count_quote, pred_quotes, pred_relevance)
        quote_match_score = quote_match_res[0]
        quote_match_indexes = quote_match_res[1]

        if len(quote_match_indexes)>0:
            categories_match_score = categories_match_lm(expected_categories, pred_categories, quote_match_indexes, lm_gpt) 

    else:
        # print('The example quote is not found in the target data. No validation for that example.')
        quote_match_score = 0.0

        for pred_item in pred_relevance:
            # print(pred_item)
            try:
                pred_item = (100 - float(pred_item))/100
            except:
                pred_item = 0.5
            quote_match_score = quote_match_score + pred_item   
        
        quote_match_score = quote_match_score/len(pred_relevance)
    
    final_score = ((2*quote_match_score) + (2*categories_match_score) + reasoning_relevance_score) / 5

    # add trace for not None (boolean)
    if trace != None:
        final_score = (quote_match_score > 0.3) & (categories_match_score > 0.3) & (reasoning_relevance_score > 0.3)

    return final_score
