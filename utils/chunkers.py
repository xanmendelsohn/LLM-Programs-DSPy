import re

def read_german_abbreviations(path):
    abbreviations = []
    
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # Split the line into individual abbreviations
            abbrevs = line.strip().split()
            
            # Remove the final full stop from each abbreviation if present
            abbrevs = [abbrev[:-1] if abbrev.endswith('.') else abbrev for abbrev in abbrevs]
            
            # Add the cleaned abbreviations to the list
            abbreviations.extend(abbrevs)
    
    return abbreviations

class chunk_german_sentences:
    def __init__(self, abbreviations):
        self.abbreviations = abbreviations

    def __call__(self, text):
        # List of common German abbreviations (extend as needed)
        abbr_pattern = '|'.join(r'\b' + re.escape(abbr) + r'\.' for abbr in self.abbreviations)

        # Pattern for ordinal numbers
        ordinal_pattern = r'\d+\.'

        # Combine patterns
        exception_pattern = f"({abbr_pattern}|{ordinal_pattern})"

        # Split the text into potential sentences
        potential_sentences = re.split(r'([.!?]+)', text)

        sentences = []
        current_sentence = ""

        for i in range(0, len(potential_sentences) - 1, 2):
            current_sentence += potential_sentences[i].strip() + potential_sentences[i+1]

            # Check if the sentence ends with an exception
            if not re.search(exception_pattern + r'\s*$', current_sentence):
                yield current_sentence.strip() 
                current_sentence = ""

            # Add any remaining text as the last sentence
        if current_sentence:
            yield current_sentence.strip() 

class chunk_german_multi_sentences:
    ## to-do : can only handle overlap = 1
    def __init__(self, abbreviations, sentences_per_chunk=3, overlap=1):
        self.abbreviations = abbreviations
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def __call__(self, text):
        # List of common German abbreviations (extend as needed)
        abbr_pattern = '|'.join(r'\b' + re.escape(abbr) + r'\.' for abbr in self.abbreviations)

        # Pattern for ordinal numbers
        ordinal_pattern = r'\d+\.'

        # Combine patterns
        exception_pattern = f"({abbr_pattern}|{ordinal_pattern})"

        # Split the text into potential sentences
        potential_sentences = re.split(r'([.!?]+)', text)

        sentences = []
        current_sentence = ""
        chunk = ""

        for i in range(0, len(potential_sentences) - 1, 2):
            current_sentence += potential_sentences[i].strip() + potential_sentences[i+1]

            # Check if the sentence ends with an exception
            if not re.search(exception_pattern + r'\s*$', current_sentence):
                sentences.append(current_sentence.strip())
                current_sentence = ""

        # Add any remaining text as the last sentence
        if current_sentence:
            sentences.append(current_sentence.strip())
        
        for k in range(0, len(sentences) - self.sentences_per_chunk + 1, self.overlap):
            yield ' '.join(sentences[k:k + self.sentences_per_chunk])

def extract_output(text):
    # Find the position of 'Output: '
    output_start = text.find("Output: ")
    
    if output_start == -1:
        output_text = text
    else:
        # Extract everything after 'Output: '
        output_text = text[output_start + len("Output: "):].strip()
    
    return output_text

def extract_reasoning(text):
    # clean text 
    text = re.sub(r"\\n\d+\.", '', text)
    text = re.sub(r"\\n", '', text)
    # Find the position of 'Reasoning: '
    output_start = text.find("Reasoning: ")

    if output_start == -1:
        output_text = text
    else:
        # Extract everything after 'Reasoning: '
        output_text = text[output_start + len("Reasoning: "):].strip()
    
    return output_text

# def chunk_german_sentences(text, abbreviations):
#     # List of common German abbreviations (extend as needed)
#     abbr_pattern = '|'.join(r'\b' + re.escape(abbr) + r'\.' for abbr in abbreviations)

#     # Pattern for ordinal numbers
#     ordinal_pattern = r'\d+\.'

#     # Combine patterns
#     exception_pattern = f"({abbr_pattern}|{ordinal_pattern})"

#     # Split the text into potential sentences
#     potential_sentences = re.split(r'([.!?]+)', text)

#     sentences = []
#     current_sentence = ""

#     for i in range(0, len(potential_sentences) - 1, 2):
#         current_sentence += potential_sentences[i].strip() + potential_sentences[i+1]

#         # Check if the sentence ends with an exception
#         if not re.search(exception_pattern + r'\s*$', current_sentence):
#             sentences.append(current_sentence.strip())
#             current_sentence = ""

#     # Add any remaining text as the last sentence
#     if current_sentence:
#         sentences.append(current_sentence.strip())

#     return sentences

