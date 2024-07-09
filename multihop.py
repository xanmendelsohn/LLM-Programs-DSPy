import dspy

class MultiHop(dspy.Module):
    def __init__ (self, passages_per_hop):
        #self.retrieve = dspy.Retrieve(k = passages_per_hop)
        self.generate_query = dspy.ChainOfThought("context, question -> search_query")
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
        
    def forward(self, context, question):
        deep_context = []
    
        for hop in range(2):
            query = self.generate_query(context = context, question = question).search_query
            deep_context += self.retrieve(query).passages
    
        return self.generate_answer(context = deep_context, question = question)
    
    #multihop = MultiHop(passages_per_hop =3)
