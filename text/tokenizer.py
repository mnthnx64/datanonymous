import spacy
import random

class Tokenizer:
    nlp = spacy.load('en_core_web_lg')
    input_str = None
    tokens = None
    city_list = ['Perrsia','Australia','Armenia','Russia','Sweden','Switzerland','India','LosAngels','Egypt']
    final_map = dict()
    
    def __init__(self, string):
        self.input_str = string
        self.tokens = self.nlp(self.input_str)
        
    def getLocations(self):
        self.locations = self.tokens.ents
        return self.locations
        
    def doSomething(self):
        self.split_str = self.input_str.split(' ')
        print(self.split_str)
        m = list()
        for loc in self.locations:
            x = str(loc)
            m.append(x.split(' '))

        for cities in m:
            idx = self.split_str.index(cities[0])
            if len(cities) == 1:
                replace_val = random.choice(self.city_list)
                print(replace_val)
                self.final_map[replace_val] = self.split_str[idx]
                self.split_str[idx] = replace_val
            else:
                end_idx = self.split_str.index(cities[-1])
                replace_val = random.choice(self.city_list)
                self.final_map[replace_val] = self.split_str[idx:end_idx+1]
                self.split_str[idx] = replace_val
                for i in range(idx+1,end_idx+1):
                    self.split_str.remove(self.split_str[i])
    
    def getFinal(self):
        return self.split_str

    def getJson(self):
        return self.final_map
        
