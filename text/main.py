from tokenizer import Tokenizer
from coder import Coder

model = Tokenizer("the tribal people in the southern part of asia have set up an NGO and it helps many women in Afghanistan")
model.getLocations()
model.doSomething()
print("Coded = " + (' ').join(model.getFinal()))
plotter = Coder(model.getJson())
plotter.getFormat()
