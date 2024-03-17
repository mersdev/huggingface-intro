from transformers import pipeline

classfier = pipeline("sentiment-analysis")
res = classfier("I Love You!")
print(res)