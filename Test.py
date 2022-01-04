from transformers import pipeline
print(pipeline("sentiment-analysis")("I hate you"))