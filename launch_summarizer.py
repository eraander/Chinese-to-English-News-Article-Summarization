from newspaper import Article
from transformers import pipeline
import gradio as gr

def analyzeArticle(url):
    a = Article(url, language='zh')
    a.download()
    a.parse()
    translator = pipeline("translation_zh_to_en", model='Helsinki-NLP/opus-mt-zh-en', max_length=50)
    translation = translator(a.text)
    summarizer = pipeline("summarization")
    summary = summarizer(translation[0]['translation_text'])
    return summary[0]['summary_text']

demo = gr.Interface(
    fn=analyzeArticle,
    inputs=["text"],
    outputs=[gr.Textbox(label="summary", lines=2)],
)

demo.launch()