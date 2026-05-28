"""
W15-D3 Pipeline 推理实战
========================
使用 Pipeline API 进行多种 NLP 任务推理:
sentiment-analysis, text-classification, ner,
question-answering, text-generation, zero-shot-classification
"""

import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("W15-D3 Pipeline 推理实战")
print("=" * 60)

from transformers import pipeline

# ============================================================
# 1. 情感分析 (Sentiment Analysis)
# ============================================================
print("\n--- 1. 情感分析 (Sentiment Analysis) ---")
print("  Pipeline: 'sentiment-analysis'")
print("  模型: distilbert-base-uncased-finetuned-sst-2-english\n")

sentiment = pipeline("sentiment-analysis")

texts = [
    "I absolutely love this product! Best purchase ever!",
    "Terrible experience, would not recommend to anyone.",
    "The movie was okay, nothing special but not bad either.",
    "HuggingFace makes it incredibly easy to use state-of-the-art NLP models!",
]

for text in texts:
    result = sentiment(text)[0]
    label_cn = "正面" if result["label"] == "POSITIVE" else "负面"
    print(f"  文本: {text}")
    print(f"  结果: {label_cn} ({result['label']}), 置信度: {result['score']:.4f}")
    print()

# 批量推理
print("  批量推理:")
results = sentiment(texts)
for text, res in zip(texts, results):
    print(f"    {res['label']:10s} ({res['score']:.3f}) ← {text[:50]}")

# ============================================================
# 2. 文本分类 (Text Classification)
# ============================================================
print("\n--- 2. 文本分类 (Text Classification) ---")
print("  Pipeline: 'text-classification'")
print("  模型: bert-base-uncased (可自定义)\n")

# 使用情感分析 pipeline 演示多标签
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    top_k=None  # 返回所有标签的分数
)

text = "The new smartphone has amazing features but battery life is disappointing."
scores = classifier(text)[0]
print(f"  文本: {text}")
print(f"  所有类别得分:")
for s in scores:
    label_cn = "正面" if s["label"] == "POSITIVE" else "负面"
    print(f"    {s['label']:10s} ({label_cn}): {s['score']:.4f}")

# ============================================================
# 3. 命名实体识别 (NER)
# ============================================================
print("\n--- 3. 命名实体识别 (Named Entity Recognition) ---")
print("  Pipeline: 'ner'")
print("  模型: dbmdz/bert-large-cased-finetuned-conll03-english\n")

ner = pipeline("ner", grouped_entities=True)

ner_text = "Apple was founded by Steve Jobs in Cupertino, California in 1976. Tim Cook is the current CEO."
entities = ner(ner_text)

print(f"  文本: {ner_text}")
print(f"  识别到的实体:")
for ent in entities:
    entity_type_map = {
        "PER": "人名", "PERSON": "人名",
        "ORG": "组织", "LOC": "地点",
        "DATE": "日期", "MISC": "其他"
    }
    type_cn = entity_type_map.get(ent["entity_group"], ent["entity_group"])
    print(f"    {ent['word']:20s} → {ent['entity_group']:8s} ({type_cn}), 置信度: {ent['score']:.4f}")

# ============================================================
# 4. 问答系统 (Question Answering)
# ============================================================
print("\n--- 4. 问答系统 (Question Answering) ---")
print("  Pipeline: 'question-answering'")
print("  模型: distilbert-base-cased-distilled-squad\n")

qa = pipeline("question-answering")

context = """
HuggingFace is a company that develops tools for building applications
using machine learning. It is most notable for its Transformers library
built for natural language processing applications and its platform that
allows users to share machine learning models and datasets. The company
was founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf
in New York City.
"""

qa_pairs = [
    ("What is HuggingFace known for?", "HuggingFace以什么闻名?"),
    ("When was HuggingFace founded?", "HuggingFace何时成立?"),
    ("Who founded HuggingFace?", "谁创立了HuggingFace?"),
    ("Where is HuggingFace headquartered?", "HuggingFace总部在哪里?"),
]

for question, question_cn in qa_pairs:
    result = qa(question=question, context=context)
    print(f"  问题: {question}")
    print(f"        ({question_cn})")
    print(f"  答案: {result['answer']}")
    print(f"  置信度: {result['score']:.4f}, 位置: [{result['start']}:{result['end']}]")
    print()

# ============================================================
# 5. 文本生成 (Text Generation)
# ============================================================
print("\n--- 5. 文本生成 (Text Generation) ---")
print("  Pipeline: 'text-generation'")
print("  模型: gpt2 (OpenAI GPT-2)\n")

try:
    generator = pipeline("text-generation", model="gpt2")

    prompts = [
        "Artificial intelligence will transform",
        "The future of natural language processing is",
        "In the next decade, machine learning",
    ]

    for prompt in prompts:
        result = generator(
            prompt,
            max_new_tokens=40,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=50256
        )
        generated = result[0]["generated_text"]
        print(f"  Prompt: {prompt}")
        print(f"  生成:   {generated}")
        print()

except Exception as e:
    print(f"  [!] 文本生成加载失败: {e}")
    print("  需要安装: pip install transformers[torch]")

# ============================================================
# 6. 零样本分类 (Zero-Shot Classification)
# ============================================================
print("\n--- 6. 零样本分类 (Zero-Shot Classification) ---")
print("  Pipeline: 'zero-shot-classification'")
print("  模型: facebook/bart-large-mnli\n")

try:
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    text = "The new electric car has a range of 400 miles and charges in just 30 minutes."
    candidate_labels = ["technology", "sports", "politics", "automotive", "health", "finance"]

    result = zero_shot(text, candidate_labels)
    print(f"  文本: {text}")
    print(f"  候选标签: {candidate_labels}")
    print(f"  分类结果:")
    for label, score in zip(result["labels"], result["scores"]):
        bar = "█" * int(score * 30)
        print(f"    {label:15s}: {score:.4f} {bar}")

    # 第二个例子
    text2 = "The stock market crashed today amid concerns about inflation and interest rates."
    result2 = zero_shot(text2, candidate_labels)
    print(f"\n  文本: {text2}")
    print(f"  分类结果:")
    for label, score in zip(result2["labels"], result2["scores"]):
        bar = "█" * int(score * 30)
        print(f"    {label:15s}: {score:.4f} {bar}")

except Exception as e:
    print(f"  [!] 零样本分类加载失败: {e}")
    print("  跳过此部分")

# ============================================================
# 7. 总结
# ============================================================
print("\n--- 7. 总结 ---")
print("""
  本节学习了 6 种 Pipeline 推理任务:
  1) sentiment-analysis   - 情感分析 (正面/负面)
  2) text-classification  - 文本分类 (可自定义标签)
  3) ner                  - 命名实体识别 (人名/地名/组织)
  4) question-answering   - 抽取式问答 (从上下文中找答案)
  5) text-generation      - 文本生成 (GPT-2)
  6) zero-shot-classification - 零样本分类 (无需训练数据)

  Pipeline 三行代码即可使用:
    from transformers import pipeline
    pipe = pipeline("任务名", model="模型名")
    result = pipe("输入文本")

  下一步: d4_text_classification.py - 文本分类微调实战
""")
