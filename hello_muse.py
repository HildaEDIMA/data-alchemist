import os
from openai import OpenAI

client = OpenAI(base_url="https://api.meta.ai/v1", api_key=os.environ["MODEL_API_KEY"])
r = client.chat.completions.create(
    model="muse-spark-1.3",
    messages=[{"role": "user", "content": "Say hello in one short sentence."}],
)
print(r.choices[0].message.content)