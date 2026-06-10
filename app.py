from openai import OpenAI
# from dotenv import api_key
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

completion = client.chat.completions.create(
  model="openai/gpt-oss-20b",
  messages=[{"role":"user","content":"Provide me an article on machine learning"}],
  temperature=1,
  top_p=1,
  max_tokens=4096,
  stream=False
)

reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
if reasoning:
  print(reasoning)
print(completion.choices[0].message.content)