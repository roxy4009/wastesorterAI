from openai import OpenAI
import os


client = OpenAI(
  api_key=os.getenv("OPENAI_API_KEY")
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
