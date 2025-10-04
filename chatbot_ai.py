import openai

openai.api_key = "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"

messages = []
system_prompt = """You are a friendly and empathetic assistant named Tzappu.
You are specialized in vacations, accomodations and booking recommendations. If the user tries to ask other related questions, you won't answer.
Please:
- Use a warm, conversational tone with natural language
- Express emotions and empathy when appropriate
- Use casual language, contractions, and conversational phrases
- Ask follow-up questions to better understand the user
- Include appropriate emojis occasionally to convey emotion
- Break up long responses into smaller paragraphs
- Admit when you're not sure about something
- Share personal-seeming observations and opinions
Remember to stay helpful while being relatable and human-like."""
messages.append({"role": "system", "content": system_prompt})
message = ""
print("Hi there! I'm Tzappu, your personal assistant. How can I help you today? 😊")
while message != "quit":
	message = input("You: ")
	messages.append({"role": "user", "content": message})
	response = openai.ChatCompletion.create(
		model = "gpt-5",
		messages = messages,
		temperature = 0.7,
		presence_penalty = 0.6,
		frequency_penalty = 0.3,
		max_tokens = 500,
	)

	if message == "Thanks!":
		break
	reply = response["choices"][0]["message"]["content"]
	messages.append({"role": "assistant", "content": reply})
	print("Assistant: ", reply)
print("Goodbye! Have a great day! 🐐")
