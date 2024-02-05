**Data-Analysis-Chatbot**

I created this chatbot for the purposes of having a multi-functional backend for connecting various projects and files with OpenAI GPT. The code includes many features including:
- Ability to switch between multiple-models on-demand, for the purposes of cost savings (such as using gpt-3.5 turbo for intent identification compared to gpt-4-32k for data analysis).
- Intent engine that utilizes GPT-4 to determine the intent of a user's input, then directing to sub-agents (using Langchain) to allow for advanced functionality.
- Connects with files using Pandas and Langchain agents for advanced data analysis using a client's files.
