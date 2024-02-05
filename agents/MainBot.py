import os
import time
import openai #OpenAI Libraries
import pandas as pd #Pandas - Dataframes
from dotenv import load_dotenv #Utilizes env file for variables

import tiktoken #Count tokens
from typing import Union

from datetime import datetime

##Langchain Libraries
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
#from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent


class MainBot:

    def __init__(self, history=None, system_message=None, file_df=None, log_text=None):
        print('__________________INIT CALLED_________________________')
        self.base_path = os.path.join(os.getcwd(),  "agents")
        self.base_env_path = os.path.join(self.base_path, "envs")
        self.base_log_path = os.path.join(self.base_path, "logs")
        #self.file_path = os.path.join(self.base_path, "files", "Data Analysis.csv")
        self.file_path = r"https://azure.blob.core.windows.net/ai_demo/Data Analysis.csv".replace(" ", "%20")
        self.start_timestamp = self.get_timestamp()
        self.start_timestamp_text = self.get_file_timestamp()
        
        if system_message is None: #if initial run, instantiate the variables
            self.message_history = []
            self.system_message = ""
            self.df = self.load_file(self.file_path)
            self.log_text = "\nRun Start Time: " + self.start_timestamp
        else: #otherwise, pull the existing from the function call
            self.message_history = history
            self.system_message = system_message
            self.df = file_df
            self.log_text = log_text

        self.current_intent = ""
        self.current_env = "" #Current environment file being used (GPT-4-32k or GPT-4) 
        self.prior_env = "" #
        self.prior_requested_model = ""
        self.dataframe_tokens = 0
        self.load_env_variables()
        self.load_intent_system_prompts()
    
    def load_env_variables(self, use_gpt4_32k=True):
        print(self.base_path)
        self.current_env = os.path.join(self.base_env_path, "gpt-4-32k.env" if use_gpt4_32k else "gpt-35-turbo.env")
        print('Env path: ', self.current_env)
        load_dotenv(self.current_env)

        self.tiktoken_model = os.getenv("MODEL_NAME")
        print('Model name: ', self.tiktoken_model)

        self.max_tokens = int(os.getenv("MAX_TOKENS"))-50 #adds a padding of 50 tokens from the max
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        AzureChatOpenAI.deployment_name = os.getenv("DEPLOYMENT_NAME")
        print('Deployment name: ',  os.getenv("DEPLOYMENT_NAME"))
        self.remaining_tokens = self.max_tokens
        #print("Remaining tokens:",self.remaining_tokens)
        self.total_tokens = 0
        #print("Total tokens:",self.total_tokens)

    def load_intent_system_prompts(self):

        self.df_cleanse_sys=r"""Hello. Your job is to clean up the data in this dataframe. Please return a dataframe that has the same data, but the dataframe is cleaned up, including trimming extra whitespace, and ensure that all numerical columns can be properly summed (while fixing errors if any exist)."""

        self.user_message_intent_sys=r"""Hello! Your job is to read this prompt from a human being sent to a computer to find what the user wants the computer to do given the context. You must choose a single intent from the list provided below. Please respond with only one option with no additional context or feedback. For example, if you believe the user's dataset should be analyzed, please respond with 'AnalyzeDataFrame' without quotes. If you believe that the user wants to draw a chart based on the dataset provided, please respond with 'DrawChart' without quotes. You should only respond with an intent if you believe the user's latest query is related to the intent. If you are not sure, you can say 'Other', or, 'UnsupportedRequest'.

Intent List:
0) "UnsupportedRequest" - You can say this if you believe the request from the user does not fall into any of the other intents, or you believe the intent list does not currently support the requested ask.
1) "Other" - You should say this if you think the user is only asking for an opinion or clarification, or, if you believe they need to provide follow-up information.
2) "AnalyzeDataFrame" - You should say this if you think that the data that the user is using in the analysis should be sent to the AI to be analyzed along with their prompt/question.
3) "DrawChart" - You should say this if you think the user wants a chart (pie chart, bar chart or line chart) created for them based on the dataset provided.
4) "CompareToPeers" - You should say this if you think the user wants to compare their company's financial performance to their peers in the industry.

Please provide the intent without any quotes."""

        self.user_analyse_df_intent = r"""Hello! Your job is to read this prompt from a human being sent to a computer to find out."""

        self.user_chart_type_intent_sys=r"""Hello! Your job is to read this prompt from a human being sent to a computer to find what type of chart would best present the information being requested by the user. You must choose a single chart type from the list provided below. Please respond with only one option with no additional context or feedback. For example, if you believe the user's query would be best satisfied by the creation of a bar chart, please respond with 'BarChart' without quotes. If you are not sure, you can say 'UnableToDetermine', or, 'UnsupportedRequest'.

Chart List:
0) "UnableToDetermine" - You should say this if you are unable to make a good guess as to what type of chart would best fit a user's prompt.
2) "LineChart" - You should say this if you believe that a line chart vizualization would be the best way to present the user's prompt.
3) "BarChart" - You should say this if you believe that a bar chart vizualization would be the best way to present the user's prompt.

Please provide the best chart type without any quotes.
"""

        self.chart_format_sys=r"""Hello! Take a deep breath and think through this problem step-by-step. Please read what you are given and and format it
        in this specific way. If you receive an input that looks like this:
---Output from AI:---
Type: line
Columns: "A", "B", "C", "D", ...
Data: 1, 2, 3, 4, ...
------

---Your Response:---
You would output this:
{"line": {"columns": ["A", "B", "C", "D", ...], "data": [1, 2, 3, 4, ...]}}
------
Your response should be a string. Do not respond with anything else. Only line or bar types are supported. If it's already formatted, please return the input. Please ensure that all strings in "columns" list and data list, should be in double quotes.
"""
        self.companyName = f"Banana Corp"
        self.compareToPeerPrompt = r"""Please take a deep breath and break this down step-by-step. You must try to solve the ask given the dataset and the prompt following these instructions. You are an expert in reading financial metrics and must help """+self.companyName+r""" compare their performance to similar peers in their industry.
        1) Always start with importing pandas, and import numpy if you think it will be needed.
        2) Get the data needed that would be used to answer the user's inquiry.
        3) Respond using a list, tabular format, or plaintext, based on the ask.

        Please be as detailed as possible in your answer. If comparing numbers, be sure to return the numbers for the company and the numbers for it's peers (or industry average) in your response.
                            
        Here is the prompt: """

    def change_model(self, requested_model):
        if requested_model=="gpt-4":
            self.prior_env = self.current_env
            self.current_env = os.path.join(self.base_env_path, "gpt-4.env")
        elif requested_model=="gpt-4-32k":
            self.prior_env = self.current_env
            self.current_env = os.path.join(self.base_env_path, "gpt-4-32k.env")
        elif requested_model=="gpt-4-16k":
            self.prior_env = self.current_env
            self.current_env = os.path.join(self.base_env_path, "gpt-4-16k.env")
        elif requested_model=="gpt-35-turbo":
            self.prior_env = self.current_env
            self.current_env = os.path.join(self.base_env_path, "gpt-35-turbo.env")
        elif requested_model=="prior": #used the active ENV switch to the main env file being used for the dataset
            self.current_env = self.prior_env
        else:
            print("Error loading selected model. Reloading existing.")

        print("Loading env: ",str(self.current_env))
        self.create_log_item('Loading Model:', str(self.current_env))
        load_dotenv(self.current_env)
        self.tiktoken_model = os.getenv("MODEL_NAME")
        self.max_tokens = int(os.getenv("MAX_TOKENS"))-50 #adds a padding of 50 tokens from the max
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_type = os.getenv("OPENAI_API_TYPE")
        openai.api_base = os.getenv("OPENAI_API_BASE")
        openai.api_version = os.getenv("OPENAI_API_VERSION")
        AzureChatOpenAI.deployment_name = os.getenv("DEPLOYMENT_NAME")
        print('Deployment name: ',  os.getenv("DEPLOYMENT_NAME"))

    def load_file(self, file_path):
        #filePath = os.getcwd() + "\\files\\ETR Analysis.xlsx"
        #df = pd.read_csv(filePath, na_values=['','#N/A', 'n/a', None, 'NaN'])
        print('File path: ', file_path)
        df = pd.read_csv(file_path, na_values=['','#N/A', 'n/a', None, 'NaN', '-'])
        #df = df.fillna('')
        return df

    def get_timestamp(self):
        now = datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def get_file_timestamp(self):
        now = datetime.now()
        return now.strftime("%Y%m%d-%H%M%S")
    
    def create_log_item(self, type, message):
        if type=='user' or type=='assistant':
            timestamp = self.get_timestamp()
            self.log_text=self.log_text+"\n Timestamp: "+timestamp+"\n From: "+type+"\n Message: "+message+"\n_____________________________________________"
        else:
            
            self.log_text=self.log_text+"\n"+type+"\n"+message+"\n_____________________________________________" 

    def log_time_taken(self, action, start_time, end_time):
        duration = end_time - start_time
        self.log_text=self.log_text+"\n"+action+" "+str(duration)
        print(f"{action}{end_time - start_time}")

    def log_convo(self):
        timestamp_text = self.get_file_timestamp()
        log_file_path = self.base_log_path + f"\\log-{timestamp_text}.txt"
        self.create_log_item('user', 'end')
        with open(log_file_path, "w") as file:
            file.write(self.log_text)

    def analyze_pandas_df(self, message, df):
        def start_chat_model():
            #change model based on token count
            #df tokens = 
            llm = AzureChatOpenAI(
                openai_api_base=openai.api_base,
                openai_api_version=openai.api_version,
                deployment_name=AzureChatOpenAI.deployment_name,
                openai_api_key=openai.api_key,
                openai_api_type=openai.api_type,
                temperature=0.01)
            print("Azure Chat Deployment Loaded.")
            return llm

        def generate_response(userinput, llm, df):
            agent_response = ""
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=True)
            print("agent created")
            try:
                agent_response=agent.run(userinput)
            except Exception as e:
                print("Error while executing pandas df agent: "+str(e))
            return agent_response

        agent_llm = start_chat_model()
        response = generate_response(message, agent_llm, df)
        return response

    def get_intent(self, prompt, system_prompt): #Returns the intent of the user's most recent message
        def get_intent_tokens(system_prompt, prompt): #returns the number of tokens that are used in the intent system prompt and the user's prompt where intent is being determined
            encoding = tiktoken.encoding_for_model(self.tiktoken_model)  # Gets the tokenizer for the set model (GPT-3.5-Turbo)
            system_tokens = len(encoding.encode(system_prompt))  # Encodes the system_prompt into tokens and gets the length of the tokens
            print("\nSystem Tokens: ", system_tokens)
            prompt_tokens = len(encoding.encode(prompt))
            print("\nPrompt Tokens: ", prompt_tokens)
            total_tokens = system_tokens + prompt_tokens
            print("\nTotal Intent Prompt Tokens: ", total_tokens)
            return total_tokens

        self.change_model('gpt-4') #changes the ENV variables to gpt-4 as gpt-32k is not needed
        intent_tokens = get_intent_tokens(system_prompt, prompt)
        print("Tokens in Full Intent Prompt: " + str(intent_tokens))

        intentChat=[{"role":"system","content":system_prompt}, 
             {"role":"user","content":prompt}]

        try:
            response = openai.ChatCompletion.create(
                    engine=AzureChatOpenAI.deployment_name,
                    messages = intentChat,
                    temperature=0.01,
                    max_tokens=8192-50-intent_tokens, #adds a padding of 50 tokens away from the max. You'll never get close to this.
                    top_p=0.95,
                    frequency_penalty=0,                
                    presence_penalty=0,
                    stop=None)
            self.current_intent = response['choices'][0]['message']['content'].strip()
            self.current_intent = self.current_intent.replace('"', '') #
            print("Identified Message Intent: ", self.current_intent) #Print guessed intent
            self.create_log_item("Identified Message Intent:", self.current_intent)
        except Exception as e:
            print('Something went wrong while fetching the intent: '+str(e))
            self.create_log_item("Error Message:", str(e), f"-end of error-\n")
            self.current_intent = "Other"
        
        self.change_model('prior') #changes the ENV variables back to the previously used model. To add support for dynamically changing model during the initial DF loading process.

    #returns the number of tokens that the dataframe contains. Work in progress.
    def get_dataframe_tokens(self, df: pd.DataFrame) -> int: 
        encoding = tiktoken.encoding_for_model(self.tiktoken_model)  # Gets the tokenizer for the set model (GPT-3.5-Turbo)
        
        #Tokenize Headers
        token_count = sum(len(encoding.encode(column)) for column in df.columns) 

        #Tokenize Rows
        for _, row in df.iterrows():
            for cell in row:
                if isinstance(cell, (str, int, float)):
                    token_count += len(encoding.encode(str(cell)))
        
        return token_count
    
    #Returns the remaining tokens that can be used in the calls to the OpenAI API
    def get_history_tokens(self, history): 
        """
        Args:
        Self: n/a
        History: History is an array that contains items formatted like so: {'role':'assistant','content':'CONTENT HERE'}. It supports system (only one, at the beginning), assistant (AI response), and user (user messages)
        """

        encoding = tiktoken.encoding_for_model(self.tiktoken_model) #Gets the tokenizer for the set model (GPT-3.5-Turbo)
        self.total_tokens = 0 #resets total tokens each time you get the new history tokens
        for message in history:
            text = message['content']
            self.total_tokens = self.total_tokens + len(encoding.encode(text)) #Encodes the system_prompt into tokens, and gets the length of the tokens
        print("\nContext Tokens: ",self.total_tokens) #Prints out the 
        self.remaining_tokens = self.max_tokens-self.total_tokens 

    def send_to_openai(self, message, remaining_tokens):
        print('AzureChatOpenAI.deployment_name:', AzureChatOpenAI.deployment_name)
        try:
            response = openai.ChatCompletion.create(
                engine=AzureChatOpenAI.deployment_name,
                messages = message,
                temperature=0.01,
                max_tokens=remaining_tokens,
                top_p=0.95,
                frequency_penalty=0,                
                presence_penalty=0,
                stop=None)    
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return 'Something went wrong. Please try again: '+str(e)
        
    def get_system_prompt(self, use_predefined_sys_prompt):
        if self.system_message == "":
            if use_predefined_sys_prompt is False:
                aiSystemPromptGenerate = r"Hello. You are an expert at understanding and examining datasets. You will be given data, and you should return a system prompt that will be used in a conversation with a large language model. Please note that the dataset will be appended after the system prompt you create, so you should not include the data in your response, instead, focus on explaining the dataset as if the person you were speaking to doesn't know anything about the dataset, the columns, etc. The data will typically be related to tax, accounting, or finance. Note that your response will be used directly with Python and the OpenAI API, so please do not provide any additional commentary other than the system message. Thank you!"

                self.dataframe_tokens = self.get_dataframe_tokens(self.df)
                self.create_log_item("Dataframe Token Amount", str(self.dataframe_tokens))
                print('Dataframe tokens: ' + str(self.dataframe_tokens))

                startConversationPromptGenerate = f"""Data provided in a dataframe:\n{self.df.to_string(index=False)}\nPlease create the system prompt that explains the dataset. No commentary, please: """
                message_sys = [
                    {"role": "system", "content": aiSystemPromptGenerate},
                    {"role": "user", "content": startConversationPromptGenerate}
                ]
                self.create_log_item("Default System Prompt used to Generate the Conversation System Prompt from Dataset using OpenAI:", aiSystemPromptGenerate)
                self.create_log_item("Initial Prompt used with Default System prompt for Generating System Prompt:", startConversationPromptGenerate)
                
                self.get_history_tokens(message_sys)
                print('Remaining tokens: ' + str(self.remaining_tokens))

                system_message = self.send_to_openai(message_sys, self.remaining_tokens)  # generated system message
                self.create_log_item("AI Generated System Prompt:", system_message)
                self.system_message = system_message + f"\nHere is the data provided in a dataframe:\n{self.df.to_string(index=False)}\n"
            else:
                self.system_message="""You are a helpful assistant."""
                print("\Predefined System Message:\n", self.system_message)
            self.create_log_item( "Chat Model Selected:", self.tiktoken_model)
            self.message_history.append({"role": "system", "content": self.system_message}) #append the system messge
        else:
            print('Getting system prompt from cache.')
         
    def query_bot(self, prompt):
        if prompt == 'end':
            self.log_convo()
            return ('', 'Conversation Ended.', self.message_history, self.system_message, self.df, self.log_text)

        elif prompt == 'restart':
            self.create_log_item('dev', "Message History Cleared. New Conversation Starting")
            self.log_convo()
            return (None, 'Restarting conversation.', None, None, None, None)
        else:
            start_time = time.time()
            self.get_system_prompt(use_predefined_sys_prompt=True) #True = Only works for the given dataset.
            end_time = time.time()
            self.log_time_taken("Time taken to generate system message: ", start_time, end_time)
            
            self.create_log_item('user', prompt)
            self.message_history.append({"role": "user", "content": prompt})
            print("User Prompt: ", prompt)

            start_time = time.time()
            self.get_intent(prompt, self.user_message_intent_sys) #Determine overarching intent of the base user message
            end_time = time.time()
            self.log_time_taken("Time taken to get intent: ", start_time, end_time)
            
            # AnalyzeDataFrame
            if self.current_intent == "AnalyzeDataFrame":
                print("AnalyzeDataFrame Intent")
                analyze_prompt = r"""Please take a deep breath and break this down step-by-step. You must try to solve the ask given the dataset and the prompt following these instructions:
                1) If the values being returned in the response represent a currency, then please format the values by adding a '$' symbol without quotes in front of the numerical values. For example, '23000.00' should be formated as '$23000.00' without quotes. Please ensure that the negative symbol '-' should always be placed before the currency symbol '$'.
                2) If the values being returned in the response represent percentages, then format them into proper percentage values with a '%' symbol added at the end, without any quotes. For example, '0.23' should be '23%', '-0.023' should be '-2.3%'. Please do not add quotes.
                3) Do not make any changes to numerical values if they don't represent a currency or percentage value.

                """ + prompt
                start_time = time.time()
                response = self.analyze_pandas_df(analyze_prompt, self.df)
                end_time = time.time()
                self.log_time_taken("Time taken to get response: ", start_time, end_time)

                self.create_log_item('assistant', response)
                self.message_history.append({"role": "assistant", "content": response})
                return (self.current_intent, response, self.message_history, self.system_message, self.df, self.log_text)
            
            elif self.current_intent == "DrawChart":  #CreateVisualization --> #DetermineChartType
                start_time = time.time()
                self.get_intent(prompt, self.user_chart_type_intent_sys)
                end_time = time.time()
                self.log_time_taken("Time taken to get chart type intent: ", start_time, end_time)
                if self.current_intent == "BarChart":
                    chartPrompt = r"""Please take a deep breath and break this down step-by-step. You must try to solve the ask given the dataset and the prompt following these instructions.
1) Always start with importing pandas, and import numpy if you think it will be needed.
2) Get the data needed that would be used in a bar chart based on the user's prompt (provided below).
3) Your response should be formatted in the same way as the following:
{"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
----
Please ensure that all strings in "columns" list and data list, should be in double quotes.
                    
Here is the prompt: """+prompt

                elif self.current_intent == "LineChart":    
                    chartPrompt = r"""Please take a deep breath and break this down step-by-step. You must try to solve the ask given the dataset and the prompt following these instructions.
1) Always start with importing pandas, and import numpy if you think it will be needed.
2) Get the data needed that would be used in a line chart based on the user's prompt (provided below).
3) Your response should be formatted in the same way as the following:
{"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
----
Please ensure that all strings in "columns" list and data list, should be in double quotes.                   
Here is the prompt: """+prompt

                else:
                    # self.create_log_item('system', 'Error. Please try another prompt.')
                    # return (self.current_intent, 'Error. Please try another prompt.', self.message_history, self.system_message, self.df, self.log_text)
                    self.current_intent = "DrawChart"
                    chartPrompt = """Take a deep breath and break it down step by step. For the following query, if it requires creating a bar chart, reply as follows:
                {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
                
                If the query requires creating a line chart, reply as follows:
                {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}
                
                There can only be two types of chart, "bar" and "line". Please ensure to import pandas and numpy where necessary. Return all output as a string and ensure that variables are replaced with the actual values at all times when providing the output. All strings in "columns" list and data list, should be in double quotes, For example: {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}
                """ + prompt

                start_time = time.time()
                response = self.analyze_pandas_df(chartPrompt, self.df) #Generates the data for use
                end_time = time.time()
                self.create_log_item('assistant-chart-data-agent', response)
                self.log_time_taken("Time taken to get response: ", start_time, end_time)
                ai_example_input=r"""Type: bar
Columns: "New York", "Virginia", "Texas", "Alabama", ...
Data: 120.91, 390.12, 459.53, 234.56, ...
"""
                ai_example_response=r"""{"bar": {"columns": ["New York", "Virginia", "Texas", "Alabama", ...], "data": [120.91, 390.12, 459.53, 234.56, ...]}}"""
                chartChat = [
                {"role": "system", "content": self.chart_format_sys},
                {"role": "user", "content": ai_example_input},
                {"role": "assistant", "content": ai_example_response},
                {"role": "user", "content": response}]

                temp_tokens = self.get_history_tokens(chartChat)

                start_time = time.time()
                response = self.send_to_openai(chartChat, temp_tokens) #Provides chart formatting
                end_time = time.time()
                self.create_log_item('assistant', response)
                self.log_time_taken("Time taken to get response: ", start_time, end_time)
                self.message_history.append({"role": "assistant", "content": response})
                return (self.current_intent, response, self.message_history, self.system_message, self.df, self.log_text)

            elif self.current_intent == "CompareToPeers":
                print("CompareToPeers Intent")
                start_time = time.time()
                peer_file_path = os.path.join(self.base_path, "files", "PeerData.csv")
                peer_df = self.load_file(peer_file_path)
                prompt = self.compareToPeerPrompt+prompt
                response = self.analyze_pandas_df(prompt, peer_df)
                end_time = time.time()
                self.log_time_taken("Time taken to get response: ", start_time, end_time)

                self.create_log_item('assistant', response)
                self.message_history.append({"role": "assistant", "content": response})
                return (self.current_intent, response, self.message_history, self.system_message, self.df, self.log_text)   
            
            elif self.current_intent == "Other":
                print("Other Intent")
                self.remaining_tokens = self.get_history_tokens(self.message_history)
                start_time = time.time()
                response = self.send_to_openai(self.message_history, self.remaining_tokens)
                end_time = time.time()
                self.create_log_item('assistant', response)
                self.log_time_taken("Time taken to get response: ", start_time, end_time)

                self.message_history.append({"role": "assistant", "content": response})
                return (self.current_intent, response, self.message_history, self.system_message, self.df, self.log_text)
            
            # NotSupported
            elif self.current_intent == "UnsupportedRequest":
                print("UnsupportedRequest Intent")
                self.remaining_tokens = self.get_history_tokens(self.message_history)
                start_time = time.time()
                response = self.send_to_openai(self.message_history, self.remaining_tokens)
                end_time = time.time()
                self.create_log_item('assistant', response)
                self.log_time_taken("Time taken to get response: ", start_time, end_time)
                self.message_history.append({"role": "assistant", "content": response})
                return (self.current_intent, response, self.message_history, self.system_message, self.df, self.log_text)
            
            else:
                print("Error - Intent not recognized")