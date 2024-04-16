from flask import Flask, request, jsonify, render_template
import json
import openai
import weaviate
import time
import ast

app = Flask(__name__)

global lm_client
global knowledge_base 

question_generate_function = [
    {
        "name": "return_questions",
        "description": "to be used to return list of questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_list": {
                    "type": "array",
                    "description": "List of questions.",
                    "items": {"type": "string"},
                },
            },
            "required": ["item_list"],
        },
    }
]

generate_response_function = [
    {
        "name": "return_response",
        "description": "to be used to return response.",
        "parameters": {
            "type": "object",
            "properties": {
                "response_answer": {
                    "type": "string",
                    "description": "This should be the answer that was generated from the context, given the question",
                },
            },
            "required": ["response_answer"],
        },
    }
]

def readAndWriteJsonData(path, mode, data=None):
    if mode == "r":
        try:
            with open(path, "r") as file:
                json_data = json.load(file)
            return json_data
        except FileNotFoundError:
            print(f"File '{path}' not found.")
            return None
        except json.JSONDecodeError:
            print(f"Failed to decode JSON from '{path}'.")
            return None
    elif mode == "w":
        try:
            with open(path, "w") as file:
                json.dump(data, file, indent=4)
            print(f"Data written to '{path}' successfully.")
        except Exception as e:
            print(f"Error writing data to '{path}': {e}")
    else:
        print("Invalid mode. Use 'r' for reading or 'w' for writing.")



def initiate_clients():
    global lm_client, knowledge_base
    config_data = readAndWriteJsonData("configu.json", "r")

    try:
        lm_client = openai.OpenAI(api_key=config_data["openaiKey"])
        msg = [
            {"role": "system", "content": "system_message"},
            {"role": "user", "content": "user_message"},
        ]

        response = lm_client.chat.completions.create(
            model="gpt-4",
            messages=msg,
            max_tokens=1000,
            temperature=0.0,
        )
    except Exception as error:
        print(error)

    try:
        knowledge_base = weaviate.Client(
            url=config_data["vectordb_url"],
            auth_client_secret=weaviate.AuthApiKey(api_key=config_data["vectordb_api"]),
            additional_headers={"X-OpenAI-Api-Key": config_data["openaiKey"]},
        )
    except Exception as error:
        print(error)

initiate_clients()


def ask_model(system_message, user_message, custom_function, generate_reply=True):
    global lm_client
    msg = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg,
        max_tokens=500,
        temperature=0.0,
        functions=custom_function,
        function_call="auto",
    )
    reply = response.choices[0].message.content

    if generate_reply:
        reply = json.loads(response.choices[0].message.function_call.arguments)["response_answer"]

    else:
        try:
            reply = ast.literal_eval(reply)
        except:
            try:
                reply = json.loads(response.choices[0].message.function_call.arguments)["item_list"]
                print(reply)
            except:
                print(reply)
                reply = []
    return reply

def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def qdb(query, db_client, name, cname, chunk_id, limit, question=True):
    context = None
    metadata = []
    try:
        res = (
            db_client.query.get(name, ["text", "metadata"])
            .with_near_text({"concepts": query})
            .with_limit(limit)
            .do()
        )
        
        context = "" if not question else []
        metadata = []
        for i in range(limit):
            if question:
                context.append(res["data"]["Get"][cname][i]["text"])
            else:
                context += "Chunk ID: " + str(chunk_id) + "\n"
                context += res["data"]["Get"][cname][i]["text"] + "\n\n"
            met = res["data"]["Get"][cname][i]["metadata"]
            metadata.append(met)
            chunk_id += 1
    except Exception as e:
        print("Exception in DB")
        print(e)
        time.sleep(3)
        context, metadata = qdb(query, db_client, name, cname, chunk_id, limit)
    return context, metadata

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/return_questions', methods=['POST'])
def return_questions():
    data = request.json
    if not data or 'conversation' not in data or 'efficient_search' not in data:
        missing_fields = []
        if 'conversation' not in data:
            missing_fields.append("'conversation'")
        if 'efficient_search' not in data:
            missing_fields.append("'efficient_search'")
        return jsonify({"error": f"Missing {', '.join(missing_fields)} in payload"}), 400

    conversation = data['conversation']
    efficient_search = data['efficient_search']

    
    if efficient_search==0:
        questions = read_text_file('questions.txt')
    else:
        config_data = readAndWriteJsonData("configu.json", "r")
        question, _ = qdb(conversation, knowledge_base, config_data['class_name_questions'], config_data['class_name_questions'].capitalize(), 0, 15)
        top_10_questions = question
        questions = "\n".join(question)

    if (efficient_search==1 or efficient_search==0):
        print("LLM based ordering")
        system_message = "You will be given a peice of conversation between a contact center agent and the customer. You will also be provided a list of pre-generated questions. Based on current conversation, make a guess as to where it is headed, and return a list of the top 5 most relevant questions that the customer might ask."
        user_message = "Conversation: " + conversation + " \n\n Questions: " + questions
        top_10_questions = ask_model(system_message, user_message, question_generate_function, generate_reply=False)
    else:
        print("Embedding based ordering")

    return jsonify(top_10_questions)


@app.route('/get_response', methods=['POST'])
def get_response():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Missing 'question' in payload"}), 400

    question = data['question']
    config_data = readAndWriteJsonData("configu.json", "r")
    context, _ = qdb(question, knowledge_base, config_data['class_name_answers'], config_data['class_name_answers'].capitalize(), 0, 5)
    
    if isinstance(context, list):
        context_str = "\n".join(context)  # Join list elements into a single string
    else:
        context_str = context  # Or handle as a string directly if not a list
    
    system_message = "You will be given a list of chunks of information extracted from a list of documents. You will also be given a question. You must formulate an answer, if it exists, only using the context and return the response."
    user_message = f"Question:  {question} \n\n Context: \n{context_str}"

    response = ask_model(system_message, user_message, generate_response_function, generate_reply=True)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)