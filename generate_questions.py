import PyPDF2
from io import StringIO
import glob
import docx2txt
import weaviate
import openai
import os
import ast

import tracemalloc
import json

tracemalloc.start()

file_directory = "data"

def split_text(text, file_name, chunk_size, chunk_overlap):
    start = 0
    end = chunk_size
    while start < len(text):
        yield (text[start:end], file_name)
        start += chunk_size - chunk_overlap
        end = start + chunk_size

def split_pdf_text_by_page(pdf_path):
    pages = []
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()

            pages.append(text)
    return pages

def load_documents(directory, glob_patterns):
    documents = []
    for glob_pattern in glob_patterns:
        file_paths = glob.glob(os.path.join(directory, glob_pattern))
        for fp in file_paths:
            try:
                if fp.endswith(".docx"):
                    text = docx2txt.process(fp)
                    pages = [text]  # Treat the whole document as a single "page"
                elif fp.endswith(".pdf"):
                    pages = split_pdf_text_by_page(fp)
                else:
                    print(f"Warning: Unsupported file format for {fp}")
                    continue
                documents.extend(
                    [
                        (page, os.path.basename(fp), i + 1)
                        for i, page in enumerate(pages)
                    ]
                )
            except Exception as e:
                print(f"Warning: The file {fp} could not be processed. Error: {e}")
    return documents

def split_documents(documents, chunk_size, chunk_overlap):
    texts = []
    metadata = []
    for doc_text, file_name, page_number in documents:
        for chunk in split_text(doc_text, file_name, chunk_size, chunk_overlap):
            sentence = chunk[0]
            texts.append(sentence)
            metadata.append(str(file_name) + " Pg: " + str(page_number))

    return texts, metadata

def append_to_file(file_path, data):
    with open(file_path, 'a') as file:
        file.write('\n' + data)


def generate_questions(directory, chunk_size, chunk_overlap):
    glob_patterns = ["*.docx", "*.pdf"]
    documents = load_documents(directory, glob_patterns)

    chunk_overlap = 0
    texts, metadata = split_documents(documents, chunk_size, chunk_overlap)

    data_objs = [{"text": tx, "metadata": met} for tx, met in zip(texts, metadata)]
    total = len(data_objs)

    system_message = "You will be given content from documents. You will need to return 5 questions that are important and technical and whose answers lies within the context provided. Return a list of 5 questions"
    for text in data_objs:
        user_message = "Content: " + text["text"]
        questions = ask_model(system_message, user_message, question_generate_function)
        questions = "\n".join(questions)
        append_to_file("questions.txt", questions)



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

def ask_model(system_message, user_message, custom_function):
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


config_data = readAndWriteJsonData("configu.json", "r")

global lm_client

lm_client = openai.OpenAI(api_key=config_data["openaiKey"])

generate_questions(file_directory, 1000, 0)