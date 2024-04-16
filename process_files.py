import PyPDF2
from io import StringIO
import glob
import docx2txt
import weaviate
import openai
import os
import tracemalloc
import json

tracemalloc.start()

openaikey = "<Key>"
vectordburl = "https://hatest-mhjogoio.weaviate.network"
vectordbapi = "AvRHcHnVHYhoowtRwVVcH4W1SjfFbSc4nSuQ"
class_name_answers = "ha_answers"
class_name_questions = "ha_questions"
path = "configu.json"
file_directory = "data"

config_data = {
                "openaiKey":openaikey,
                "vectordb_url": vectordburl,
                "vectordb_api": vectordbapi,
                "class_name_answers": class_name_answers,
                "class_name_questions": class_name_questions
              }

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

def read_lines_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]
    lines = [{"text": text, "metadata": str(index)} for index, text in enumerate(lines, start=1)]
    return lines

def pdf_vectorization(layer_config, chunk_size=400, vectorize_text_questions=False, directory=None):
    url = layer_config["vectordb_url"]
    auth_key = layer_config["vectordb_api"]
    openai_key = layer_config["openaiKey"]

    client = weaviate.Client(
        url=url,
        auth_client_secret=weaviate.AuthApiKey(api_key=auth_key),
        additional_headers={"X-OpenAI-Api-Key": openai_key},
    )
    

    if vectorize_text_questions:
        class_name = layer_config["class_name_questions"]
        data_objs = read_lines_from_file('questions.txt')
    else:
        class_name = layer_config["class_name_answers"]
        glob_patterns = ["*.docx", "*.pdf"]
        documents = load_documents(directory, glob_patterns)
        chunk_overlap = 0
        texts, metadata = split_documents(documents, chunk_size, chunk_overlap)
        data_objs = [{"text": tx, "metadata": met} for tx, met in zip(texts, metadata)]

    total = len(data_objs)

    i = 0

    class_obj = {
        "class": class_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "metadata",
                "dataType": ["text"],
            },
        ],
        "vectorizer": "text2vec-openai",
        "moduleConfig": {
            "text2vec-openai": {
                "vectorizeClassName": False,
                "model": "ada",
                "modelVersion": "002",
                "type": "text",
            },
        },
    }
    try:
        client.schema.create_class(class_obj)
    except Exception as e:
        print("Error:", e)
        
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)

    print(total)

    client.batch.configure(batch_size=50)
    with client.batch as batch:
        for data_obj in data_objs:
            i += 1
            if i > -1:
                loading_status = "Uploaded: "+ str(i)+ "/"+ str(total)
                print(loading_status)
                batch.add_data_object(data_obj, class_name)
            else:
                print("Already present.", i)
    print("Total: ",i)

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


readAndWriteJsonData(path, "w", config_data)

pdf_vectorization(layer_config=config_data, directory=file_directory)

#If you want to use efficient retrival in the app.py
pdf_vectorization(layer_config=config_data, vectorize_text_questions=True)