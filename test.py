import requests

url = 'http://localhost:5000/return_questions'

conversation_data = {
    'conversation': 'Lets start a meeting.',
    'efficient_search': 2
}

response = requests.post(url, json=conversation_data)

if response.status_code == 200:
    print(response.json())
else:
    print(f'Error: {response.status_code}, Message: {response.json()}')

url = 'http://localhost:5000/get_response'

question_data = {
    'question': response.json()[0]
}

response = requests.post(url, json=question_data)

if response.status_code == 200:
    print(response.json())
else:
    print(f'Error: {response.status_code}, Message: {response.json()}')