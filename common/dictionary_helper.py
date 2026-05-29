import json

def clean_teacher_response(response):
    if response[0:7] == "```json":
        return json.loads(response.replace('```json\n', '').replace('```', ''))
    else:
        return json.loads(response)