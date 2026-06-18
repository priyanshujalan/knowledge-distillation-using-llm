from models.teacher import gemini, qwen, nemotron, kimi, gemma
from models.teacher.schema import TeacherResponse


def get_response(model, video_path):
    if model == 'gemini':
        response = gemini.get_response(video_path)
    elif model == 'qwen':
        response = qwen.get_response(video_path)
    elif model == 'nemotron':
        response = nemotron.get_response(video_path)
    elif model == 'kimi':
        response = kimi.get_response(video_path)
    elif model == 'gemma':
        response = gemma.get_response(video_path)
    else:
        raise ValueError('Unknown model: {}'.format(model))

    parsed_response = TeacherResponse(**response)
    return parsed_response.model_dump()