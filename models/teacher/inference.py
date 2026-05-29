from models.teacher import gemini, qwen, nemotron, kimi


def get_response(model, video_path):
    if model == 'gemini':
        return gemini.get_response(video_path)
    elif model == 'qwen':
        return qwen.get_response(video_path)
    elif model == 'nemotron':
        return nemotron.get_response(video_path)
    elif model == 'kimi':
        return kimi.get_response(video_path)
    else:
        raise ValueError('Unknown model: {}'.format(model))