def update_cache(cache, key, value):
    if key in cache:
        cache[key].append(value)
    else:
        cache[key] = [value]
    return value

def coalesce(*args):
    cache = {}
    for arg in args:
        if arg is not None:
            return arg

    return None