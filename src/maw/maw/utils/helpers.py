import random


def sample_params(space: dict) -> dict:
    sampled = {}
    for k, v in space.items():
        if isinstance(v, (list, tuple)):
            sampled[k] = random.choice(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            sampled[k] = v
        else:
            raise ValueError(f"Invalid param type: {type(v)}")
    return sampled



