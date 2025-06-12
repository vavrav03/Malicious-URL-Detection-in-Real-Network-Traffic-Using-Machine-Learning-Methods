import json

def print_dict_level1_inline(d: dict, float_precision=10):
    def format_value(v):
        if isinstance(v, float):
            return round(v, float_precision)
        if isinstance(v, dict) or isinstance(v, list):
            return json.dumps(v, indent=None, separators=(", ", ": "))
        return repr(v)

    for k, v in d.items():
        print(f"{repr(k)}: {format_value(v)}")