

def _normalize_number(text: str) -> str:
    if _is_number(text):
        return str(float(text))
    else:
        return text
