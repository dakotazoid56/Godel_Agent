

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False
