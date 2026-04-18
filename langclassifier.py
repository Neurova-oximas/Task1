def arabic_or_english(text: str) ->str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u08FF')
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())

    if arabic_chars==0 and english_chars==0:
        return "unknown"

    return "arabic" if arabic_chars>=english_chars else "english"