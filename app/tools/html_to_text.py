import re
def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<script.*?>.*?</script>", " ", html)
    html = re.sub(r"(?is)<style.*?>.*?</style>", " ", html)
    text = re.sub(r"(?s)<.*?>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
