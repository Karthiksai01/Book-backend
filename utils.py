from docx import Document as DocxDocument
from pypdf import PdfReader


def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def extract_text_from_docx(file_bytes: bytes) -> str:
    import io
    doc = DocxDocument(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_pdf(file_bytes: bytes) -> str:
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
        text += "\n"
    return text


def extract_text(filename: str, file_bytes: bytes) -> str:
    filename = filename.lower()

    if filename.endswith(".txt"):
        return extract_text_from_txt(file_bytes)

    if filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    raise ValueError("Unsupported file type. Only PDF, DOCX, TXT allowed.")
