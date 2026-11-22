"""Prepara los libros (Project Gutenberg, dominio público): limpia, normaliza y separa entrenamiento/examen."""
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = [
    # id, archivo, título, autor, idioma
    ("shakespeare", "pg100.txt", "Shakespeare", "William Shakespeare", "en"),
    ("quijote", "pg2000.txt", "Don Quijote", "Miguel de Cervantes", "es"),
    ("fierro", "pg14765.txt", "Martín Fierro", "José Hernández", "es"),
    ("dante", "pg1012.txt", "Divina Commedia", "Dante Alighieri", "it"),
]
PUNCT = set(" .,;:!?'-")
# letras poco comunes → su base, así los 4 libros comparten un alfabeto parejo
FOLD = str.maketrans({"â": "a", "ä": "a", "æ": "ae", "ê": "e", "ë": "e", "î": "i", "ï": "i", "ö": "o", "ô": "o", "œ": "oe", "ç": "c", "û": "u"})
# dónde empieza de verdad cada obra (se saltea índice, prólogos legales, cartas)
START = {"shakespeare": "from fairest creatures", "fierro": "aquí me pongo a cantar", "dante": "nel mezzo del cammin", "quijote": "en un lugar de la mancha"}


def strip_gutenberg(raw: str) -> str:
    s = raw.replace("\r", "")
    a = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG[^\n]*\n", s)
    if a:
        s = s[a.end():]
    end = s.find("*** END OF")
    if end != -1:
        s = s[:end]
    return s.strip()


def normalize(text: str) -> str:
    """minúsculas, sin saltos de línea, solo letras (con acentos), espacio y puntuación básica"""
    t = unicodedata.normalize("NFC", text).lower().translate(FOLD)
    t = t.replace("’", "'").replace("‘", "'").replace("—", "-").replace("–", "-")
    out = []
    for ch in t:
        if ch.isalpha() or ch in PUNCT:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
    return re.sub(r" +", " ", "".join(out)).strip()


def load_all():
    books = []
    for bid, fname, title, author, lang in BOOKS:
        raw = (ROOT / "books" / fname).read_text(encoding="utf-8", errors="ignore")
        clean = strip_gutenberg(raw)
        norm = normalize(clean)
        k = norm.find(START[bid])
        k2 = re.sub(r"\s+", " ", clean).lower().find(START[bid])
        books.append({
            "id": bid, "title": title, "author": author, "lang": lang,
            "text": norm[k:] if k >= 0 else norm,
            # fragmento con mayúsculas y puntuación para dibujar las páginas
            "excerpt": re.sub(r"\s+", " ", clean)[max(k2, 0):max(k2, 0) + 9000],
        })
    return books


if __name__ == "__main__":
    for b in load_all():
        print(f"{b['id']:12} {len(b['text']):>9,} letras · {b['text'][200:280]!r}")
