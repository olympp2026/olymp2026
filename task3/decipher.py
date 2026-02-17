import pandas as pd
import re
from pathlib import Path

# Русский алфавит без "ё"
RUS = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
RUS_UP = RUS.upper()

# файл с расшифрованными телефонами
PHONES_TXT = Path(r"C:\Users\annam\olymp\task3\out\decrypted_telephone_numbers.txt")


def shift_char(ch, shift):
    if ch in RUS:
        return RUS[(RUS.index(ch) + shift) % len(RUS)]
    if ch in RUS_UP:
        return RUS_UP[(RUS_UP.index(ch) + shift) % len(RUS_UP)]
    if 'a' <= ch <= 'z':
        return chr((ord(ch) - ord('a') + shift) % 26 + ord('a'))
    if 'A' <= ch <= 'Z':
        return chr((ord(ch) - ord('A') + shift) % 26 + ord('A'))
    return ch


def decrypt_text(text, shift):
    return "".join(shift_char(c, -shift) for c in str(text))


def find_shift_from_kv(address):
    matches = list(re.finditer(r'([А-Яа-я]{2})\.(\d+)', str(address)))
    if not matches:
        return None

    match = matches[-1]  # последнее совпадение
    encrypted = match.group(1).lower()

    shift1 = (RUS.index(encrypted[0]) - RUS.index('к')) % len(RUS)
    shift2 = (RUS.index(encrypted[1]) - RUS.index('в')) % len(RUS)

    return shift1 if shift1 == shift2 else None


def load_source_table(xlsx_path: str) -> pd.DataFrame:
    """
    Под формат файла:
    - строка 0: пустая/мусор
    - строка 1: заголовки
    - строки 2..: данные
    """
    df_raw = pd.read_excel(xlsx_path, header=None)
    headers = df_raw.iloc[1].tolist()
    df = df_raw.iloc[2:].copy()
    df.columns = headers

    # берем только нужные колонки
    df = df[["email", "Адрес"]].reset_index(drop=True)
    return df


def load_phones_txt(path: Path) -> list[str]:
    """
    Берём только непустые строки
    """
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл с телефонами: {path}")

    phones = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line:
            phones.append(line)
    return phones


def main():
    src = load_source_table("Задание-3-данные.xlsx")
    phones_plain = load_phones_txt(PHONES_TXT)

    if len(phones_plain) != len(src):
        raise ValueError(
            f"Количество телефонов в txt ({len(phones_plain)}) "
            f"не совпадает с числом строк в таблице ({len(src)})."
        )

    rows = []
    for i, row in src.iterrows():
        addr_enc = str(row["Адрес"])
        shift = find_shift_from_kv(addr_enc)

        if shift is None:
            # если ключ не нашли — оставим пустым (но телефон все равно подставим)
            rows.append([phones_plain[i], "", "", ""])
            continue

        addr_dec = decrypt_text(addr_enc, shift)
        email_dec = decrypt_text(row["email"], shift)

        rows.append([phones_plain[i], email_dec, addr_dec, shift])

    out = pd.DataFrame(rows, columns=["Телефон", "email", "Адрес", "key"])
    out.to_excel("deanonymized_correct.xlsx", index=False)
    print("Готово: deanonymized_correct.xlsx")


if __name__ == "__main__":
    main()
