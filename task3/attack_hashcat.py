import subprocess
from pathlib import Path
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

INPUT_XLSX = BASE_DIR / "Задание-3-данные.xlsx"

HASHCAT_EXE = BASE_DIR / "hashcat-7.1.2" / "hashcat.exe"
HASHCAT_CWD = HASHCAT_EXE.parent

HASH_MODE = "100"   # SHA1
ATTACK_MODE = "3"   # mask

HASHES_TXT = OUT_DIR / "hashes_phone.txt"
HC_OUT = OUT_DIR / "hashcat_phone_out.txt"

# локальный potfile, чтобы не мешали старые глобальные результаты
POTFILE = OUT_DIR / "phones.potfile"

DECRYPTED_TXT = OUT_DIR / "decrypted_telephone_numbers.txt"
OUT_XLSX = OUT_DIR / "phones_recovered.xlsx"

SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def load_phone_hashes_from_excel(path: Path) -> list[str]:
    """
    Берем колонку 'Телефон' и сохраняем порядок строк.
    """
    df_raw = pd.read_excel(path, header=None)
    headers = df_raw.iloc[1].tolist()
    df = df_raw.iloc[2:].copy()
    df.columns = headers

    if "Телефон" not in df.columns:
        raise ValueError(f"нет колонки телефон. Колонки: {list(df.columns)}")

    hashes_in_order = df["Телефон"].astype(str).str.strip().tolist()
    hashes_in_order = ["" if h.lower() == "nan" else h for h in hashes_in_order]
    return hashes_in_order


def write_hashes_for_hashcat(hashes_in_order: list[str]) -> list[str]:
    """
    В hashcat отдаем только валидные SHA1 (40 hex).
    """
    valid = [h.lower() for h in hashes_in_order if SHA1_RE.fullmatch(h or "")]
    HASHES_TXT.write_text("\n".join(valid) + ("\n" if valid else ""), encoding="utf-8")
    return valid


def parse_hash_plain_text(text: str) -> dict[str, str]:
    """
    Парсим строки формата hash:plain в dict.
    """
    mapping = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        h, plain = line.split(":", 1)
        mapping[h.strip().lower()] = plain.strip()
    return mapping


def parse_hash_plain_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return parse_hash_plain_text(path.read_text(encoding="utf-8", errors="ignore"))


def run_hashcat_only_8plus10(outfile: Path):
    """
    Запускаем взлом формата 8 + 10 цифр.
    ВАЖНО:
      - используем локальный potfile (out/phones.potfile)
      - если hashcat решит "all hashes found as potfile", мы позже достанем их через --show
    """
    mask = "8?d?d?d?d?d?d?d?d?d?d"

    cmd = [
        str(HASHCAT_EXE),
        "-a", ATTACK_MODE,
        "-m", HASH_MODE,

        "-D", "1",
        "-d", "2",
        "--force",
        "-O",
        "-w", "1",

        "--kernel-accel", "38",
        "--kernel-loops", "512",

        # локальный potfile
        "--potfile-path", str(POTFILE),

        # outfile hash:plain
        "--outfile-format", "2",
        "--status",
        "-o", str(outfile),

        str(HASHES_TXT),
        mask,
    ]

    print("\nRUN:", " ".join(cmd))
    subprocess.run(cmd, check=False, cwd=str(HASHCAT_CWD))


def hashcat_show_to_file(outfile: Path):
    """
    Если hashcat говорит "All hashes found as potfile" и не пишет -o,
    то принудительно выгружаем решения через --show и сохраняем в outfile.
    """
    cmd = [
        str(HASHCAT_EXE),
        "--show",
        "-m", HASH_MODE,
        "--potfile-path", str(POTFILE),
        "--outfile-format", "2",
        str(HASHES_TXT),
    ]

    print("\nRUN:", " ".join(cmd))
    p = subprocess.run(
        cmd,
        check=False,
        cwd=str(HASHCAT_CWD),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    # hashcat --show печатает в stdout. Сохраняем как hash:plain
    outfile.write_text(p.stdout or "", encoding="utf-8")


def main():
    if not HASHCAT_EXE.exists():
        raise FileNotFoundError(f"hashcat.exe not found: {HASHCAT_EXE}")
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"Excel not found: {INPUT_XLSX}")

    hashes_in_order = load_phone_hashes_from_excel(INPUT_XLSX)
    print("Rows in phone column:", len(hashes_in_order))

    valid = write_hashes_for_hashcat(hashes_in_order)
    print("Valid SHA1 hashes for hashcat:", len(valid), "->", HASHES_TXT)

    # 1) запускаем атаку (может ничего не делать, если всё уже в potfile)
    run_hashcat_only_8plus10(HC_OUT)

    # 2) гарантированно получаем результаты через --show (даже если -o пустой)
    hashcat_show_to_file(HC_OUT)

    # 3) парсим map hash -> phone
    mapping = parse_hash_plain_file(HC_OUT)

    # 4) собираем телефоны в исходном порядке строк Excel
    phones_in_order = []
    found = []
    for h in hashes_in_order:
        hh = (h or "").strip().lower()
        phone = mapping.get(hh, "") if SHA1_RE.fullmatch(hh) else ""
        phones_in_order.append(phone)
        found.append(bool(phone))

    # 5) пишем итоговый txt
    DECRYPTED_TXT.write_text(
        "\n".join(phones_in_order) + ("\n" if phones_in_order else ""),
        encoding="utf-8"
    )
    print("Saved:", DECRYPTED_TXT)

    # 6) контрольный excel
    pd.DataFrame({
        "Телефон_hash": hashes_in_order,
        "Телефон_plain": phones_in_order,
        "found": found,
    }).to_excel(OUT_XLSX, index=False)

    print("Saved:", OUT_XLSX)
    print("Total recovered:", sum(found), "of", len(found))


if __name__ == "__main__":
    main()
