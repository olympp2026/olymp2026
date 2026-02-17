import argparse
import heapq
import os
import shutil
import socket
import tempfile
from multiprocessing import Pool, cpu_count
from typing import BinaryIO, List, Optional, Tuple

RECORD_SIZE = 16


def ipv6_to_packed(addr: str) -> bytes:
    """
    Преобразует IPv6-адрес из строки в каноническое бинарное представление длиной 16 байт (packed).
    """
    return socket.inet_pton(socket.AF_INET6, addr)


# =========================
#  Вспомогательные функции
# =========================


def read_record(f: BinaryIO) -> Optional[bytes]:
    """
        Считывает одну запись IPv6 длины 16 байт.
        return: None при конце файла (EOF) или если в конце файла осталось меньше 16 байт.
    """
    b = f.read(RECORD_SIZE)
    if len(b) != RECORD_SIZE:
        return None
    return b


# =========================
#  Базовое решение в памяти (критерий III)
# =========================

def count_unique_in_memory(input_path: str) -> int:
    """
    Читает файл и кладет все адреса (в packed-виде) в множество set().
    Создано для небольших файлов.
    """
    seen = set()
    with open(input_path, "r", encoding="ascii", errors="strict", newline="") as fin:
        for line in fin:
            s = line.strip()
            if s:
                seen.add(ipv6_to_packed(s))
    return len(seen)


# =========================
#  Внешняя сортировка
# =========================

def flush_run(buf: List[bytes], tmp_dir: str, run_idx: int) -> str:
    """
    Сортирует буфер packed-IPv6 и записывает его в бинарный run-файл.

    Удаляем дубликаты внутри одного run (после сортировки),
    чтобы уменьшить объём временных файлов и ускорить слияния.
    """
    buf.sort()

    deduped: List[bytes] = []
    prev: Optional[bytes] = None
    for rec in buf:
        if rec != prev:
            deduped.append(rec)
            prev = rec

    path = os.path.join(tmp_dir, f"run_{run_idx:06d}.bin")
    with open(path, "wb", buffering=1024 * 1024) as f:
        for rec in deduped:
            f.write(rec)
    return path


def generate_initial_runs(input_path: str, tmp_dir: str, chunk_records: int) -> List[str]:
    """
    Читает входной текстовый файл и создаёт начальные отсортированные run-файлы.

    Алгоритм:
    - читаем строки потоком;
    - переводим IPv6 в packed (16 байт);
    - накапливаем chunk_records записей;
    - сортируем, удаляем дубликаты и сохраняем run на диск.
    """
    runs: List[str] = []
    buf: List[bytes] = []
    run_idx = 0

    with open(input_path, "r", encoding="ascii", errors="strict", newline="") as fin:
        for line in fin:
            s = line.strip()
            if not s:
                continue

            buf.append(ipv6_to_packed(s))

            if len(buf) >= chunk_records:
                runs.append(flush_run(buf, tmp_dir, run_idx))
                run_idx += 1
                buf.clear()

    if buf:
        runs.append(flush_run(buf, tmp_dir, run_idx))
        buf.clear()

    return runs


def open_runs(run_paths: List[str]) -> List[BinaryIO]:
    """Открывает run-файлы для бинарного чтения с увеличенным буфером."""
    return [open(p, "rb", buffering=1024 * 1024) for p in run_paths]


# =========================
#  K-way merge: слияние отсортированных файлов
# =========================

def merge_runs_to_file(run_paths: List[str], out_path: str) -> None:
    """
    Сливает несколько отсортированных run-файлов в один отсортированный.

    Алгоритм:
    - открываем все входные файлы,
    - кладем первый элемент каждого файла в кучу (heap),
    - каждый раз достаём минимальный, пишем в выход,
    - дочитываем следующий из того же файла и снова кладём в кучу.

    Используется read_record(), чтобы не ошибиться на неполной записи в конце.
    """
    files = open_runs(run_paths)
    try:
        heap: List[Tuple[bytes, int]] = []
        for i, f in enumerate(files):
            rec = read_record(f)
            if rec is not None:
                heap.append((rec, i))
        heapq.heapify(heap)

        with open(out_path, "wb", buffering=1024 * 1024) as out:
            while heap:
                rec, i = heapq.heappop(heap)
                out.write(rec)

                nxt = read_record(files[i])
                if nxt is not None:
                    heapq.heappush(heap, (nxt, i))
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass


# =========================
#  Многоэтапное слияние временных файлов (runs) с распараллеливанием
# =========================

def _merge_batch_job(args: Tuple[List[str], str]) -> str:
    """
    Задача для отдельного процесса:
    слить партию run-файлов в один файл и удалить исходные файлы партии.
    """
    batch, merged_path = args
    merge_runs_to_file(batch, merged_path)

    for p in batch:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    return merged_path


def reduce_runs(run_paths: List[str], tmp_dir: str, fan_in: int) -> List[str]:
    """
    Поэтапно сокращает количество временных отсортированных файлов (runs), выполняя слияние группами.

    Это необходимо, чтобы:
    - не превышать ограничение ОС на число одновременно открытых файлов;
    - уменьшить число файлов перед финальным этапом слияния и подсчёта.

    Оптимизация (критерий V): слияние групп на одном этапе выполняется параллельно,
    так как группы независимы и не влияют друг на друга.
    """
    level = 0
    runs = run_paths[:]

    cores = cpu_count() or 2
    workers = max(1, cores - 1)

    while len(runs) > fan_in:
        jobs: List[Tuple[List[str], str]] = []
        for batch_start in range(0, len(runs), fan_in):
            batch = runs[batch_start: batch_start + fan_in]
            merged_path = os.path.join(
                tmp_dir, f"merge_{level:03d}_{batch_start // fan_in:06d}.bin"
            )
            jobs.append((batch, merged_path))

        new_runs: List[str] = []

        if workers <= 1 or len(jobs) == 1:
            for j in jobs:
                new_runs.append(_merge_batch_job(j))
        else:
            with Pool(processes=workers) as pool:
                for merged in pool.imap_unordered(_merge_batch_job, jobs, chunksize=1):
                    new_runs.append(merged)

        runs = sorted(new_runs)
        level += 1

    return runs


def count_unique_across_runs(run_paths: List[str]) -> int:
    """
    Финальный подсчет уникальных адресов:
    выполняем k-way merge по нескольким отсортированным run-файлам
    и считаем количество смен значений (rec != prev).

    Выходной файл полностью не создаем — сразу считаем ответ.
    """
    if not run_paths:
        return 0

    files = open_runs(run_paths)
    try:
        heap: List[Tuple[bytes, int]] = []
        for i, f in enumerate(files):
            rec = read_record(f)
            if rec is not None:
                heap.append((rec, i))
        heapq.heapify(heap)

        prev: Optional[bytes] = None
        uniq = 0

        while heap:
            rec, i = heapq.heappop(heap)
            if rec != prev:
                uniq += 1
                prev = rec

            nxt = read_record(files[i])
            if nxt is not None:
                heapq.heappush(heap, (nxt, i))

        return uniq
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass


# =========================
#  Подсчет уникальных адресов с автоматическим выбором стратегии
# =========================

def solve(input_path: str, output_path: str) -> int:
    """
    Выбор стратегии:
    - маленький файл -> считаем в памяти через set()
    - большой файл -> внешняя сортировка, временные файлы, распараллеливание
    """
    try:
        fsize = os.path.getsize(input_path)
    except OSError as e:
        raise SystemExit(f"Ошибка: не удалось открыть входной файл: {e}")

    SMALL_BYTES = 200 * 1024 * 1024  # 200 МБ

    if fsize <= SMALL_BYTES:
        return count_unique_in_memory(input_path)

    chunk_records = 1_000_000
    fan_in = 128

    tmp_dir = tempfile.mkdtemp(prefix="ipv6_uniq_")
    try:
        runs = generate_initial_runs(input_path, tmp_dir, chunk_records)
        runs = reduce_runs(runs, tmp_dir, fan_in)
        return count_unique_across_runs(runs)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    """
    params:
      1) input_path  — входной файл
      2) output_path — выходной файл
    """
    p = argparse.ArgumentParser(description="Подсчёт уникальных IPv6-адресов в большом файле")
    p.add_argument("input_path", help="Путь к входному текстовому файлу (IPv6 по одному в строке)")
    p.add_argument("output_path", help="Путь к выходному файлу (одно целое число)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ans = solve(args.input_path, args.output_path)

    with open(args.output_path, "w", encoding="utf-8", newline="\n") as fout:
        fout.write(str(ans) + "\n")


if __name__ == "__main__":
    main()
