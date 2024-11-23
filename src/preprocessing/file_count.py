import os
from collections import defaultdict


def count_files(dir):
    total_files = 0
    ext_count = defaultdict(int)

    for root, dirs, files in os.walk(dir):
        total_files += len(files)
        for file in files:
            _, ext = os.path.splitext(file)
            ext_count[ext] += 1
    return total_files, ext_count


def print_stats(dir):
    print("=" * 30)
    print(dir)
    total, ext_count = count_files(dir)
    print(f"Total number of files: {total}")
    print("Number of files by extension:")
    for ext, count in ext_count.items():
        print(f"  {ext if ext else 'No extension'}: {count}")
    print("=" * 30)
