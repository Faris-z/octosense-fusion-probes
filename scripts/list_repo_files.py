"""List files in the OctoSense HF dataset repo, to locate the metadata table.

Run:  .venv/Scripts/python.exe scripts/list_repo_files.py
"""
from huggingface_hub import list_repo_files

REPO = "anthonytec2/OctoSense"

files = list_repo_files(REPO, repo_type="dataset")
print(f"total files in repo: {len(files)}\n")

# Anything that looks like a metadata / index table rather than sequence payload.
interesting = [
    f for f in files
    if f.endswith((".parquet", ".csv", ".json", ".md", ".yaml"))
    and "/" not in f.rstrip("/").replace("semantic_search/", "")
    or f.count("/") <= 1 and f.endswith((".parquet", ".csv", ".json", ".yaml", ".md"))
]
print("--- candidate metadata files (top-level-ish) ---")
for f in sorted(set(interesting)):
    print(" ", f)

print("\n--- first 25 files overall ---")
for f in files[:25]:
    print(" ", f)
