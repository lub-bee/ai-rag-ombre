import os
import json
import hashlib

CACHE_FILE = "embed_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
    
def file_needs_embedding(filepath, cache):
    current_hash = get_file_hash(filepath)
    cached_hash = cache.get(filepath)
    return current_hash != cached_hash

def update_cache(filepath, cache):
    cache[filepath] = get_file_hash(filepath)