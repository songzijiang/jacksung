import hashlib


def calculate_file_hash(file_path, hash_algorithm="md5", chunk_size=4096):
    hash_obj = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as file:
        while True:
            data = file.read(chunk_size)
            if not data:
                break
            hash_obj.update(data)
    return hash_obj.hexdigest()


def hash_files(file_paths, hash_algorithm="md5"):
    hash_list = []
    for filepath in file_paths:
        hash_list.append(calculate_file_hash(filepath, hash_algorithm))
    return hash_string(''.join(hash_list), hash_algorithm)


def hash_string(s, hash_algorithm="md5"):
    hash_obj = hashlib.new(hash_algorithm)
    # 对连接后的字符串进行哈希
    hash_obj.update(s.encode())
    return hash_obj.hexdigest()
