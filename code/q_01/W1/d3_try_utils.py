# 1. 字符串反转（多种实现）
def reverse_string(s):
    return s[::-1]  # 切片反转

# 2. 列表去重（保持顺序）
def unique_list(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

# 3. 字典按值排序
def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))

# 4. 文件读写
def read_write_file(filepath, content=None):
    if content is not None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def parse_csv(csv_text):
    lines = csv_text.strip().split('\n')
    headers = lines[0].split(',')
    data = []
    for line in lines[1:]:
        values = line.split(',')
        data.append(dict(zip(headers, values)))
    return data