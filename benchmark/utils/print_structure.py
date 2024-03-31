import os

def print_directory_structure(directory, level=0):
    for root, dirs, files in os.walk(directory):
        current_level = root.replace(directory, '').count(os.sep)
        if current_level == level:
            indent = '│   ' * (level - 1) + '├── '
        else:
            indent = '│   ' * (level - 1) + '└── '
        print(indent + os.path.basename(root) + '/')
        for file in files:
            sub_indent = '│   ' * current_level + '├── '
            print(sub_indent + file)

# 指定要打印目录结构的目录
directory_path = '/workspace/ViTMatting/'

print_directory_structure(directory_path)