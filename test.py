import os
from pathlib import Path

def tree(directory, prefix=""):
    output = ""

    entries = sorted(os.listdir(directory))
    entries_count = len(entries)

    for index, entry in enumerate(entries):
        path = os.path.join(directory, entry)

        connector = "└── " if index == entries_count - 1 else "├── "

        output += prefix + connector + entry + "\n"

        if os.path.isdir(path):
            extension = "    " if index == entries_count - 1 else "│   "
            output += tree(path, prefix + extension)

    return output

# Usage
root_dir = 'Unet'

out = tree(root_dir)

out = '```\n' + out + '\n```'


with open('r.md', 'w',encoding='utf-8') as f:
    f.write(out)