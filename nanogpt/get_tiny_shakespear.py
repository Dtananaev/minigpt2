#
# Author: Denis Tananaev
# Date: 09.10.2024
#
import os
import urllib.request

def get_tiny_shakespear(output_directory: str, filename: str = "input.txt"):
    """Loads tiny shakespear data from web."""

    
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    output_path = os.path.join(output_directory,filename )
    urllib.request.urlretrieve(url, output_path)

    with open(output_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
