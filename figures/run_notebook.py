import io
from IPython.nbformat import current

def run(fname):

    result = {}

    with io.open(fname, 'r', encoding='utf-8') as f:
        nb = current.read(f, 'json')

    for cell in nb.worksheets[0].cells:
        if cell.cell_type == 'code' and cell.language == 'python':
            exec cell.input.replace('%matplotlib', '#%matplotlib') in result

    return result
