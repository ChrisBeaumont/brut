ms = open('ms.tex').read()
import os
def numbers():
    fig_no = 0
    subfig_no = 0

    results = []
    for line in ms.splitlines():
        if 'begin{figure' in line:
            fig_no += 1
            subfig_no = 0
        if 'includegraphics' in line:
            filename = line.split('{')[-1].split('}')[0]
            results.append('%s\t\t\tFigure %i%s' %
                           (filename, fig_no, ' BCDEFGHIJKLMNOP'[subfig_no]))
            subfig_no += 1
            if subfig_no == 2:
                results[-2] = results[-2][:-1] + 'A'

    for r in sorted(results):
        print r

def figures():

    for line in ms.splitlines():
        if 'includegraphics' in line:
            filename = line.split('{')[-1].split('}')[0]
            if os.path.exists(filename + '.eps'):
                print filename + '.eps'
            elif os.path.exists(filename + '.png'):
                print filename + '.png'
            else:
                raise RuntimeError("File not found:%s" % filename)


if __name__ == "__main__":
    import sys
    if 'numbers' in sys.argv:
        numbers()
    else:
        figures()
