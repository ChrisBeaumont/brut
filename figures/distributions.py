from run_notebook import run
import os

def main():
    os.chdir('../notebooks/')
    ns = run('../notebooks/distributions.ipynb')
    os.chdir('../figures/')

    for fname, fig in ns['figures'].items():
        if fname not in ['lon', 'lat', 'hii_score', 'map']:
            continue
        fig.set_tight_layout(True)
        fig.savefig('dist_'+fname + '.eps')


if __name__ == "__main__":
    main()
