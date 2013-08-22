from run_notebook import run

def main():

    ns = run('../notebooks/distributions.ipynb')

    for fname, fig in ns['figures'].items():
        if fname not in ['lon', 'lat', 'hii_score']:
            continue
        fig.set_tight_layout(True)
        fig.savefig('dist_'+fname + '.eps')


if __name__ == "__main__":
    main()
