from run_notebook import run

def main():

    ns = run('../notebooks/full_search_distributions.ipynb')

    for fname, fig in ns['figures'].items():
        if fname == 'new_gallery':
            fig.savefig(fname + '.eps')
        if fname == 'score':
            fig.savefig('new_score.eps')


if __name__ == "__main__":
    main()
