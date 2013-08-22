from run_notebook import run

def main():

    ns = run('../notebooks/wide_area_plots.ipynb')

    for fname, fig in ns['figures'].items():
        if fname not in ['l317', 'l305', 'l299']:
            continue
        fig.set_tight_layout(True)
        fig.savefig(fname + '.eps')


if __name__ == "__main__":
    main()
