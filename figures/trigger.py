from run_notebook import run

def main():

    ns = run('../notebooks/trigger.ipynb')

    for fname, fig in ns['figures'].items():
        if fname not in ['trigger']:
            continue
        fig.savefig(fname + '.pdf')



if __name__ == "__main__":
    main()
