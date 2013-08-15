from run_notebook import run

def main():

    ns = run('../notebooks/Expert_Votes.ipynb')

    for fname, fig in ns['figures'].items():
        fig.set_tight_layout(True)
        fig.savefig("expert_" + fname + '.eps')


if __name__ == "__main__":
    main()
