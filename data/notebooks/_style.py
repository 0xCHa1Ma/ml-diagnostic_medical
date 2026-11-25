"""Style graphique commun aux notebooks (couleurs cohérentes rapport/dashboard)."""
import matplotlib.pyplot as plt
import seaborn as sns

COL_B = "#2E86AB"   # bénin  : bleu
COL_M = "#E4572E"   # malin  : rouge-orangé
PALETTE = {"B": COL_B, "M": COL_M}
FAMILY_COLORS = {"mean": "#2E86AB", "se": "#9C9C9C", "worst": "#E4572E"}


def setup():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "font.size": 10,
    })
