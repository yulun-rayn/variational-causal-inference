import argparse

from vci.train import train

def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser()

    # setting arguments
    parser.add_argument("--name", default="default_run")
    parser.add_argument("--artifact_path", type=str, required=True)
    parser.add_argument("--device", default="cuda")

    # dataset arguments
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, required=True, help="gene;celebA;morphoMNIST")
    parser.add_argument("--label_names", type=str, default=None)

    # model arguments
    parser.add_argument("--omega0", type=float, default=1.0, help="weight for individual-specific log-likelihood")
    parser.add_argument("--omega1", type=float, default=1.0, help="weight for covariate-specific log-likelihood")
    parser.add_argument("--omega2", type=float, default=1.0, help="weight for negative Kullback–Leibler divergence")
    parser.add_argument("--dist_outcomes", type=str, default="normal", help="nb;zinb;normal")
    parser.add_argument("--dist_mode", type=str, default="match", help="classify;discriminate;match")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint_classifier", type=str, default=None)
    parser.add_argument("--hparams", type=str, default=None)

    # training arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--eval_mode", type=str, default="native", help="classic;native")

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
