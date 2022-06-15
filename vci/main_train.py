from train.train import train

import argparse

def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description="Drug combinations.")

    # output arguments
    parser.add_argument('--name', default='default_run')
    parser.add_argument("--artifact_path", type=str, required=True)

    # mode argument
    parser.add_argument("--dist_mode", type=str, default="match", help='classify;fit;match')
    parser.add_argument("--outcome_dist", type=str, default="normal", help='nb;zinb;normal')

    # dataset arguments
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--perturbation_key", type=str, default="perturbation")
    parser.add_argument("--control_key", type=str, default="control")
    parser.add_argument("--dose_key", type=str, default=None)
    parser.add_argument("--covariate_keys", nargs="*", type=str, default=[])
    parser.add_argument("--split_key", type=str, default=None)

    # model arguments
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hparams", type=str, default="") # see set_hparams_() in model.model

    # training arguments
    parser.add_argument("--max_epochs", type=int, default=2000)
    parser.add_argument("--max_minutes", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint_freq", type=int, default=5)

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_arguments())