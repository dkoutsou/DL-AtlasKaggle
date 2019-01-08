import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-check_nb', '--checkpoint_nb',
        default='None',
        help='The number of the checkpoint to use in predict'
    )
    args = argparser.parse_args()
    return args
