from argparse import ArgumentParser

from .utils.tomo_run import run


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('input', default='stdin', type=str,
                        help='Path to input .dat file, or use "stdin" for '
                             'taking input from stdin.')
    parser.add_argument('-t', '--tomoscope', default=False,
                        action='store_true', help='Run in Tomoscope mode.')
    parser.add_argument('-o', '--output', type=str,
                        help='Output directory for saving stuff.')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='Plot phase space after reconstruction.')

    args = parser.parse_args()

    run(args.input, tomoscope=args.tomoscope, output_dir=args.output,
        plot=args.plot)
