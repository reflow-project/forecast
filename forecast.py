import pandas as pd

data_dir = '.'

def main():
    purchased = pd.read_csv(f'{data_dir}/purchased.csv')
    sold = pd.read_csv(f'{data_dir}/sold.csv')
    shelf_life = pd.read_csv(f'{data_dir}/shelf_life.csv')
    breakpoint()


if __name__ == "__main__":
    import argparse
    from six import text_type
    import sys
    

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # parser.add_argument(
    #     'suffix',
    #     type=text_type,
    #     nargs=1,
    #     help='specifies the suffix of the file to consider',
    # )

    args, unknown = parser.parse_known_args()

    try:
        main(
            # args.suffix[0],
        )
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("Done\n")


