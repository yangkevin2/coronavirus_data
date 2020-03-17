from argparse import ArgumentParser

def deduplicate(data_path, save_path):
    cids = set()
    with open(data_path, 'r') as rf:
        for line in rf:
            cids.add(line.strip())
    with open(save_path, 'w') as wf:
        for cid in cids:
            wf.write(cid + '\n')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to file with data to deduplicate')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Path where deduplicated data will be saved')
    args = parser.parse_args()

    deduplicate(**vars(args))
