"""Compare the policy weights of two Stable-Baselines3 checkpoints.

Example (retrained final model vs. the supplied one):
    python compare_checkpoints.py ^
        experiments/flight_policy/my_flight_run/final/candidate.zip ^
        experiments/flight_policy/flight_v2_seed63011_run03/final/candidate.zip

The .zip files themselves always differ byte-for-byte (they embed save
timestamps), so this compares the weight tensors stored inside them.
"""

import argparse
import io
import zipfile

import torch


def policy_weights(path):
    with zipfile.ZipFile(path) as archive:
        return torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('first')
    parser.add_argument('second')
    args = parser.parse_args()
    a, b = policy_weights(args.first), policy_weights(args.second)
    if a.keys() != b.keys():
        raise SystemExit('Different network structure: parameter names differ')
    identical = all(torch.equal(a[k], b[k]) for k in a)
    worst = max(float((a[k].double() - b[k].double()).abs().max()) for k in a)
    print(f'{len(a)} weight tensors compared')
    print(f'bit-identical: {identical}')
    print(f'max absolute difference: {worst:.3e}')
    raise SystemExit(0 if identical else 1)


if __name__ == '__main__':
    main()
