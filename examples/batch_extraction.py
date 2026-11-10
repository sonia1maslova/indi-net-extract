"""Run a batch from already prepared images; no raw-data preprocessing."""
from __future__ import annotations

import argparse
import json

from indinetextract import extract_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-ids", nargs="+", required=True)
    parser.add_argument("--fmri-paths", nargs="+", required=True)
    parser.add_argument("--templates", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--method", choices=["dual_regression", "gig_ica"], required=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=2)
    args = parser.parse_args()
    result = extract_batch(**vars(args))
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
