"""Re-use one saved fit for independent smoothing widths and exact Dice."""
from __future__ import annotations

import argparse

from indinetextract import NetworkResult, postprocess_maps, template_dice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", required=True)
    parser.add_argument("--binary-reference", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fwhm-mm", type=float, nargs="+", default=[0, 2, 4, 6])
    args = parser.parse_args()
    fit = NetworkResult.load(args.extraction)
    for width in args.fwhm_mm:
        label = "fwhm_" + format(width, "g") + "mm"
        binary = postprocess_maps(fit, fwhm_mm=width, count_reference=args.binary_reference)
        binary.save(args.output_root, label=label)
        result = template_dice(binary, reference=args.binary_reference)
        result.save(args.output_root, label=label + "/template_dice")
        print(width, result.mean_dice)


if __name__ == "__main__":
    main()
