"""Inspect engineered sensory-channel overlap; this is not a biological claim."""
import json
from pathlib import Path
import numpy as np
from scripts.compare_readout_v2 import RawMotors


def main():
    controller = RawMotors(verbose=False)
    e = controller.encoder_coeff
    norms = np.linalg.norm(e, axis=0)
    overlap = (e.T @ e) / np.outer(norms, norms)
    report = dict(axes=['roll','pitch','yaw'], sensory_neurons=len(e),
                  rank=int(np.linalg.matrix_rank(e)), cosine_overlap=overlap.tolist(),
                  note='Overlap of programmed sensory stimulation patterns. '
                  'It can make decoding harder but does not establish its cause of failure.')
    path = Path('results/readout_v2/encoder_overlap.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
