# Third-party data, software and images

The root MIT license applies to original NeuroFlight code and documentation.
It does not relicense upstream data, images or dependencies.

## MaleCNS

Source: https://male-cns.janelia.org/ and https://male-cns.janelia.org/download/.
The project identifies the dataset as **CC BY 4.0**:
https://creativecommons.org/licenses/by/4.0/.

Credit: FlyEM Project Team at HHMI Janelia Research Campus, University of Cambridge
(Department of Zoology), MRC Laboratory of Molecular Biology, and Google Research.

The downloaded v1.0 annotations, neurotransmitters and connectome weights are not
bundled. NeuroFlight filters these inputs and builds a local cache. Its LIF dynamics,
sensory encoding and motor readout are modeling choices, not upstream biological
validation or endorsement.

The README embeds the unmodified anatomical image directly from
https://male-cns.janelia.org/_static/mcns_all_neurons2_lateral.png,
shown on the project homepage and https://male-cns.janelia.org/media/.
It is externally hosted rather than bundled (the original is approximately 7 MB).
Credit and CC BY 4.0 attribution above apply. The gallery supplies no separate
individual creator for this still image. Do not attribute the still to a video
creator or describe it as activity from a NeuroFlight simulation.

## MuJoCo and Crazyflie

MuJoCo: https://github.com/google-deepmind/mujoco, Apache-2.0.
The simulator is installed as a dependency, not vendored here.

Crazyflie model: https://github.com/google-deepmind/mujoco_menagerie/tree/main/bitcraze_crazyflie_2.
Its license is MIT, copyright (c) 2024 whoenig. The model is installed through
MuJoCo Menagerie; model files are not copied into this repository.
The figure `docs/assets/drone_roll_replay.png` renders that model using logged
NeuroFlight roll angles. The original model license is retained in
`docs/licenses/CRAZYFLIE-MIT.txt` for attribution with the rendering.

`docs/assets/roll_comparison.png` is generated from this repository's saved
experimental traces. Both generated figures can be recreated with
`python -m scripts.render_readme_media`. No generated anatomical artwork is used.
