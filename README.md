<div align="center">

# Neural Modeling of Magnetic Tape Recorders

This repository contains the code for the research article:

```
“Neural Modeling Of Magnetic Tape Recorders,”
in Proceedings of the International Conference on Digital Audio Effects (DAFx),
Copenhagen, Denmark, 4-7 September 2023.
```
<!-- [![Demo](https://img.shields.io/badge/Web-Demo-blue)](https://csteinmetz1.github.io/DeepAFx-ST)
[![arXiv](https://img.shields.io/badge/arXiv-2207.08759-b31b1b.svg)](https://arxiv.org/abs/2207.08759) -->

The pre-print of the article can be found from [here](https://arxiv.org/abs/2305.16862).<br>
The accompanying web page can be found from [here](http://research.spa.aalto.fi/publications/papers/dafx23-neural-tape/).<br>
The datasets can be found from [here](http://www.zenodo.org).

![alt text](./reel-to-reel.png?raw=true)

</div>

- [NEURAL MODELING OF MAGNETIC TAPE RECORDERS](#neural-modeling-of-magnetic-tape-recorders)
  - [SETUP](#setup)
  - [USAGE](#usage)
  - [CITATION](#citation)
  - [ACKNOWLEDGMENTS](#acknowledgments)

## SETUP

Clone the repository and submodules
```
git clone [REPOSITORY]
cd neural-tape-model
git submodule init && git submodule update
```

Create the Python virtual environment with [mamba](https://mamba.readthedocs.io/en/latest/) (preferred) / [conda](https://docs.conda.io/en/latest/)
```
mamba/conda env create --file environment.yaml
```

Download and symlink the data to `audio/`
```
wget -O neural-tape-audio.zip [ZENODO URL]
unzip -d ../neural-tape-audio/ {file.zip}
```

Optional: To generate target audio with [CHOWTape](https://github.com/jatinchowdhury18/AnalogTapeModel), a VST instance of the plugin should be compiled. Check instructions in the corresponding repository.

## USAGE

The folder `scripts/` contains the various processing pipelines for interacting with the system, as well as a separate `README.md` with instructions.

## CITATION

Cite the work as follows
```
@conference{mikkonen_neural_2023,
  title = {Neural Modeling of Magnetic Tape Recorders},
  booktitle = {Proceedings of the {{International Conference}} on {{Digital Audio Effects}} ({{DAFx}})},
  author = {Mikkonen, Otto and Wright, Alec and Moliner, Eloi and V{\"a}lim{\"a}ki, Vesa},
  year = {2023},
  month = sep,
  address = {{Copenhagen, Denmark}}
}
```

## ACKNOWLEDGMENTS

:black_heart::black_heart::black_heart:
- VST instance of a reel-to-reel tape machine: [CHOWTape](https://github.com/jatinchowdhury18/AnalogTapeModel)
- Python VST wrapper: [pedalboard](https://github.com/spotify/pedalboard)
- Dataloader extended and customized from [microtcn](https://github.com/csteinmetz1/micro-tcn)
- Error-to-signal ratio (ESR) loss from [Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)
- ESR loss with DC blocker from [GreyBoxDRC](https://github.com/Alec-Wright/GreyBoxDRC)
- Multi-resolution short-time Fourier transform (STFT) loss from [auraloss](https://github.com/csteinmetz1/auraloss)-library
- Codebase kept clean with with [yapf](https://github.com/google/yapf), [isort](https://github.com/pycqa/isort/), [pylint](https://github.com/pylint-dev/pylint) and [beautysh](https://github.com/lovesegfault/beautysh)

:black_heart::black_heart::black_heart: