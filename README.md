<div align="center">

# Neural Modeling of Magnetic Tape Recorders

This repository contains the code for the research article:

```
O. Mikkonen, A. Wright, E. Moliner and V. Välimäki, “Neural Modeling Of Magnetic Tape Recorders,”
in Proceedings of the International Conference on Digital Audio Effects (DAFx),
Copenhagen, Denmark, 4-7 September 2023.
```

The pre-print of the article can be found from [here](https://arxiv.org/abs/2305.16862).<br>
The accompanying web page can be found from [here](http://research.spa.aalto.fi/publications/papers/dafx23-neural-tape/).<br>
The datasets can be found from [here](https://zenodo.org/record/8026272).

![System block diagram](./reel-to-reel.png?raw=true)

</div>

- [NEURAL MODELING OF MAGNETIC TAPE RECORDERS](#neural-modeling-of-magnetic-tape-recorders)
  - [SETUP](#setup)
  - [USAGE](#usage)
  - [CITATION](#citation)
  - [ACKNOWLEDGMENTS](#acknowledgments)

## SETUP

Clone the repository and submodules
```
git clone git@github.com:01tot10/neural-tape-modeling.git
cd neural-tape-modeling
git submodule init && git submodule update
```

Create the Python virtual environment with [mamba](https://mamba.readthedocs.io/en/latest/) (preferred) / [conda](https://docs.conda.io/en/latest/)
```
mamba/conda env create --file environment.yaml
```

Download data to symlinked location `audio/ -> ../neural-tape-audio/`
```
# create a directory for contents
mkdir ../neural-tape-audio
# download and extract toy data
wget -P ../neural-tape-audio 'https://zenodo.org/record/8026272/files/neural-tape-audio_CHOWTAPE.tar'
tar -xzvf ../neural-tape-audio/neural-tape-audio_CHOWTAPE.tar -C ../neural-tape-audio/
# download and extract real data
wget -P ../neural-tape-audio 'https://zenodo.org/record/8026272/files/neural-tape-audio_AKAI.tar'
tar -xzvf ../neural-tape-audio/neural-tape-audio_AKAI.tar -C ../neural-tape-audio/
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
