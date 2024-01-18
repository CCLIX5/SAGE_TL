# SAGE_TL

## Install
Please run the code in a Python 3.9 environment and install the required packages according to the `requirements.txt` file.

## Data
We use the MIMIC-IV dataset for illustration. Please follow the steps on [website](https://physionet.org/content/mimiciv/2.2/) to get access to this dataset.

## SAGE
The oringinal SAGE code see [here](https://github.com/iancovert/sage).

## uLSIF
The oringinal uLSIF code see [here](https://www.ms.k.u-tokyo.ac.jp/sugi/software.html). We upload the python version.

## Usage
`heter_split.R` is used to divide the data into two groups with different distributions.
`main.py` is what we use to test our proposed method.
The `rank figure` folder contains some code that we use to generate the rank plot (see [here](https://github.com/nyilin/Figures/tree/main) for details).

## References
Johnson, Alistair EW, et al. "MIMIC-IV, a freely accessible electronic health record dataset." *Scientific data* 10.1 (2023): 1.

Covert, Ian, Scott M. Lundberg, and Su-In Lee. "Understanding global feature contributions with additive importance measures." *Advances in Neural Information Processing Systems* 33 (2020): 17212-17223.

Kanamori, Takafumi, Shohei Hido, and Masashi Sugiyama. "A least-squares approach to direct importance estimation." *The Journal of Machine Learning Research* 10 (2009): 1391-1445.


