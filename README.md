# Transfer Learning for Global Feature Importance Measurements

Python and R code for reproducing the experiment. The original code of SAGE is available [here](https://github.com/iancovert/sage), and the original uLSIF code in Matlab is available [here](https://www.ms.k.u-tokyo.ac.jp/sugi/software.html), from which we adapted the current Python version.

## System Requirements

- **R**: version 4.2.1.
- **Python**: version 3.9.

  Use `requirements.txt` to install the required Python packages.

## Data
The dataset used can be obtained by first accessing the original MIMIC-IV-ED data [here](https://physionet.org/content/mimiciv/2.2/) and then following the processing pipelines in this [paper](https://www.nature.com/articles/s41597-022-01782-9).

## Usage
- `heter_split.R`: partitioning the data into two groups with different distributions by a given variable (e.g., age).
- `main.py`: testing our proposed method.
- `rank figure` folder: code used to generate the rank plot (see [this repo](https://github.com/nyilin/Figures/tree/main) for details).


## References
Johnson, Alistair EW, et al. "MIMIC-IV, a freely accessible electronic health record dataset." *Scientific data* 10.1 (2023): 1.

Covert, Ian, Scott M. Lundberg, and Su-In Lee. "Understanding global feature contributions with additive importance measures." *Advances in Neural Information Processing Systems* 33 (2020): 17212-17223.

Kanamori, Takafumi, Shohei Hido, and Masashi Sugiyama. "A least-squares approach to direct importance estimation." *The Journal of Machine Learning Research* 10 (2009): 1391-1445.


