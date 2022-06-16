# HRASSISTant
<details><summary> <b>Installing libraries</b>  </summary>

First, install conda and create an environment. I suggest you specify the Python version, as newer versions aren't always bug-free. `conda create -n hr python=3.8`, then `conda activate hr`.

After that, it becomes OS oriented.

### Windows
The last step is `pip install -r requirements.txt`. 

Don't forget to install [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [cuDNN](https://developer.nvidia.com/cudnn).

### Mac
If you are on an ARM Mac, use `pip install -r requirements-mac.txt`. 

`Tensorflow` and `Transformers` will have to be installed separately, as they are trickier to install on M1 Macs. Follow [this link](https://towardsdatascience.com/hugging-face-transformers-on-apple-m1-26f0705874d7) for a complete tutorial.
</details>

