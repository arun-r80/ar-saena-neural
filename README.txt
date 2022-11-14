My Neural Network playbook

Instructions to install this playbook

As this code was developed and committed on a Apple M1 chipset based Macbook, the tensorflow libraries were installed with
support for this chipset. This seemingly simple dependency injection is actually tedious, if we do not understand that pip installer does not support M1-native tensorflow package, and hence it is imperative that we use conda package manager.
The following steps would help to understand how this installation can be achieved.

a. M1-native tensorflow packages are supplied by Miniforge3, so it is imperative that Miniforge3 is installed in the system.
 i. Install Miniforge3 using Homebrew -
    - Homebrew can be installed using the instructions in their website.
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    - Use Homebrew to install Miniforge3
    brew install miniforge3
 ii. Another option to install Miniforge3 is to just follow the documentation given in their page: https://github.com/conda-forge/miniforge

 b.  Install xcode tools utilities if not done already.
 xcode-select --install

 c. We need to use the conda distribution that comes with Miniforge3 to move ahead. This is usuall present under /Users/<username>/miniforge/

 d. We will use the yml file, like the one attached to create a new conda environment. The conda environment will have M1-native packages.

 conda create -n <envname> -f tensorflow-apple-metal-conda_1.yml

 e. The conda env needs to be activated, to get all the packages mentioned :)
 conda activate <envname>

 We are all set to install this playbook now!!