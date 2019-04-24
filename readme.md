#Global Optimization Benchmark#

This benchmark is based on the setup described in Global optimization of lipschitz functions (C. Malherbe et. al. 2017) and used in a poster submission of a new global optimization approach called gradopt. We publish it here for the poster submission "Graduated Optimization of Black-Box Function" 2019 @ ICML Workshop.

#Usage#
The experiment consists of two different parts:

Experiment.py - running the optimizers on the problems, collecting the data.
Experiment.ipynb - evaluating the collected data.

The benchmark specifics settings can be configured in /experiment/config.py. It is preconfigured to run out-of-the box the whole benchmark on all processors. For testing, use the debug flag in config.py that enables a very short function test.

The benchmark can be run by running a docker container via the following command (modify TARGETDIR to where you checked out this repo):
export TARGETDIR=GlobalOptLipschitz
docker run -d -it --name=$USER-$(uuidgen)-GlobalLipschitz --log-opt max-size=10m -v /home/$USER/$TARGETDIR/py_src:/app sklearn

The results will be saved in a json file that will be opened by the jupyther notebook. The notebook will read the results and outputs a html table containing all the numbers.



