#Global Optimization Benchmark#

This benchmark is based (but modified) on the setup described in Global optimization of lipschitz functions (C. Malherbe et. al. 2017) and used in a poster submission of a new global optimization approach called gradopt. We publish it here for the poster submission "Graduated Optimization of Black-Box Function" 2019 @ ICML Workshop.

#Usage#
The experiment consists of two different parts:

Experiment.py - running the optimizers on the problems, collecting the data.
Experiment.ipynb - evaluating the collected data.

The benchmark specifics settings can be configured in /experiment/config.py. It is preconfigured to run out-of-the box the whole benchmark on all processors. For testing, set DEBUG_SETTINGS_OVERWRITE=True in config.py that enables a very short function test.

We recommend executing the experiment by running a docker container via the following command (modify TARGETDIR to where you checked out this repo):
export TARGETDIR=GlobalOptLipschitz
docker run -d -it --name=$USER-$(uuidgen)-GlobalLipschitz --log-opt max-size=10m -v /home/$USER/$TARGETDIR/py_src:/app sklearn

The results will be saved in a json file (results.json) that will be opened by the jupyther notebook "Experiments.ipynb". The notebook will read the results, analyzes it (calculate target values) and outputs the refined results as a html table.

#Extention#
If you want to add your own optimizer, just implement a wrapper in ./experiment/optimizers.py inheriting from OptimizerWrapper. All optimizers that inherit from that class are loaded when Experiment.py is started and will be executed during the experiment.

#Acknowledgement#
This work is supported in part by the German Federal Ministry of Education and Research
(BMBF) under the grant number 01IS16046.



