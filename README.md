###################################################################
#                                                                 #
#    pSVM-pin V1.0                                                #
#    Limeng Cui (lmcui932-at-gmail.com)                           #
#                                                                 #
###################################################################

1. Introduction.

pSVM-pin is a proportion learning framework with noise insensitivity.

Part of the Matlab code is supported on Felix X. Yu’s pSVM (felixyu.org/pSVM.html) and Xiaolin Huang's pin-SVM (http://www.esat.kuleuven.be/stadius/ADB/huang/softwarePINSVM.php).

If you use this toolbox, we appreciate it if you cite an appropriate subset of the following papers:

@article{shi2017learning,<br />
&nbsp;&nbsp;title={Learning from label proportions with pinball loss},<br />
&nbsp;&nbsp;author={Shi, Yong and Cui, Limeng and Chen, Zhensong and Qi, Zhiquan},<br />
&nbsp;&nbsp;journal={International Journal of Machine Learning and Cybernetics},<br />
&nbsp;&nbsp;pages={1--19},<br />
&nbsp;&nbsp;year={2017},<br />
&nbsp;&nbsp;publisher={Springer}<br />

###################################################################

2. License.

The software is made available for non-commercial research purposes only.

###################################################################

3. Installation.

a) This code is written for the Matlab interpreter (tested with versions R2014b). 

b) The code requires CVX, gurobi, and libsvm. 
http://cvxr.com/cvx/
http://www.gurobi.com/
http://www.csie.ntu.edu.tw/~cjlin/libsvm/
Please download the software in the above websites, and setup init.m accordingly.

###################################################################

4. Getting Started.

 - Make sure to carefully follow the installation instructions above.
 - Please see "demo_toy.m" to run demos and get basic usage information.

###################################################################

5. History.

Version 1.0 (2015/10/27)
 - initial version

###################################################################
