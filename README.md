# Gradient Non-Linearity Correction for Arbitrary Gradient Waveform


We currently support ["Topgaard style"](https://github.com/daniel-topgaard/md-dmri/blob/master/acq/bruker/paravision/make_waveform.m) and ["NOW style"](https://github.com/jsjol/NOW) waveform files.




**gnl_b_tensor.py** Has the basic math of the problem of GNL for tensor valued diffusion encoding. It contains the gist of the closed-form formula for the distorted B-tensor.

**proof_helper.py** Contains the symbolic math computation of the earlyclosed-form derivation.

**btensor.py** contains the functions to compute q-vectors and B-tensor numerically and analytically and distort them with GNL.

**gnl_tensor.py** contains the GNL tensor computation formula.

**viz.py** contains quick vizualisation functions for waveform and q-vectors.

**io_waveform.py** Contains reading function for gradient waveforms.

Various (overlapping) example serves as documentation for the various functions. 

**example1.py** loads a Topgaard waveform and numerically computes q-vector and B-tensor.

**example2.py** loads a NOW waveform, resample it, numerically computes q-vector and b-tensor then distort it with a GNL tensor and recompute numerically waveform, q-vector and B-tensor.  

**example3.py** loads a Topgaard waveform,  numerically computes q-vector and b-tensor then distort it with a GNL tensor and recompute numerically waveform, q-vector and B-tensor.  

**example4.py** loads a NOW (2 files style) waveform, resample it, numerically computes q-vector and b-tensor then distort it with a GNL tensor and recompute numerically waveform, q-vector and B-tensor.  
**example5.py** Comparaison of the close-form B-tensor distortion formula to the numerical approximation for random GNL tensor.  

**exampleGNL1.py** Shows b-value and related metric acorss a full brain with Connectom level GNl tensor.





