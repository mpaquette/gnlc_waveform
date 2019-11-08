# Gradient Non-Linearity Correction for Arbitrary Gradient Waveform


We currently support ["Topgaard style"](https://github.com/daniel-topgaard/md-dmri/blob/master/acq/bruker/paravision/make_waveform.m) and ["NOW style"](https://github.com/jsjol/NOW) waveform files.

**example1.py** loads a Topgaard waveform and numerically computes q-vector and B-tensor.

**example2.py** loads a NOW waveform, resample it, numerically computes q-vector and b-tensor then distort it with a GNL tensor and recompute numerically waveform, q-vector and B-tensor.  

**example3.py** loads a Topgaard waveform,  numerically computes q-vector and b-tensor then distort it with a GNL tensor and recompute numerically waveform, q-vector and B-tensor.  

**example4.py** loads a NOW (2 files style) waveform, resample it, numerically computes q-vector and b-tensor then distort it with a GNL tensor and recompute numerically waveform, q-vector and B-tensor.  




