# pytorch implementation of DNN-HSMM for TTS
- This software is distributed under the BSD 3-Clause license. Please see LICENSE for more details.
- Paper: Keiichi Tokuda, Kei Hashimoto, Keiichiro Oura, and Yoshihiko Nankaku, "Temporal modeling in neural network based statistical parametric speech synthesis,'' 9th ISCA Speech Synthesis Workshop, pp. 113-118, September, 2016. http://ssw9.talp.cat/papers/ssw9_OS2-2_Tokuda.pdf

# Requirements
- python >= 3.7
- numpy
- scipy
- pytorch >= 1.6 (https://pytorch.org/)
- GNU parallel
- SPTK == 3.11 (http://sp-tk.sourceforge.net/)

# Usage
- By running 00_data.sh, you can create serialized training and test data (npz) from pre-prepared linguistic and acoustic features (lab, lf0, mgc, bap). Directory names (dnames) and dimentions (dims) written in 00_data.sh need to be modified.
- By running 01_run.py, you can train a model and generate acoustic features (featdims written in Config.py need to be modified). You can find generated features in 'gen' directory.

# Demo
- Japanese (m001)
```
$ cd DNN-HSMM
$ wget https://xxxx/demo_data_Japanese.tar.gz
$ tar -zxvf demo_data_Japanese.tar.gz
$ cp demo_data_Japanese/00_data.sh .
$ cp demo_data_Japanese/Config.py .
$ bash 00_data.sh
$ python 01_run.py
```
- English (slt)
```
$ cd DNN-HSMM
$ wget https://xxxx/demo_data_English.tar.gz
$ tar -zxvf demo_data_English.tar.gz
$ cp demo_data_English/00_data.sh .
$ cp demo_data_English/Config.py .
$ bash 00_data.sh
$ python 01_run.py
```

# Who we are
- Shinji Takaki (http://www.sp.nitech.ac.jp/~takaki/)
- Kei Hashimoto (http://www.sp.nitech.ac.jp/~bonanza/)
- Keiichiro Oura (http://www.sp.nitech.ac.jp/~uratec/)
- Yoshihiko Nankaku (http://www.sp.nitech.ac.jp/~nankaku/)
- Keiichi Tokuda (http://www.sp.nitech.ac.jp/~tokuda/)
