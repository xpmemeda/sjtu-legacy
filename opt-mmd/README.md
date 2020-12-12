Code for the paper "Generative Models and Model Criticism via Optimized Maximum Mean Discrepancy" ([arXiv:1611.04488](https://arxiv.org/abs/1611.04488); [published at](https://openreview.net/forum?id=HJWHIKqgl) ICLR 2017), by [Dougal J. Sutherland](http://www.gatsby.ucl.ac.uk/~dougals/) ([@dougalsutherland](https://github.com/dougalsutherland)), [Hsiao-Yu Tung](http://sfish0101.bitbucket.io/), [Heiko Strathmann](http://herrstrathmann.de/about/) ([@karlnapf](https://github.com/karlnapf)), Soumyajit De ([@lambday](https://github.com/lambday)), [Aaditya Ramdas](https://people.eecs.berkeley.edu/~aramdas/), [Alex Smola](https://alex.smola.org/), and [Arthur Gretton](http://www.gatsby.ucl.ac.uk/~gretton/).

- Implementations of the variance estimator are in Theano in [`two_sample/mmd.py`](two_sample/mmd.py) and in Tensorflow in [`gan/mmd.py`](gan/mmd.py).
- General code for learning kernels for a fixed two-sample test, with Theano, is in [two_sample](two_sample).
- Code for the GAN variants, using TensorFlow, is in [gan](gan).
- Code for the efficient permutation test described in Section 3 is in the 6.0 release of [Shogun](http://shogun.ml); look under [`shogun/src/shogun/statistical_testing`](https://github.com/shogun-toolbox/shogun/tree/develop/src/shogun/statistical_testing). An example of using it in the Python API is in [`two_sample/mmd_test.py`](two_sample/mmd_test.py).

This code is under a BSD license, but if you use it, please cite the paper.
