# NALU-tensorflow
A Tensorflow Implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508)

# Environment
Ubuntu 18<br/>
Tensorflow 1.10.1<br/>
numpy 1.14.5<br/>

# Result
- **Synthetic Arithmetic Tasks**

|NALU|
|op|interpolation|exterpolation|
|-----|-----|-----|
|a+b| 0.0000008 | 0.0000006 |
|a-b| 0.0000002 | 0.0000006 |
|a*b| 0.0000000 | 0.0000004 |
|a/b| 0.0059779 | 0.0387195 |
|a^2| 0.0000001 | 0.0000130 |
|sqrt(a)| 0.0000138 | 0.0000194 |


- **MNIST Counting**
 
