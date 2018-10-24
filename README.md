# NALU-tensorflow
A Tensorflow Implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508)

# Environment
Ubuntu 18<br/>
Tensorflow 1.10.1<br/>
numpy 1.14.5<br/>

# Result
- **Synthetic Arithmetic Tasks**

**NALU**

|op|interpolation|extrapolation|
|-----|-----|-----|
|a+b| 0.0000008 | 0.0000006 |
|a-b| 0.0000002 | 0.0000006 |
|a*b| 0.0000000 | 0.0000004 |
|a/b| 0.0059779 | 0.0387195 |
|a^2| 0.0000001 | 0.0000130 |
|sqrt(a)| 0.0000138 | 0.0000194 |

**NAC**

|op|interpolation|extrapolation|
|-----|-----|-----|
|a+b|0.000000|0.000000|
|a-b|0.000000|0.000000|
|a*b|607877.875000|3598053.750000|
|a/b|548.708923|523.257935|
|a^2|1175723.500000|35157152.000000|
|sqrt(a)|0.977279|18.971621|

- **MNIST Counting**
 
