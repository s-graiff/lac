# Fairing of planar curves to LAC

Source code of the algorithm presented in
[[1]](https://doi.org/10.1007/s13160-023-00567-w), to fair a planar
curve to its closets Log-aesthetic curve, in a $L^2$-distance sense.

## Requirements

Python 3 (tested on v3.9.15) with

```
numpy==1.21.2
scipy==1.7.1
cyipopt==1.1.0
matplotlib==3.4.3
```

## How to use

- For the synthetic data, run
  
  ```bash
  $ python main.py
  ```
  
- For the Prius data, first run

  ```bash
  $ python preprocess-rotate.py
  ```
  
  and then

  ```bash
  $ python main_prius.py
  ```


## Reference

<a name="2023GZ"> [1]
    S. E. Graiff Zurita, K. Kajiwara and K. T. Miura. 
    Fairing of planar curves to log-aesthetic curves. 
    Japan J. Indust. Appl. Math. 40, 1203–1219 (2023). 
    [https://doi.org/10.1007/s13160-023-00567-w](https://doi.org/10.1007/s13160-023-00567-w)
</a>
