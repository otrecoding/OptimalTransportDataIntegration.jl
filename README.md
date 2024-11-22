# OptimalTransportDataIntegration.jl

Data integration using optimal transport theory

## Simulations

### OTjoin

	- `lambda_reg` = 0:0.1:1
	- `maxrelax` = 0:0.1:2

### Unbalanced Modality

    - balanced : 
      + reg = 0
      + reg_m = 0 
    - unbalanced : 
      + rho = [0.01 0.05 0.1 0.25 0.5 0.75 1]
      + reg_m = 0
 
### Simple Learning

We launch `M=1000` simulations 
		
Data generation parameters

- `mA` = (0, 0, 0)
- `mB` = (1, 0, 0)
- `a` = (1, 1, 1, 1, 1, 1)
- `p` = 0.2 
- `nA` = `nB` = 1000
- `epsilon` = 0
