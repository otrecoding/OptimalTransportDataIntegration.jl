# OptimalTransportDataIntegration.jl

Data integration using optimal transport theory

## Simulations

We launch `M=1000` simulations 
		
Data generation parameters

- `mA` = (0, 0, 0)
- `mB` = (1, 0, 0)
- `a` = (1, 1, 1, 1, 1, 1)
- `p` = 0.2 
- `nA` = `nB` = 1000
- `epsilon` = 0


## Parameters for OTjoin method

- `lambda_reg` = 0:0.1:1
- `maxrelax` = 0:0.1:2

## Parameters for Unbalanced Modality

- balanced : 
  + `reg` = 0.0
  + `reg_m` = 0.0 
- unbalanced : 
  + `reg` = [0.001, 0.01, 0.1]
  + `reg_m` = [0.01 0.05 0.1 0.25 0.5 0.75 1]

## Parameters for Simple Learning

- `hidden_layer_size` =  10
- `learning_rate` = 0.01
- `batchsize` =  64
- `epochs` = 1000
- `loss` = `logitcrossentropy`
- 
