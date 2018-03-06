# Transfer-Stacking-Gaussian-Process

TSGP (Transfer Stacking Gaussian Process) is an algorithm that adaptively stacks pre-built gaussian process models from both the source and target domains
in order to improve the predictive performance of the target regression problem.

TSGP takes as inputs (x_target,y_target,source_models), accepted inputs are shaped (n,d); n = # of instances, d = dimensions
Accepted outputs are shaped (n,1), source_models are cells of RegressionGP models.

Example Use:
```matlab
model = TSGP(x_target,y_target,source_models)
yhat = model.predict(x_test)  
```
