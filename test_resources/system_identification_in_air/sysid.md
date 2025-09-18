# Principle


# Comparison

## Metric Comparison and Position Error Analysis

Without system identification, the metric is higher, which means bigger gap in motor response between simulation and reality.

```
damping 0.0 
 stiffness 0.0 
 armature 0.0 
 kp tensor([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.],
       device='cuda:0') 
 kd tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000], device='cuda:0') 
 metric 2260.5503
 ```

 With system identification, the metric is lower:
 
 ```
 damping 0.29746901988983154 
 stiffness 0.011716208420693874 
 armature 0.01645776256918907 
 kp tensor([20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.],
       device='cuda:0') 
 kd tensor([0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000, 0.5000,
        0.5000, 0.5000, 0.5000], device='cuda:0') 
 metric 1090.7388
 ```
 Considering the data length as 11500, the metric per step is 1090/11500 = 0.0947. Since the metric is the norm of joint position errors across 12 joints, the square sum of position errors of 12 joints is 0.0947^2 = 0.00896809. Assume that all 12 joints have equal error, then the average square of position error for one joint is 0.00896809/12 = 0.0007473. The average position error for one joint is sqrt(0.0007473) = 0.0273.

## Delay Steps Analysis

