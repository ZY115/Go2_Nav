# ControlGain.msg 
# This message defines the control gains for a controller. It contains 
# two arrays of 36 floating-point numbers each, representing the proportional (kp) 
# and derivative (kd) gains for the controller.

# Optimization Status: optimal(0), primal infeasible(1), dual infeasible(2), or unknown(3)

uint8 status  
float64[36] kp
float64[36] kd