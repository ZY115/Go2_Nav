# 12D vector (ℝ¹²)
# Example use case: tracking error state
#   [0:3)   position error
#   [3:6)   orientation error
#   [6:9)   linear velocity error
#   [9:12)  angular velocity error

std_msgs/Header header
float64[12] data
