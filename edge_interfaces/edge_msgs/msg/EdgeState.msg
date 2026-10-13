# ============================================================
# EdgeState.msg
#
# Unified robot state message.
# Supports multiple state semantics with flexible dimensions.
# ============================================================

std_msgs/Header header
# stamp: time when state is generated
# frame_id: reference frame (e.g., "base_link", "odom")

# ============================================================
# State Type (State Space Definition)
# ============================================================
uint8 ROBOT_STATE=0       # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
uint8 TRACKING_ERROR=1    # delta state vector (same layout as ROBOT_STATE)

uint8 state_type
# Specifies which state space state_value belongs to.

# ============================================================
# State Value
# ============================================================
float64[] state_value
# State vector corresponding to state_type.
#
# Expected dimensions:
# - ROBOT_STATE     → length = 12
# - TRACKING_ERROR  → length = 12
