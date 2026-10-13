# ============================================================
# EdgeAction.msg
#
# Unified control action message.
# Supports multiple control spaces with flexible dimensions.
# ============================================================

std_msgs/Header header
# stamp: time when action is generated
# frame_id: reference frame (e.g., "base_link")

# ============================================================
# Action Type (Control Space Definition)
# ============================================================
uint8 BASE_ACCEL=0      # [ax, ay, az, wx, wy, wz]  (6D base acceleration)
uint8 CMD_VEL=1         # [vx, vy, wz]              (3D planar velocity)
uint8 JOINT_POS=2       # Joint position targets    (#joints)
uint8 JOINT_TORQUE=3    # Joint torque commands     (#joints)

uint8 action_type
# Specifies which control space action_value belongs to.

# ============================================================
# Action Value
# ============================================================
float64[] action_value
# Control vector corresponding to action_type.
#
# Expected dimensions:
# - BASE_ACCEL   → length = 6
# - CMD_VEL      → length = 3
# - JOINT_POS    → length = number_of_joints
# - JOINT_TORQUE → length = number_of_joints

# ============================================================
# Action Mode (Action Source)
# ============================================================
uint8 STUDENT=0     # Learned policy output
uint8 TEACHER=1     # Expert / MPC / reference controller

uint8 action_mode
# Indicates who generated this action.