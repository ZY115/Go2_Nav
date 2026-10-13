import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

class DebugPCD(Node):
    def __init__(self):
        super().__init__("debug_pcd")
        self.sub = self.create_subscription(
            PointCloud2,
            "/scan/points",
            self.cb,
            10
        )

    def cb(self, msg):
        print("Fields:")
        for f in msg.fields:
            print(f"  -", f.name, f.datatype, f.offset)
        print("Point step:", msg.point_step)
        print("Row step:", msg.row_step)
        print("Width:", msg.width)
        print("Height:", msg.height)
        print("-------")

def main():
    rclpy.init()
    rclpy.spin(DebugPCD())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
