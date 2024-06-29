import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelStates

class ModelStateChecker(Node):
    def __init__(self):
        super().__init__('model_state_checker')
        self.model_states_subscriber = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_states_callback,
            10
        )

    def model_states_callback(self, msg):
        try:
            self.get_logger().info(f"Received ModelStates message with {len(msg.name)} models.")
            for i, name in enumerate(msg.name):
                self.get_logger().info(f"Model {i}: {name}, Position: {msg.pose[i].position}")
            if "target_box" in msg.name:
                index = msg.name.index("target_box")
                self.get_logger().info(f"Box detected at position: {msg.pose[index].position}")
            else:
                self.get_logger().warning("Box model not found in the model states.")
        except Exception as e:
            self.get_logger().error(f"Error in model_states_callback: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ModelStateChecker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
