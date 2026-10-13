#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class H264Repub : public rclcpp::Node
{
public:
    H264Repub()
    : Node("h264_repub")
    {
        // -----------------------------
        // Declare parameters
        // -----------------------------
        this->declare_parameter<std::string>("multicast_iface", "eth0");
        this->declare_parameter<std::string>("multicast_address", "230.1.1.1");
        this->declare_parameter<int>("port", 1720);
        this->declare_parameter<int>("width", 1280);
        this->declare_parameter<int>("height", 720);
        this->declare_parameter<double>("publish_rate", 100.0);   // Hz

        // -----------------------------
        // Get parameters
        // -----------------------------
        multicast_iface_ = this->get_parameter("multicast_iface").as_string();
        multicast_address_ = this->get_parameter("multicast_address").as_string();
        port_ = this->get_parameter("port").as_int();
        width_ = this->get_parameter("width").as_int();
        height_ = this->get_parameter("height").as_int();
        publish_rate_ =
            this->get_parameter("publish_rate").as_double();

        if (publish_rate_ <= 0.0) {
            RCLCPP_WARN(this->get_logger(),
                        "publish_rate <= 0, defaulting to 30 Hz");
            publish_rate_ = 30.0;
        }

        RCLCPP_INFO(this->get_logger(),
                    "iface=%s address=%s port=%d rate=%.2f Hz",
                    multicast_iface_.c_str(),
                    multicast_address_.c_str(),
                    port_,
                    publish_rate_);

        // -----------------------------
        // Publisher
        // -----------------------------
        image_publisher_ =
            this->create_publisher<sensor_msgs::msg::Image>(
                "/camera/image/raw", 10);

        // -----------------------------
        // GStreamer pipeline
        // -----------------------------
        std::string gstreamer_str =
            "udpsrc address=" + multicast_address_ +
            " port=" + std::to_string(port_) +
            " multicast-iface=" + multicast_iface_ +
            " ! application/x-rtp, media=video, encoding-name=H264 "
            "! rtph264depay ! h264parse ! avdec_h264 "
            "! videoconvert "
            "! video/x-raw,width=" + std::to_string(width_) +
            ",height=" + std::to_string(height_) +
            ",format=BGR "
            "! appsink drop=1";

        cap_.open(gstreamer_str, cv::CAP_GSTREAMER);

        if (!cap_.isOpened()) {
            RCLCPP_ERROR(this->get_logger(),
                         "Failed to open video capture");
        }

        // -----------------------------
        // Create timer based on publish_rate
        // -----------------------------
        auto period =
            std::chrono::duration<double>(1.0 / publish_rate_);

        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(period),
            std::bind(&H264Repub::timer_callback, this));
    }

private:
    void timer_callback()
    {
        if (!cap_.isOpened())
            return;

        cv::Mat frame;
        if (cap_.read(frame)) {
            auto ros_image =
                cv_bridge::CvImage(
                    std_msgs::msg::Header(),
                    "bgr8",
                    frame)
                    .toImageMsg();

            ros_image->header.frame_id = "camera";
            ros_image->header.stamp = this->get_clock()->now();

            image_publisher_->publish(*ros_image);
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    cv::VideoCapture cap_;

    std::string multicast_iface_;
    std::string multicast_address_;
    int port_;
    int width_;
    int height_;
    double publish_rate_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<H264Repub>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}