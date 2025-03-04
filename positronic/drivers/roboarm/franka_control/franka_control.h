#pragma once

#include <franka/robot.h>
#include <franka/robot_state.h>
#include <franka/model.h>
#include <thread>
#include <atomic>
#include <Eigen/Dense>

namespace positronic::hardware::franka_control {

extern const char* kVersion;

class Controller {
public:
    explicit Controller(const std::string& fci_hostname);
    virtual ~Controller() noexcept;

    void start();
    void stop();

private:
    std::unique_ptr<franka::Robot> robot_;
    std::unique_ptr<franka::Model> model_;
    std::atomic<bool> running_{false};
    std::thread control_thread_;

    // Impedance control parameters
    Eigen::Matrix<double, 6, 6> stiffness_;
    Eigen::Matrix<double, 6, 6> damping_;
    Eigen::Vector3d position_d_;
    Eigen::Quaterniond orientation_d_;

    void setupImpedanceParams();
};

}
