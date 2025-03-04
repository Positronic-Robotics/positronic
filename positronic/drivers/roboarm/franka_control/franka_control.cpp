#include "franka_control.h"

#include <atomic>
#include <franka/exception.h>
#include <array>
#include <cmath>
#include <iostream>

namespace positronic::hardware::franka_control {

const char* kVersion = "0.1.0";

Controller::Controller(const std::string& fci_hostname)
    : robot_(std::make_unique<franka::Robot>(fci_hostname, franka::RealtimeConfig::kIgnore)),
      model_(std::make_unique<franka::Model>(robot_->loadModel())),
      running_(false)
{
    setupImpedanceParams();
}

Controller::~Controller() noexcept {
    stop();
}

void Controller::setupImpedanceParams() {
    // Compliance parameters
    const double translational_stiffness{150.0};
    const double rotational_stiffness{10.0};

    stiffness_.setZero();
    stiffness_.topLeftCorner(3, 3) = translational_stiffness * Eigen::Matrix3d::Identity();
    stiffness_.bottomRightCorner(3, 3) = rotational_stiffness * Eigen::Matrix3d::Identity();

    damping_.setZero();
    damping_.topLeftCorner(3, 3) = 2.0 * sqrt(translational_stiffness) * Eigen::Matrix3d::Identity();
    damping_.bottomRightCorner(3, 3) = 2.0 * sqrt(rotational_stiffness) * Eigen::Matrix3d::Identity();
}

void Controller::start() {
    if (running_) return;

    // Set collision behavior
    robot_->setCollisionBehavior(
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}},
        {{100.0, 100.0, 100.0, 100.0, 100.0, 100.0}});

    // Get initial state for equilibrium point
    franka::RobotState initial_state = robot_->readOnce();
    Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));
    position_d_ = initial_transform.translation();
    orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());

    running_ = true;
    control_thread_ = std::thread([this]() {
        try {
            robot_->control([this](const franka::RobotState& robot_state,
                                 franka::Duration /*duration*/) -> franka::Torques {
                // Get state variables
                std::array<double, 7> coriolis_array = model_->coriolis(robot_state);
                std::array<double, 42> jacobian_array =
                    model_->zeroJacobian(franka::Frame::kEndEffector, robot_state);

                // Convert to Eigen
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
                Eigen::Map<const Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
                Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
                Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
                Eigen::Vector3d position(transform.translation());
                Eigen::Quaterniond orientation(transform.rotation());

                // Compute error
                Eigen::Matrix<double, 6, 1> error;
                error.head(3) = position - position_d_;

                // Orientation error
                if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
                    orientation.coeffs() = -orientation.coeffs();
                }
                Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
                error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
                error.tail(3) = -transform.rotation() * error.tail(3);

                // Compute control
                Eigen::VectorXd tau_task(7), tau_d(7);
                tau_task = jacobian.transpose() * (-stiffness_ * error - damping_ * (jacobian * dq));
                tau_d = tau_task + coriolis;

                std::array<double, 7> tau_d_array{};
                Eigen::VectorXd::Map(&tau_d_array[0], 7) = tau_d;

                if (!running_) {
                    return franka::MotionFinished(franka::Torques(tau_d_array));
                }
                return tau_d_array;
            }, true);
        } catch (const franka::Exception& e) {
            std::cerr << "Franka control loop error: " << e.what() << std::endl;
            running_ = false;
        }
    });
}

void Controller::stop() {
    running_ = false;
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
}

}
