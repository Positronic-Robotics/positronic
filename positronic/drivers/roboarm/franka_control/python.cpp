#include <pybind11/pybind11.h>
#include "franka_control.h"

namespace py = pybind11;

PYBIND11_MODULE(_franka_control, m) {
    m.doc() = "Positronic's Franka control module";

    m.attr("VERSION") = positronic::hardware::franka_control::kVersion;

    // Expose the Controller class
    py::class_<positronic::hardware::franka_control::Controller>(m, "Controller")
        .def(py::init<const std::string&>())
        .def("start", &positronic::hardware::franka_control::Controller::start)
        .def("stop", &positronic::hardware::franka_control::Controller::stop)
        .def("__repr__",
            [](const positronic::hardware::franka_control::Controller& self) {
                return "Controller()";
            });
}
