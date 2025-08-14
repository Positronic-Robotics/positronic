#!/usr/bin/env python3
"""
URDF Generator for Parametric Robotic Arms.

This script generates URDF files for robotic arms based on configurable parameters
including link dimensions, motor specifications, and joint configurations.
"""

import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Sequence, Tuple
from dataclasses import dataclass


@dataclass
class LinkParameters:
    """Parameters for a single link in the robotic arm."""
    length: float  # Length of the link cylinder (m)
    radius: float  # Radius of the link cylinder (m)
    mass: float   # Mass of the link (kg)


@dataclass
class MotorParameters:
    """Parameters for a single motor in the robotic arm."""
    radius: float  # Radius of the motor cylinder (m)
    height: float  # Height of the motor cylinder (m)
    mass: float   # Mass of the motor (kg)
    effort_limit: float = 100.0  # Nm
    velocity_limit: float = 2.0  # rad/s


@dataclass
class JointConfiguration:
    """Configuration for a single joint."""
    name: str
    joint_type: str = "revolute"
    axis: Tuple[float, float, float] = (0, 0, 1)
    origin_xyz: Tuple[float, float, float] = (0, 0, 0)
    origin_rpy: Tuple[float, float, float] = (0, 0, 0)
    effort_limit: float = 100.0
    velocity_limit: float = 2.0


def calculate_center_of_mass(masses, positions):
    """
    Calculate center of mass for multiple components.

    Args:
        masses: (list) List of masses for each component
        positions: (list) List of position vectors for each component's center of mass

    Returns:
        numpy.ndarray: Center of mass position vector
    """
    total_mass = sum(masses)
    weighted_positions = [m * np.array(pos) for m, pos in zip(masses, positions)]
    center_of_mass = sum(weighted_positions) / total_mass
    return center_of_mass, total_mass


def cylindrical_inertia(mass, radius, height):
    """
    Calculate inertia matrix for cylinder about its center of mass.
    Assumes z-axis is along the cylinder axis.

    Args:
        mass: (float) Mass in kg
        radius: (float) Radius in meters
        height: (float) Height in meters

    Returns:
        numpy.ndarray: 3x3 inertia matrix
    """
    Ixx = Iyy = (1/12) * mass * (3 * radius**2 + height**2)
    Izz = (1/2) * mass * radius**2

    return np.array([
        [Ixx, 0, 0],
        [0, Iyy, 0],
        [0, 0, Izz]
    ])


def parallel_axis_theorem(I_cm, mass, displacement):
    """
    Apply parallel axis theorem to translate inertia matrix.

    Args:
        I_cm: (numpy.ndarray) Inertia matrix about center of mass
        mass: (float) Mass in kg
        displacement: (numpy.ndarray) Displacement vector [x, y, z]

    Returns:
        numpy.ndarray: Translated inertia matrix
    """
    d = np.array(displacement)
    d_squared = np.dot(d, d)

    # Outer product matrix
    d_outer = np.outer(d, d)

    # Parallel axis theorem: I_new = I_cm + m * (d²*I - d⊗d)
    I_translated = I_cm + mass * (d_squared * np.eye(3) - d_outer)

    return I_translated


def calculate_composite_inertia(
        motor_params: MotorParameters,
        link_params: LinkParameters,
        link_offset: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Calculate composite inertia for motor + link combination.

    Args:
        motor_params: Motor specifications
        link_params: Link specifications
        link_offset: Offset of link center from motor center (m)

    Returns:
        Tuple of (inertia_matrix, total_mass, center_of_mass)
    """
    # Component positions (motor at origin, link offset)
    motor_position = np.array([0.0, 0.0, 0.0])
    link_position = np.array([0.0, 0.0, link_offset])

    # Calculate composite center of mass
    masses = [motor_params.mass, link_params.mass]
    positions = [motor_position, link_position]
    composite_com, total_mass = calculate_center_of_mass(masses, positions)

    # Calculate individual inertias about their centers of mass
    I_motor_cm = cylindrical_inertia(motor_params.mass, motor_params.radius, motor_params.height)
    I_link_cm = cylindrical_inertia(link_params.mass, link_params.radius, link_params.length)

    # Calculate displacements from composite center of mass
    motor_displacement = motor_position - composite_com
    link_displacement = link_position - composite_com

    # Apply parallel axis theorem
    I_motor_com = parallel_axis_theorem(I_motor_cm, motor_params.mass, motor_displacement)
    I_link_com = parallel_axis_theorem(I_link_cm, link_params.mass, link_displacement)

    # Composite inertia about composite center of mass
    I_composite = I_motor_com + I_link_com

    return I_composite, total_mass, composite_com


class URDFGenerator:
    """Generate URDF files for parametric robotic arms."""

    def __init__(self, robot_name: str = "positronic_roboarm"):
        """
        Initialize the URDF generator.

        Args:
            robot_name: (str) Name of the robot in the URDF
        """
        self.robot_name = robot_name
        self.root = ET.Element("robot", name=robot_name)

    def _add_link(
            self,
            link_name: str,
            motor_params: MotorParameters,
            link_params: LinkParameters | None = None,
    ) -> None:
        """
        Add a link element to the URDF.

        Args:
            link_name: Name of the link
            motor_params: Motor specifications
            link_params: Link specifications (None for end effector)
        """
        link_elem = ET.SubElement(self.root, "link", name=link_name)

        if link_params is not None:
            # Calculate composite inertia
            link_offset = motor_params.height / 2 + link_params.length / 2
            inertia_matrix, total_mass, com = calculate_composite_inertia(motor_params, link_params, link_offset)

            # Add inertial properties
            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "origin", xyz=f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{total_mass:.2f}")
            ET.SubElement(
                inertial,
                "inertia",
                ixx=f"{inertia_matrix[0, 0]:.6f}",
                ixy=f"{inertia_matrix[0, 1]:.6f}",
                ixz=f"{inertia_matrix[0, 2]:.6f}",
                iyy=f"{inertia_matrix[1, 1]:.6f}",
                iyz=f"{inertia_matrix[1, 2]:.6f}",
                izz=f"{inertia_matrix[2, 2]:.6f}"
            )
        else:
            # End effector - just motor inertia
            inertia_matrix = cylindrical_inertia(motor_params.mass, motor_params.radius, motor_params.height)

            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{motor_params.mass:.2f}")
            ET.SubElement(
                inertial,
                "inertia",
                ixx=f"{inertia_matrix[0, 0]:.6f}",
                ixy=f"{inertia_matrix[0, 1]:.6f}",
                ixz=f"{inertia_matrix[0, 2]:.6f}",
                iyy=f"{inertia_matrix[1, 1]:.6f}",
                iyz=f"{inertia_matrix[1, 2]:.6f}",
                izz=f"{inertia_matrix[2, 2]:.6f}"
            )

    def _add_joint(self, joint_config: JointConfiguration, parent_link: str, child_link: str) -> None:
        """
        Add a joint element to the URDF.

        Args:
            joint_config: Joint configuration parameters
            parent_link: Name of parent link
            child_link: Name of child link
        """
        joint_elem = ET.SubElement(self.root, "joint", name=joint_config.name, type=joint_config.joint_type)

        # Origin
        xyz_str = f"{joint_config.origin_xyz[0]:.6f} {joint_config.origin_xyz[1]:.6f} {joint_config.origin_xyz[2]:.6f}"
        rpy_str = f"{joint_config.origin_rpy[0]:.6f} {joint_config.origin_rpy[1]:.6f} {joint_config.origin_rpy[2]:.6f}"
        ET.SubElement(joint_elem, "origin", xyz=xyz_str, rpy=rpy_str)

        # Parent and child
        ET.SubElement(joint_elem, "parent", link=parent_link)
        ET.SubElement(joint_elem, "child", link=child_link)

        # Axis
        axis_str = f"{joint_config.axis[0]} {joint_config.axis[1]} {joint_config.axis[2]}"
        ET.SubElement(joint_elem, "axis", xyz=axis_str)

        # Limits
        ET.SubElement(
            joint_elem,
            "limit",
            effort=f"{joint_config.effort_limit}",
            velocity=f"{joint_config.velocity_limit}",
        )

    def generate_serial_arm(
            self,
            motor_params: Sequence[MotorParameters],
            link_params: Sequence[LinkParameters],
            joint_configs: Sequence[JointConfiguration],
    ) -> str:
        """
        Generate a serial robotic arm URDF.

        Args:
            motor_params: List of motor parameters for each joint
            link_params: List of link parameters for each joint
            joint_configs: List of joint configurations

        Returns:
            str: Generated URDF as XML string

        Raises:
            ValueError: If parameter lists have inconsistent lengths
        """
        num_joints = len(joint_configs)

        if len(motor_params) != num_joints:
            raise ValueError(f"Motor parameters length ({len(motor_params)}) must match joint count ({num_joints})")

        if len(link_params) != num_joints - 1:
            raise ValueError(
                f"Link parameters length ({len(link_params)}) must be one less than joint count ({num_joints})"
            )

        # Add base link
        ET.SubElement(self.root, "link", name="base")

        for i in range(num_joints):
            link_name = f"link{i+1}"

            self._add_link(link_name, motor_params[i], link_params[i] if i < len(link_params) else None)

            parent_link = "base" if i == 0 else f"link{i}"

            self._add_joint(joint_configs[i], parent_link, link_name)

        return self._format_xml()

    def _format_xml(self) -> str:
        """Format the XML with proper indentation."""
        rough_string = ET.tostring(self.root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


def create_arm(
        motors: Sequence[MotorParameters],
        link_lengths: Sequence[float] = (0.05, 0.05, 0.2, 0.05, 0.2, 0.05),
        joint_rotations: Sequence[float] = (np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2),
        link_density: float = 0.2,
        payload_mass: float = 2.0,
) -> str:
    """
    Create a robotic arm configuration.

    Args:
        link_lengths: (Sequence[float]) Lengths of the links
        motor_masses: (Sequence[float]) Masses of the motors
        motor_limits: (Sequence[float]) Limits of the motors
        joint_rotations: (Sequence[float]) Rotations of the joints in the frame of the previous links
        link_density: (float) Density of the links in terms of kg/m of length, since radius is fixed
        payload_mass: (float) Mass of the payload to be added to the last motor

    Returns:
        str: Generated URDF as XML string
    """

    assert len(link_lengths) + 1 == len(motors) == len(joint_rotations) + 1

    # simulate payload by adding it's mass to the last motor
    motors[-1].mass += payload_mass

    link_params = []
    joint_configs = [JointConfiguration(
        name='joint_1',
        origin_xyz=(0, 0, 0),
        origin_rpy=(0, 0, 0),
        effort_limit=motors[0].effort_limit,
        velocity_limit=motors[0].velocity_limit,
    )]

    for i in range(len(link_lengths)):
        link_params.append(LinkParameters(length=link_lengths[i], radius=0.025, mass=link_density * link_lengths[i]))
        joint_configs.append(
            JointConfiguration(
                name=f"joint_{len(joint_configs) + 1}",
                origin_xyz=(0, 0, link_lengths[i] + motors[i].height / 2 + motors[i].radius),
                origin_rpy=(joint_rotations[i], 0, 0),
                effort_limit=motors[i + 1].effort_limit,
                velocity_limit=motors[i + 1].velocity_limit,
            )
        )
    generator = URDFGenerator()
    return generator.generate_serial_arm(motors, link_params, joint_configs)
