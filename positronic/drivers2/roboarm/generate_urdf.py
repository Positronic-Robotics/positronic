#!/usr/bin/env python3
"""
URDF Generator for Parametric Robotic Arms.

This script generates URDF files for robotic arms based on configurable parameters
including link dimensions, motor specifications, and joint configurations.
"""

import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add utilities to path for inertia calculations
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utilities'))
from inertia_calculation import cylindrical_inertia, parallel_axis_theorem, calculate_center_of_mass


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


@dataclass
class JointConfiguration:
    """Configuration for a single joint."""
    name: str
    joint_type: str = "continuous"
    axis: Tuple[float, float, float] = (0, 0, 1)
    origin_xyz: Tuple[float, float, float] = (0, 0, 0)
    origin_rpy: Tuple[float, float, float] = (0, 0, 0)
    effort_limit: float = 100.0
    velocity_limit: float = 1.0


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


    def _calculate_composite_inertia(self,
                                   motor_params: MotorParameters,
                                   link_params: LinkParameters,
                                   link_offset: float) -> Tuple[np.ndarray, float, np.ndarray]:
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

    def _add_link(self,
                  link_name: str,
                  motor_params: MotorParameters,
                  link_params: Optional[LinkParameters] = None) -> None:
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
            inertia_matrix, total_mass, com = self._calculate_composite_inertia(
                motor_params, link_params, link_offset
            )

            # Add inertial properties
            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "origin",
                         xyz=f"{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}",
                         rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{total_mass:.2f}")
            ET.SubElement(inertial, "inertia",
                         ixx=f"{inertia_matrix[0,0]:.6f}",
                         ixy=f"{inertia_matrix[0,1]:.6f}",
                         ixz=f"{inertia_matrix[0,2]:.6f}",
                         iyy=f"{inertia_matrix[1,1]:.6f}",
                         iyz=f"{inertia_matrix[1,2]:.6f}",
                         izz=f"{inertia_matrix[2,2]:.6f}")
        else:
            # End effector - just motor inertia
            inertia_matrix = cylindrical_inertia(motor_params.mass, motor_params.radius, motor_params.height)

            inertial = ET.SubElement(link_elem, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{motor_params.mass:.2f}")
            ET.SubElement(inertial, "inertia",
                         ixx=f"{inertia_matrix[0,0]:.6f}",
                         ixy=f"{inertia_matrix[0,1]:.6f}",
                         ixz=f"{inertia_matrix[0,2]:.6f}",
                         iyy=f"{inertia_matrix[1,1]:.6f}",
                         iyz=f"{inertia_matrix[1,2]:.6f}",
                         izz=f"{inertia_matrix[2,2]:.6f}")


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
        ET.SubElement(joint_elem, "limit",
                     effort=f"{joint_config.effort_limit}",
                     velocity=f"{joint_config.velocity_limit}")

    def generate_serial_arm(self,
                           motor_params: List[MotorParameters],
                           link_params: List[LinkParameters],
                           joint_configs: List[JointConfiguration]) -> str:
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
            raise ValueError(f"Link parameters length ({len(link_params)}) must be one less than joint count ({num_joints})")

        # Add base link
        base_link = ET.SubElement(self.root, "link", name="base")

        # Add joints and links
        for i in range(num_joints):
            link_name = f"link{i+1}"

            # Determine if this is the end effector (no link after it)
            if i < len(link_params):
                self._add_link(link_name, motor_params[i], link_params[i])
            else:
                self._add_link(link_name, motor_params[i], None)

            # Determine parent link
            parent_link = "base" if i == 0 else f"link{i}"

            # Add joint
            self._add_joint(joint_configs[i], parent_link, link_name)

        return self._format_xml()

    def _format_xml(self) -> str:
        """Format the XML with proper indentation."""
        rough_string = ET.tostring(self.root, 'unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")


def create_default_6dof_arm() -> str:
    """
    Create a default 6-DOF robotic arm configuration.

    Returns:
        str: Generated URDF as XML string
    """
    # Motor parameters (all identical for simplicity)
    motor_params = [
        MotorParameters(radius=0.05, height=0.05, mass=1.01),
        MotorParameters(radius=0.05, height=0.05, mass=1.01),
        MotorParameters(radius=0.05, height=0.05, mass=1.01),
        MotorParameters(radius=0.05, height=0.05, mass=1.01),
        MotorParameters(radius=0.05, height=0.05, mass=1.01),
        MotorParameters(radius=0.05, height=0.05, mass=1.01),
    ]

    # Link parameters (5 links for 6-DOF arm)
    link_params = [
        LinkParameters(length=0.1, radius=0.025, mass=0.01),   # Link 1
        LinkParameters(length=0.2, radius=0.025, mass=0.01),   # Link 2 (longer)
        LinkParameters(length=0.1, radius=0.025, mass=0.01),   # Link 3
        LinkParameters(length=0.2, radius=0.025, mass=0.01),   # Link 4 (longer)
        LinkParameters(length=0.1, radius=0.025, mass=0.01),   # Link 5
    ]

    # Joint configurations with alternating orientations
    joint_configs = [
        JointConfiguration(name="joint_1", origin_xyz=(0, 0, 0), origin_rpy=(0, 0, 0)),
        JointConfiguration(name="joint_2", origin_xyz=(0, 0, 0.1), origin_rpy=(1.5708, 0, 0)),
        JointConfiguration(name="joint_3", origin_xyz=(0, 0, 0.1), origin_rpy=(-1.5708, 0, 0)),
        JointConfiguration(name="joint_4", origin_xyz=(0, 0, 0.25), origin_rpy=(1.5708, 0, 0)),
        JointConfiguration(name="joint_5", origin_xyz=(0, 0, 0.1), origin_rpy=(-1.5708, 0, 0)),
        JointConfiguration(name="joint_6", origin_xyz=(0, 0, 0.25), origin_rpy=(1.5708, 0, 0)),
    ]

    generator = URDFGenerator()
    return generator.generate_serial_arm(motor_params, link_params, joint_configs)
