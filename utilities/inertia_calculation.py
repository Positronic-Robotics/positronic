import numpy as np

def rectangular_inertia(mass, length, width, height):
    """
    Calculate inertia matrix for rectangular prism about its center of mass.

    Args:
        mass: (float) Mass in kg
        length: (float) Length in meters (x-direction)
        width: (float) Width in meters (y-direction)
        height: (float) Height in meters (z-direction)

    Returns:
        numpy.ndarray: 3x3 inertia matrix
    """
    Ixx = (1/12) * mass * (width**2 + height**2)
    Iyy = (1/12) * mass * (length**2 + height**2)
    Izz = (1/12) * mass * (length**2 + width**2)

    return np.array([
        [Ixx, 0, 0],
        [0, Iyy, 0],
        [0, 0, Izz]
    ])

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

    # Identity matrix
    I = np.eye(3)

    # Parallel axis theorem: I_new = I_cm + m * (d²*I - d⊗d)
    I_translated = I_cm + mass * (d_squared * I - d_outer)

    return I_translated

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

def calculate_composite_inertia():
    """
    Calculate composite inertia for rectangular link + cylindrical motor.
    Includes center of mass calculations.
    """
    print("=== Composite Inertia Calculation ===\n")

    # Component properties (you'll need to specify actual dimensions)
    # Rectangular link
    link_mass = 0.01  # 10g
    link_length = 0.1  # Example: 10cm
    link_width = 0.02   # Example: 2cm
    link_height = 0.01  # Example: 1cm

    # Cylindrical motor
    motor_mass = 1.0   # 1kg
    motor_radius = 0.05  # Example: 5cm radius
    motor_height = 0.05  # Example: 5cm height

    print(f"Rectangular Link: {link_mass}kg, {link_length}×{link_width}×{link_height}m")
    print(f"Cylindrical Motor: {motor_mass}kg, r={motor_radius}m, h={motor_height}m\n")

    # Define component positions (centers of mass in world coordinates)
    # Assuming motor is at origin, link is offset
    motor_position = np.array([0.0, 0.0, 0.0])      # Motor at origin
    link_position = np.array([0.0, 0.0, 0.03])      # Link offset by 3cm in z

    print("Component center of mass positions:")
    print(f"Motor: {motor_position}")
    print(f"Link:  {link_position}")

    # Calculate composite center of mass
    masses = [motor_mass, link_mass]
    positions = [motor_position, link_position]
    composite_com, total_mass = calculate_center_of_mass(masses, positions)

    print(f"\nComposite center of mass: {composite_com}")
    print(f"Total mass: {total_mass}kg")

    # Calculate individual inertias about their centers of mass
    I_link_cm = rectangular_inertia(link_mass, link_length, link_width, link_height)
    I_motor_cm = cylindrical_inertia(motor_mass, motor_radius, motor_height)

    print(f"\nLink inertia (about its center of mass):")
    print(I_link_cm)
    print(f"\nMotor inertia (about its center of mass):")
    print(I_motor_cm)

    # Calculate displacements from composite center of mass to component centers
    link_displacement_from_com = link_position - composite_com
    motor_displacement_from_com = motor_position - composite_com

    print(f"\nDisplacements from composite center of mass:")
    print(f"Link:  {link_displacement_from_com}")
    print(f"Motor: {motor_displacement_from_com}")

    # Apply parallel axis theorem to translate to composite center of mass
    I_link_com = parallel_axis_theorem(I_link_cm, link_mass, link_displacement_from_com)
    I_motor_com = parallel_axis_theorem(I_motor_cm, motor_mass, motor_displacement_from_com)

    # Sum to get composite inertia about composite center of mass
    I_composite_com = I_link_com + I_motor_com

    print(f"\nComposite inertia matrix (about composite center of mass):")
    print(I_composite_com)

    # Also calculate about motor center (reference point) for comparison
    link_displacement_from_motor = link_position - motor_position
    motor_displacement_from_motor = motor_position - motor_position

    I_link_motor = parallel_axis_theorem(I_link_cm, link_mass, link_displacement_from_motor)
    I_motor_motor = parallel_axis_theorem(I_motor_cm, motor_mass, motor_displacement_from_motor)
    I_composite_motor = I_link_motor + I_motor_motor

    print(f"\nComposite inertia matrix (about motor center):")
    print(I_composite_motor)

    # URDF Format outputs
    print(f"\n" + "="*60)
    print("URDF FORMATS:")
    print("="*60)

    print(f"\n1. About composite center of mass (recommended):")
    print(f'<inertial>')
    print(f'  <origin xyz="{composite_com[0]:.6f} {composite_com[1]:.6f} {composite_com[2]:.6f}" rpy="0 0 0"/>')
    print(f'  <mass value="{total_mass:.3f}"/>')
    print(f'  <inertia ixx="{I_composite_com[0,0]:.6f}" ixy="{I_composite_com[0,1]:.6f}" ixz="{I_composite_com[0,2]:.6f}"')
    print(f'           iyy="{I_composite_com[1,1]:.6f}" iyz="{I_composite_com[1,2]:.6f}" izz="{I_composite_com[2,2]:.6f}"/>')
    print(f'</inertial>')

    print(f"\n2. About motor center:")
    print(f'<inertial>')
    print(f'  <origin xyz="{motor_position[0]:.6f} {motor_position[1]:.6f} {motor_position[2]:.6f}" rpy="0 0 0"/>')
    print(f'  <mass value="{total_mass:.3f}"/>')
    print(f'  <inertia ixx="{I_composite_motor[0,0]:.6f}" ixy="{I_composite_motor[0,1]:.6f}" ixz="{I_composite_motor[0,2]:.6f}"')
    print(f'           iyy="{I_composite_motor[1,1]:.6f}" iyz="{I_composite_motor[1,2]:.6f}" izz="{I_composite_motor[2,2]:.6f}"/>')
    print(f'</inertial>')

    return {
        'composite_com': composite_com,
        'total_mass': total_mass,
        'I_composite_com': I_composite_com,
        'I_composite_motor': I_composite_motor
    }

def visualize_center_of_mass_calculation():
    """
    Show step-by-step center of mass calculation with example.
    """
    print("\n" + "="*60)
    print("CENTER OF MASS CALCULATION EXAMPLE:")
    print("="*60)

    # Example values
    m1, m2 = 1.0, 0.01  # Motor: 1kg, Link: 10g
    pos1 = np.array([0, 0, 0])      # Motor at origin
    pos2 = np.array([0, 0, 0.03])   # Link 3cm away

    print(f"Component 1 (Motor): mass = {m1}kg, position = {pos1}")
    print(f"Component 2 (Link):  mass = {m2}kg, position = {pos2}")

    # Manual calculation
    total_mass = m1 + m2
    com_manual = (m1 * pos1 + m2 * pos2) / total_mass

    print(f"\nManual calculation:")
    print(f"Total mass = {m1} + {m2} = {total_mass}kg")
    print(f"COM = ({m1} × {pos1} + {m2} × {pos2}) / {total_mass}")
    print(f"COM = ({m1 * pos1} + {m2 * pos2}) / {total_mass}")
    print(f"COM = {m1 * pos1 + m2 * pos2} / {total_mass}")
    print(f"COM = {com_manual}")

    # Using function
    com_func, total_func = calculate_center_of_mass([m1, m2], [pos1, pos2])
    print(f"\nUsing function: COM = {com_func}, Total mass = {total_func}kg")

    print(f"\nNote: The center of mass is very close to the motor center")
    print(f"because the motor is much heavier than the link (1kg vs 10g)")

if __name__ == "__main__":
    results = calculate_composite_inertia()

    visualize_center_of_mass_calculation()

    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("1. Measure actual dimensions of your rectangular link and cylindrical motor")
    print("2. Update the dimension values in the script")
    print("3. Define the relative positions of components")
    print("4. Choose reference frame (composite COM is usually best)")
    print("5. Use the calculated inertia values in your URDF file")
    print("6. Set the <origin> in <inertial> to the chosen reference point")
    print("="*60)
