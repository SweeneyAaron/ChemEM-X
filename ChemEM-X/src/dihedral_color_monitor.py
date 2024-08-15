from chimerax.atomic import AtomicStructure, Structure
from chimerax.core.triggerset import TriggerSet
from chimerax.geometry import dihedral
from chimerax.core.colors import Color
# Define a trigger set for dihedral updates
dihedral_triggers = TriggerSet()
dihedral_triggers.add_trigger("update_dihedral")

def monitor_dihedral(session, atom1, atom2, atom3, atom4, bond, color_map):
    """
    Monitor the dihedral angle formed by four atoms and change the color of the bond
    depending on the dihedral angle.
    
    Args:
        session: The ChimeraX session object.
        atom1, atom2, atom3, atom4: The four atoms defining the dihedral.
        bond: The bond to color based on the dihedral angle.
        color_map: A function or dictionary mapping dihedral angles to colors.
    """

    def update_dihedral_angle(trigger_name, data):
        # Calculate the current dihedral angle
        angle = dihedral(atom1.scene_coord, atom2.scene_coord, atom3.scene_coord, atom4.scene_coord)
        # Determine the color based on the dihedral angle
        if callable(color_map):
            color = color_map(angle)
        else:
            color = determine_color_from_map(angle, color_map)
        
        # Update the bond color
        print(color)
        print(dir(color))
        bond.color = color

    # Add the update function to the trigger set
    dihedral_triggers.add_handler("update_dihedral", update_dihedral_angle)

    # Activate the trigger
    dihedral_triggers.activate_trigger("update_dihedral", None)

def determine_color_from_map(angle, color_map):
    """
    Determine the color for the bond based on the dihedral angle using a color map.
    You can create a color map as a dictionary with angle ranges as keys and colors as values.
    
    Args:
        angle: The dihedral angle.
        color_map: Dictionary mapping angle ranges to colors.
    
    Returns:
        The color corresponding to the dihedral angle.
    """
    for angle_range, color in color_map.items():
        if angle_range[0] <= angle <= angle_range[1]:
            return color
    return None

def register_dihedral_monitor(session, atom1, atom2, atom3, atom4, bond, color_map):
    """
    Register the dihedral monitor in the session.
    """
    monitor_dihedral(session, atom1, atom2, atom3, atom4, bond, color_map)

# Example of usage:
# Define atoms (you would normally get these from your model)
# atom1, atom2, atom3, atom4 = ...

# Define bond (also obtained from your model)
# bond = ...

# Define a color map for the dihedral angle
# For example, angles between 0-60 degrees are red, 60-120 are green, etc.
color_map = {
    (0, 60): (255, 0, 0, 255),   # Red
    (60, 120): (0, 255, 0, 255),  # Green
    (120, 180): (0, 0, 255, 255),  # Blue
    (-180, -120): (0, 0, 255, 255),  # Blue
    (-120, -60): (0, 255, 0, 255),  # Green
    (-60, 0): (255, 0, 0, 255)   # Red
}

# Register the dihedral monitor
# register_dihedral_monitor(session, atom1, atom2, atom3, atom4, bond, color_map)




