# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>


from chimerax.mouse_modes import MouseMode
from chimerax.markers import MarkerSet

class ChemEMMarkerMouseMode(MouseMode):
    def __init__(self, session, original_mode, chemem, parameter):
        super().__init__(session)
        self._marker_placement_active = True
        self.name = 'chemem_marker'
        self.original_mode = original_mode
        self.marker_wrapper = ChemEMMarker(self.session)
        self.chemem = chemem
        self.parameter = parameter
        
    def enable(self):
        self._marker_placement_active = True
        self.session.logger.info("Marker placement mode activated. Right-click to place a marker.")

    def disable(self):
        self._marker_placement_active = False
        self.session.logger.info("Marker placement mode deactivated.")

    def mouse_down(self, event):
        if self._marker_placement_active:
            x, y = event.position()
            self.place_marker(x, y)
            self._marker_placement_active = False  # Deactivate after placing the marker
            self.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode=self.original_mode)
            
    
    def place_marker(self, x, y):
        # Convert screen coordinates to scene coordinates
        view = self.session.main_view
        xyz1, xyz2 = view.clip_plane_points(x, y)
        xyz = 0.5 * (xyz1 + xyz2)
        self.session.models.add([self.marker_wrapper])
        self.marker_wrapper.place_marker(xyz)
        self.parameter.get_from_marker(self.marker_wrapper,self.chemem.current_binding_site_id, self.chemem.session)

class ChemEMMarker(MarkerSet):
    def __init__(self, session):
        super().__init__(session)

    def place_marker(self, position):
        print(position)
        marker = self.create_marker(position, (136, 179, 198, 255), 1.0)  # Red color, size 1.0
        print(type(marker))
        #self.add_marker(marker)
        

def activate_marker_placement(chemem, parameter):
    original_mode = chemem.session.ui.mouse_modes.mode(button="right")
    mode = ChemEMMarkerMouseMode(chemem.session, original_mode, chemem, parameter)
    chemem.session.ui.mouse_modes.bind_mouse_mode(mouse_button="right", mode=mode)    

