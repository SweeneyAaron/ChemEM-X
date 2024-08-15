from chimerax.mouse_modes import MouseMode 
from chimerax.tug.tugatoms import Puller2D

class DragCoordinatesMode(MouseMode):
    name = 'drag_coordinates'
    # Path to an icon file for the mouse mode

    def __init__(self, session, atom_to_idx):
        super().__init__(session)
        self.start_coord = None
        self.end_coord = None
        self.start_atom = None
        self.atom_to_idx  = atom_to_idx 
        self.atom_idx = None
        
    def mouse_down(self, event):
        # Capture the starting coordinates when the mouse button is pressed
        x, y = event.position()
        pick = self._get_scene_coordinates(x, y)
        if hasattr(pick, 'atom'):
            self.start_atom = pick.atom
            self.session.logger.info(f'Atom: {pick.atom.string(), {pick.atom.coord}}')

    def mouse_drag(self, event):
        #add move velocity for tug strength
        pass
   
    def mouse_up(self, event):
        # Capture the end coordinates when the mouse button is released
        
        if self.start_atom is not None:
            
            x, y = event.position()
            tug_object = Puller2D(x,y)
            self.start_coord, self.end_coord = tug_object.pull_to_point(self.start_atom)
            self.atom_idx = self.atom_to_idx[self.start_atom]
            self.start_atom = None
        

    def _get_scene_coordinates(self, x, y):
        # Converts window coordinates to scene coordinates
        view = self.session.main_view
        pick = view.picked_object(x, y)
        
        return pick

PLACE_LIGAND = 'place_ligand'

class PickPoint(MouseMode):
    name = 'pick_ligand_position'
    def __init__(self, session, atom):
        self.session = session 
        self.atom = atom
        self.end_coord = None
        
    def mouse_down(self, event):
        x, y = event.position()
        puller = Puller2D(x,y)
        _ , self.end_coord = puller.pull_to_point(self.atom)
        self.session.triggers.activate_trigger(PLACE_LIGAND, self)
        
        