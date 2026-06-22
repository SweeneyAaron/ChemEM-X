#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom ChimeraX mouse modes for ChemEM-X.

PlaceSolventMouseMode lets the user place the currently selected species (water,
an ion, or a polyatomic anion) exactly at the point under the cursor on each
click, with no density snapping. The species is read from chemem._solvent_species
(set via the SetSolventSpecies command). It is bound to a mouse button on demand
by the EnableSolventPlacement command (see binding_site/commands.py) rather than
registered at bundle init.
"""
from chimerax.mouse_modes import MouseMode
from chimerax.tug.tugatoms import Puller2D
from chimerax.ChemEM.dock.tools import add_molecule_to_model


class DragCoordinatesMode(MouseMode):
    """Right-button drag mode used during a live simulation: pick an atom and
    pull it toward the cursor. The active SimulationJob reads `atom_idx`
    (OpenMM index of the dragged atom) and `end_coord` (target position) each
    loop and applies a transient tug force. Bound/unbound by EnableTugMode /
    DisableTugMode. `Puller2D` is ChimeraX's stock tug helper."""

    name = 'drag_coordinates'

    def __init__(self, session, atom_to_idx):
        super().__init__(session)
        self.start_coord = None
        self.end_coord = None
        self.start_atom = None
        self.atom_to_idx = atom_to_idx
        self.atom_idx = None

    def mouse_down(self, event):
        # Capture the picked atom when the mouse button is pressed
        x, y = event.position()
        pick = self._get_scene_coordinates(x, y)
        if hasattr(pick, 'atom'):
            self.start_atom = pick.atom

    def mouse_drag(self, event):
        # add move velocity for tug strength
        pass

    def mouse_up(self, event):
        # Capture the target coordinate when the mouse button is released
        if self.start_atom is not None:
            x, y = event.position()
            tug_object = Puller2D(x, y)
            self.start_coord, self.end_coord = tug_object.pull_to_point(self.start_atom)
            self.atom_idx = self.atom_to_idx[self.start_atom]
            self.start_atom = None

    def _get_scene_coordinates(self, x, y):
        # Converts window coordinates to a picked scene object
        return self.session.main_view.picked_object(x, y)


class PlaceSolventMouseMode(MouseMode):
    name = 'place solvent'

    def __init__(self, chemem):
        super().__init__(chemem.session)
        self.chemem = chemem

    def mouse_down(self, event):
        super().mouse_down(event)

        model = self.chemem.parameters.get_parameter('current_model')
        if model is None:
            self.session.logger.status('Solvent placement: set a current model first.')
            return

        x, y = event.position()
        xyz1, xyz2 = self.view.clip_plane_points(x, y)
        if xyz1 is None or xyz2 is None:
            return

        # Place exactly on the first visible surface/atom/map-isosurface under the
        # cursor (the "mark surface" rule), with no density snapping.
        picked = self.view.picked_object_on_segment(xyz1, xyz2, max_transparent_layers=0)
        if picked is None:
            self.session.logger.status('Solvent placement: click on the model or map surface.')
            return

        # Convert the scene-coordinate intersection into the current model's
        # coordinate frame (correct even if the map/model have different
        # position transforms).
        model_xyz = model.scene_position.inverse() * picked.position

        species = getattr(self.chemem, '_solvent_species', 'HOH')
        try:
            res = add_molecule_to_model(model, species, model_xyz)
        except Exception as e:
            self.session.logger.status(f'Could not place {species}: {e}')
            return
        if res is None:
            self.session.logger.status(f'Could not place {species} (unknown or build failed).')
            return

        self.session.logger.status(f'Placed {species}.')
        self.chemem.run_js_code(f'setSolventStatus("Placed {species}. Click to place more.");')
