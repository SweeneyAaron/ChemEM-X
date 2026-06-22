#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 15:40:50 2025

@author: aaron.sweeney
"""

from . import view
import json
from json.decoder import JSONDecodeError
from urllib.parse import parse_qs
from Qt.QtCore import QObject, QEvent, QTimer
#chimeraX imports
from chimerax.ui import HtmlToolInstance
from chimerax.core.models import ADD_MODELS, REMOVE_MODELS, MODEL_POSITION_CHANGED
from chimerax.core.tasks import ADD_TASK, REMOVE_TASK
#chememX imports
from chimerax.ChemEM.core.commands  import Command, UpdateModels, UpdateMaps
from chimerax.ChemEM.core.parameters import Parameters
from chimerax.ChemEM.core.tools import get_chemem_paths,  get_platforms, JobHandler, CHEMEM_JOB, ALPHA_MASK_JOB, IONFIXER_JOB, EXPORT_SIMULATION, EXPORT_LIGAND_TORSIONS
from chimerax.ChemEM.core.tracked_ligands import normalise_model_id, tracked_ligand_display_name
from chimerax.atomic.structure import AtomicStructure
from chimerax.ChemEM.simulate.tools import get_simulation


#need to import * from ligand.commands to register Command subclasses
from chimerax.ChemEM.ligand.commands import * 
from chimerax.ChemEM.binding_site.commands import * 
from chimerax.ChemEM.ligand.parameters import * 
from chimerax.ChemEM.binding_site.parameters import * 
from chimerax.ChemEM.map_masks.commands import * 
from chimerax.ChemEM.map_masks.parameters import * 
from chimerax.ChemEM.dock.commands import *
from chimerax.ChemEM.refine.commands import *
from chimerax.ChemEM.simulate.commands import *
from chimerax.ChemEM.score.commands import *
from chimerax.ChemEM.score.tools import ScoreTable


class _LigandDropFilter(QObject):
    """Routes OS file drops on the web view into the ligand file span,
    reusing the Choose File flow so Add/AddLigandFile is unchanged."""
    LIGAND_EXTS = (".sdf", ".mol")

    def __init__(self, chemem):
        super().__init__()
        self._chemem = chemem

    def _ligand_url(self, mime):
        if not mime.hasUrls():
            return None
        for url in mime.urls():
            path = url.toLocalFile()
            if path and path.lower().endswith(self.LIGAND_EXTS):
                return path
        return None

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            if self._ligand_url(event.mimeData()) is not None:
                event.acceptProposedAction()
                self._chemem.run_js_code("setLigandDropActive(true);")
                return True
            return False
        if et == QEvent.Type.DragLeave:
            self._chemem.run_js_code("setLigandDropActive(false);")
            return False
        if et == QEvent.Type.Drop:
            path = self._ligand_url(event.mimeData())
            self._chemem.run_js_code("setLigandDropActive(false);")
            if path is not None:
                event.acceptProposedAction()
                safe = path.replace("\\", "\\\\").replace('"', '\\"')
                self._chemem.run_js_code(f'setFilePath("ligandFilePath", "{safe}");')
                return True
            return False
        return False


class CHEMEM(HtmlToolInstance):
    SESSION_ENDURING = False
    SESSION_SAVE = False  # No session saving for now
    CUSTOM_SCHEME = "chemem"
    display_name = "chemem" # HTML scheme for custom links
    PLACEMENT = None
    #help = "help:user/tools/chemem.html" #add this!!
    
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name, size_hint=(600, 1000))
        self.parameters = Parameters()
        self.avalible_chemem_exes =  get_chemem_paths()
        self.avalible_platforms = get_platforms()
        #handelers
        self.add_model_handeler = self.session.triggers.add_handler(ADD_MODELS, self.update_models)
        self.remove_model_handeler = self.session.triggers.add_handler(REMOVE_MODELS, self.update_models)
        self.model_position_handeler = self.session.triggers.add_handler(MODEL_POSITION_CHANGED, self.on_model_position_changed)
        self.job_remove_task =  self.session.triggers.add_handler(REMOVE_TASK, self.remove_task)
        self.job_handeler = JobHandler()
        #protocol varables
        self.current_protonation_states=None
        self.autosites = None
        self.rendered_site = None
        self.dock_parameters = Parameters()
        self.dock_results = None
        self.refine_parameters = Parameters()
        self.refine_results = None
        self._refine_selected_poses = []
        #Score tab: comparison table accumulating per-ligand scores
        self.score_results = ScoreTable()
        self.mask_parameters = Parameters()
        self.alpha_mask_results = None
        self.simulation_parameters = Parameters()
        self.added_ligands = {}
        #active simulation (SimulationContainer) and the compute platform used to
        #build it. platform defaults to the first available platform (CPU fallback)
        #so the build->run path works even if the user never picks one explicitly.
        self.current_simulation = None
        self.platform = self.avalible_platforms[0] if self.avalible_platforms else 'CPU'
        self.ion_site_parameters = Parameters()
        self._solvent_species = 'HOH'  #species placed by PlaceSolventMouseMode
        self._solvent_mouse_mode = None  #active PlaceSolventMouseMode, if any
        self._prev_right_mouse_mode = None  #right-button mode to restore on disable


        self._tracked_ligands = {} #track ligand models opened from sdf files
        self.view = view.ChemEMView(self) #!!!
        self.view.render()

        # Drag-and-drop ligand support (Qt-level so we recover the real path).
        self._ligand_drop_filter = _LigandDropFilter(self)
        self.html_view.setAcceptDrops(True)
        self.html_view.installEventFilter(self._ligand_drop_filter)
        # The Chromium render child (focusProxy) that otherwise eats drops is
        # created after layout, so (re)install once the page has loaded.
        self.html_view.page().loadFinished.connect(self._install_ligand_drop_filter)
        QTimer.singleShot(0, self._install_ligand_drop_filter)

        session.metadata = self

    def _install_ligand_drop_filter(self, *args):
        from Qt.QtWidgets import QWidget
        self.html_view.setAcceptDrops(True)
        self.html_view.installEventFilter(self._ligand_drop_filter)
        proxy = self.html_view.focusProxy()
        if proxy is not None:
            proxy.setAcceptDrops(True)
            proxy.installEventFilter(self._ligand_drop_filter)
        for child in self.html_view.findChildren(QWidget):
            child.setAcceptDrops(True)
            child.installEventFilter(self._ligand_drop_filter)

    def run_js_code(self, js_code):
        self.html_view.page().runJavaScript(js_code)

    def delete(self):
        # Restore the right-button mouse mode if solvent placement is still active,
        # so closing the tool doesn't leave the right button hijacked.
        try:
            if self._solvent_mouse_mode is not None:
                self.session.ui.mouse_modes.bind_mouse_mode(
                    mouse_button='right', mode=self._prev_right_mouse_mode)
                self._solvent_mouse_mode = None
        except Exception:
            pass
        super().delete()

    def clear_binding_site_tabs(self):
        self.parameters.clear_binding_site_tabs(self)

    def select_model_js(self, model):
        model_key = UpdateModels.get_model_id(model)
        return f'selectOptionByValue("models", "{model_key}");'

    @staticmethod
    def _normalise_ligand_id(ligand_id):
        return str(ligand_id).strip()

    def get_tracked_ligand_record(self, ligand_id):
        tracked = getattr(self, "_tracked_ligands", {}) or {}
        if not tracked:
            return None

        ligand_id_key = self._normalise_ligand_id(ligand_id)

        if ligand_id in tracked:
            return tracked[ligand_id]

        if ligand_id_key in tracked:
            return tracked[ligand_id_key]

        for tracked_id, record in tracked.items():
            if self._normalise_ligand_id(tracked_id) == ligand_id_key:
                return record

        return None

    def _extract_models_from_trigger_args(self, *args):
        changed_models = []

        def _append_if_model(value):
            if value is None:
                return
            if hasattr(value, "id"):
                changed_models.append(value)

        for arg in args[1:]:
            if arg is None:
                continue

            if isinstance(arg, (list, tuple, set)):
                for item in arg:
                    _append_if_model(item)
                continue

            if hasattr(arg, "models"):
                maybe_models = arg.models
                if callable(maybe_models):
                    try:
                        maybe_models = maybe_models()
                    except Exception:
                        maybe_models = None
                if isinstance(maybe_models, (list, tuple, set)):
                    for item in maybe_models:
                        _append_if_model(item)
                    continue

            _append_if_model(arg)

        return changed_models

    def tracked_ligands_payload(self):
        payload = []
        tracked = getattr(self, "_tracked_ligands", {}) or {}
        for ligand_id, record in tracked.items():
            model = record.get("model")
            model_id = record.get("model_id")

            if model is not None and getattr(model, "id", None) is not None:
                model_id = tuple(model.id)
                record["model_id"] = model_id

            model_id_text = normalise_model_id(model_id)
            model_name = getattr(model, "name", None) or "ligand"
            payload.append({
                "ligand_id": self._normalise_ligand_id(ligand_id),
                "display_name": tracked_ligand_display_name(record, fallback_name=model_name),
                "source_sdf_path": record.get("source_sdf_path"),
                "model_name": model_name,
                "model_id": model_id_text,
                "dirty": bool(record.get("dirty", False)),
                "last_saved_path": record.get("last_saved_path"),
            })

        return payload

    def push_tracked_ligands_to_ui(self):
        payload_json = json.dumps(self.tracked_ligands_payload())
        js_code = f'if (typeof renderTrackedLigands === "function") renderTrackedLigands({payload_json});'
        self.run_js_code(js_code)
        #keep the Build-tab "Available tracked ligands" picker in sync
        picker_js = (f'if (typeof renderSimulationLigandPicker === "function") '
                     f'renderSimulationLigandPicker({payload_json});')
        self.run_js_code(picker_js)

    def _prune_tracked_ligands(self, removed_models=None):
        tracked = getattr(self, "_tracked_ligands", None)
        if not tracked:
            return False

        removed_models = set(removed_models or [])
        changed = False

        for ligand_id, record in list(tracked.items()):
            model = record.get("model")

            if model is None:
                tracked.pop(ligand_id, None)
                changed = True
                continue

            if removed_models and model in removed_models:
                tracked.pop(ligand_id, None)
                changed = True
                continue

            model_id = getattr(model, "id", None)
            if model_id is None:
                tracked.pop(ligand_id, None)
                changed = True
                continue

            model_id = tuple(model_id)
            if not self.session.models.have_id(model_id):
                tracked.pop(ligand_id, None)
                changed = True
                continue

            if record.get("model_id") != model_id:
                record["model_id"] = model_id
                changed = True

        return changed

    def on_model_position_changed(self, *args):
        tracked = getattr(self, "_tracked_ligands", None)
        if not tracked:
            return

        changed_models = self._extract_models_from_trigger_args(*args)
        changed_model_ids = set()
        for model in changed_models:
            model_id = getattr(model, "id", None)
            if model_id is not None:
                changed_model_ids.add(tuple(model_id))

        did_change = False
        for record in tracked.values():
            model = record.get("model")
            model_id = None
            if model is not None and getattr(model, "id", None) is not None:
                model_id = tuple(model.id)

            # Fallback to mark dirty when trigger payload does not expose model list.
            if not changed_model_ids or model_id in changed_model_ids:
                if not record.get("dirty", False):
                    record["dirty"] = True
                    did_change = True

        if did_change:
            self.push_tracked_ligands_to_ui()
    
    def remove_task(self, *args):
        job = args[1]
        if job.job_type in (CHEMEM_JOB, ALPHA_MASK_JOB, IONFIXER_JOB, EXPORT_SIMULATION, EXPORT_LIGAND_TORSIONS):
            if job.process.returncode == 0:
                js_code = f'updateJobStatus( {job.id}, "Completed");'
            else:
                js_code = f'updateJobStatus( {job.id}, "Failed");'

            self.run_js_code(js_code)

        if job.job_type == EXPORT_LIGAND_TORSIONS:
            # Ligand-tab torsion export: do NOT build a full OpenMM simulation -
            # just register the per-ligand export dir and refresh that ligand's
            # flicker so the freshly written torsion_profiles.json loads.
            pending = getattr(self, "_pending_ligand_torsion_export", None) or {}
            self._pending_ligand_torsion_export = None
            ligand_id = pending.get("ligand_id")
            run_dir = pending.get("dir")
            if job.process.returncode != 0:
                self.run_js_code('alert("Ligand torsion export failed - see the ChemEM job output for details.");')
                return
            if ligand_id and run_dir:
                if not hasattr(self, "_ligand_torsion_export_dirs") or self._ligand_torsion_export_dirs is None:
                    self._ligand_torsion_export_dirs = {}
                self._ligand_torsion_export_dirs[ligand_id] = run_dir
                # LigandTorsionRows is imported via `from ...ligand.commands import *`.
                LigandTorsionRows.run(self, ligand_id)
            return

        if job.job_type == EXPORT_SIMULATION:
            # Only build the OpenMM simulation if the export job actually wrote
            # the prmtop/inpcrd files; otherwise surface a clear message instead
            # of crashing on the missing files.
            if job.process.returncode != 0:
                self.run_js_code('alert("Simulation export failed - see the ChemEM job output for details.");')
                return

            self.current_simulation = get_simulation(self.session,
                                                    self.simulation_parameters.get_parameter('build_params').get_parameter('output').value,
                                                    self.platform,
                                                    self.parameters.get_parameter('current_model'),
                                                    parameters = self.simulation_parameters,
                                                    current_map=self.parameters.get_parameter('current_map'),
                                                    map_bias = None,
                                                    )

            self.current_simulation.setup_simulation()
            for js_code in self.current_simulation.js_code:
                self.run_js_code(js_code)
    
    def handle_scheme(self, url):
        
        command = url.path()
        query = self.extract_query(url)
        
        #print(Command.__subclasses__())
        
        for command_class in Command.__subclasses__():
            
            command_class().execute(self, command, query)
    
        print('COM:',command)
        print('QUERY', query)
        
    
    
    def extract_query(self, url):
        query = parse_qs(url.query())
        query = query['siteValue'][0]
        #print('unfilterd_query', query)
        
        try:
            query = json.loads(query)
        except JSONDecodeError:
            pass 

        if type(query) == dict:
            query = self.parameters.get_from_query(query)
            
        
        return query
    
    # functions for updating the state upon action
    def update_models(self, *args):
        
        """
        Handles adding and removing models.
        
        Parameters
        ----------
        *args : list
            - args[0] : str
                The function executed (e.g., 'remove models').
            - args[1] : list
                The list of models involved in the operation.
        """
        #TODO! refactor-this 
        
        current_map = self.parameters.get_parameter('current_map')
        current_model = self.parameters.get_parameter('current_model')
        #current_simulation_map = self.current_simulation.get_parameter('current_map')
        #current_simulation_model = self.current_simulation.get_parameter('current_model')
        #if self.current_simulation.get_parameter('AddedSolution') is not None:
        #    current_simulation_model = None #Don't need this if it was never set
        
        
        if args[0] == 'remove models':
            tracked_changed = self._prune_tracked_ligands(removed_models=args[1])
            if tracked_changed:
                self.push_tracked_ligands_to_ui()
            
            if current_model in args[1]:
                models_left = [i for i in self.session.models if isinstance(i, AtomicStructure) and i not in args[1]]
                self.clear_binding_site_tabs()
               
                if models_left:
                    #if a model still open set this as the current model 
                    UpdateModels.execute(self, 'UpdateModels', '_')
                    self.run_js_code( self.select_model_js( models_left[0] ))
                    return
                else:
                    #remove current model and cleanupTODO!         
                    del self.parameters.parameters['current_model']
                    return
                    
            
            if current_map in args[1]:
                self.clear_binding_site_tabs()
                #set current_map to None
                UpdateMaps.execute(self, 'UpdateMaps', '_')
                return
            
        
        
        if current_map is not None:
            current_map_key = UpdateMaps.get_map_id(current_map)
            js_code_maps = f'selectOptionByValue("maps", "{current_map_key}");'
        
        if current_model is not None and getattr(current_model, "id", None) is not None:
            current_model_key = UpdateModels.get_model_id(current_model)
            js_code_model = f'selectOptionByValue("models", "{current_model_key}");'
        else:
            current_model = None
        
        #Remove this in favor of simpler ux??
        '''
        if current_simulation_map is not None:
            current_simulation_map_key = UpdateSimulationMaps.get_map_id(current_simulation_map)
            js_code_simulated_map =  f'selectOptionByValue("SimulationMaps", "{current_simulation_map_key}");'
        
        if current_simulation_model is not None:
            current_simulation_model_key = UpdateModels.get_model_id(current_simulation_model)
            js_code_simulated_model =  f'selectOptionByValue("simulationModels", "{current_simulation_model_key}");'
        '''
        
        UpdateModels.execute(self, 'UpdateModels', '_')
        UpdateMaps.execute(self, 'UpdateMaps', '_')
        #UpdateSimulationMaps.execute(self, 'UpdateSimulationMaps', '_')
        #UpdateSimulationModels.execute(self, 'UpdateSimulationModels', '_')
        
        if current_map is not None:
            self.run_js_code(js_code_maps)
        
        if current_model is not None:
            self.run_js_code(js_code_model)
        
        '''
        if current_simulation_map is not None:
            self.run_js_code(js_code_simulated_map)
        
        if current_simulation_model is not None:
            self.run_js_code(js_code_simulated_model)
        '''

        if self._prune_tracked_ligands():
            self.push_tracked_ligands_to_ui()
        
    
    
    
