#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:18:33 2025

@author: aaron.sweeney
"""
import json
import numpy as np 
from chimerax.markers import MarkerSet
from chimerax.ChemEM.core.commands import Command
from chimerax.ChemEM.binding_site.parameters import BindingSiteParameter
from chimerax.ChemEM.binding_site.tools import (BindingSiteChemEM,
                                                RenderBindingSite,
                                                convert_chimerax_atom_spec_to_chemem_atom_spec,
                                                flatten)
from chimerax.atomic import all_atoms
from chimerax.ChemEM.dock.tools import ChemEMSetUp
from chimerax.ChemEM.core.tools import ChemEMJob, IONFIXER_JOB , get_output_from_conf, launch_chemem_job

#TODO! change Rendering!!

def render_site_if_possible(chemem, site):
    """Render a binding site only when it is a rich, renderable ChemEM2BindingSite.
    Lightweight BindingSiteParameter sites (manual entry, or the marker fallback)
    have no add_to_session method and are skipped rather than crashing."""
    if not hasattr(site, "add_to_session"):
        return
    current_model = chemem.parameters.get_parameter('current_model')
    if current_model is None:
        return
    if chemem.rendered_site is not None:
        chemem.rendered_site.reset()
    chemem.rendered_site = RenderBindingSite(chemem.session, site,
                                             current_model,
                                             chemem.parameters.get_parameter('current_map'))


class RenderBindingSiteFromClick(Command):
    
    @classmethod 
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'populateFieldsFromBackendString("{site_value}");'
        return js_code
    
    @classmethod
    def run(cls, chemem, query):
        site = chemem.parameters.get_list_parameters('binding_sites', query)[0]
        render_site_if_possible(chemem, site)
        chemem.run_js_code(cls.js_code(site))

class TransferSiteToConf(Command):
    """The + Add icon: promote a candidate binding site into the conf (the list
    docking reads) and show it in the Setup 'Added Binding Sites' list."""

    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        return f'addToConfList("{site_value}", "{site.name}");'

    @classmethod
    def run(cls, chemem, query):
        candidates = chemem.parameters.get_list_parameters('binding_sites', query)
        if not candidates:
            return
        site = candidates[0]
        # Don't add the same site to the conf twice.
        if chemem.parameters.get_list_parameters('binding_sites_conf', site.name):
            chemem.run_js_code('showToast("Binding site already added.", "info");')
            return
        chemem.parameters.add_list_parameter('binding_sites_conf', site)
        chemem.run_js_code(cls.js_code(site))



class AutoSiteFinder(Command):

    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'''
        addBindingSiteEntry("{site_value}", "{site.name}");
        populateFieldsFromBackendString("{site_value}");
        '''
        return js_code
    
    @classmethod 
    def run(cls, chemem, query):
        
        chemem.autosites = BindingSiteChemEM.from_data(chemem.session,
                                        chemem.parameters.parameters['current_model'],
                                        chemem.parameters.get_parameter('current_map'))
        
        for site in chemem.autosites:
            chemem.current_binding_site_id = site.name
            render_site_if_possible(chemem, site)

            js_code = cls.js_code(site)
            chemem.run_js_code(js_code)
            
            #don't think i need this?
            chemem.current_renderd_site_id = site.name
            
            chemem.parameters.add_list_parameter('binding_sites', site)

class BindingSiteFromMarker(Command):
    
    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        js_code = f'''
        addBindingSiteEntry("{site_value}", "{site.name}");
        populateFieldsFromBackendString("{site_value}");
        '''
        return js_code
    
    @classmethod 
    def js_code_error(cls):
        message = "No Markers found.\\nPlace marker first to define binding site."
        return f'alert("{message}");'
    
    @classmethod 
    def run(cls, chemem, query):
        markers = []
        for model in chemem.session.models:
            if isinstance(model, MarkerSet):
                markers.append(model)
        
        if len(markers) == 0:
            chemem.run_js_code(cls.js_code_error())
        else:
            
            marker_centroids = [np.array(i.atoms[0].coord) for i in markers[0].residues]
            
            if chemem.autosites is None:
                chemem.autosites = BindingSiteChemEM.from_data(chemem.session,
                                                chemem.parameters.parameters['current_model'],
                                                chemem.parameters.get_parameter('current_map'))
            
            for centroid in marker_centroids:
                #TODO! pass back the ChemEM2Bindingsite object to make logic simpler
                site = chemem.autosites.site_from_centroid(centroid)
                
                if site is None:
                    #use manual binding site
                    site = BindingSiteParameter.get_from_centroid(centroid, chemem)
                    
                chemem.parameters.add_list_parameter('binding_sites', site)
                chemem.current_binding_site_id = site.name
                render_site_if_possible(chemem, site)

                js_code = cls.js_code(site)
                chemem.run_js_code(js_code)
                
                #don't think i need this?
                chemem.current_renderd_site_id = site.name
                #set current working site in backend!!!



class AddManualBindingSite(Command):
    """Manual define: register a binding site typed into the Centroid + Box form
    as a *candidate* (the 'binding_sites' list, like marker/autodetect). It only
    enters the docking conf once the user clicks + Add (TransferSiteToConf).

    The form sends a BindingSiteParameter ("(x, y, z) | (bx, by, bz)"). It is a
    lightweight site (no add_to_session), so it is listed + dockable but not drawn
    as a 3-D mesh (render_site_if_possible skips it)."""

    @classmethod
    def js_code(cls, site):
        site_value = f"Centroid: ({round(site.centroid[0], 3)}, {round(site.centroid[1], 3)}, {round(site.centroid[2], 3)}) | Box Size: ({int(site.box_size[0])}, {int(site.box_size[1])}, {int(site.box_size[2])});"
        return f'addBindingSiteEntry("{site_value}", "{site.name}");'

    @classmethod
    def run(cls, chemem, query):
        if chemem.parameters.get_parameter('current_model') is None:
            chemem.run_js_code('showToast("Load a model first.", "warning");')
            return

        site = query  # BindingSiteParameter parsed from the form (centroid + box)
        chemem.parameters.add_list_parameter('binding_sites', site)
        chemem.current_binding_site_id = site.name
        render_site_if_possible(chemem, site)

        chemem.run_js_code(cls.js_code(site))


class SetSolventSpecies(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem._solvent_species = str(query)


class EnableSolventPlacement(Command):
    @classmethod
    def run(cls, chemem, query):
        from chimerax.ChemEM.mouse_modes import PlaceSolventMouseMode

        mm = chemem.session.ui.mouse_modes
        # Remember the current right-button mode so it can be restored on disable
        # (don't clobber it if our placement mode is already active).
        prev = mm.mode(button='right')
        if not isinstance(prev, PlaceSolventMouseMode):
            chemem._prev_right_mouse_mode = prev

        mode = PlaceSolventMouseMode(chemem)
        mm.bind_mouse_mode(mouse_button='right', mode=mode)
        chemem._solvent_mouse_mode = mode
        chemem.run_js_code(
            'setSolventStatus("Solvent placement ON \\u2014 right-click the model/map surface.");')


class DisableSolventPlacement(Command):
    @classmethod
    def run(cls, chemem, query):
        mm = chemem.session.ui.mouse_modes
        mm.bind_mouse_mode(mouse_button='right',
                           mode=getattr(chemem, '_prev_right_mouse_mode', None))
        chemem._solvent_mouse_mode = None
        chemem.run_js_code('setSolventStatus("Solvent placement off.");')


class RemoveBindingSite(Command):
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites', query)
        #TODO! check the effects of this elsewhere
        if chemem.rendered_site is not None:
            if query == chemem.rendered_site.binding_site.name:
                chemem.rendered_site.reset()
                chemem.rendered_site = None


class RemoveBindingSiteFromConf(Command):
    """Remove a binding site from the docking conf (Setup 'Added Binding Sites').
    The candidate stays in the candidate list; it just won't be docked."""
    @classmethod
    def run(cls, chemem, query):
        chemem.parameters.remove_list_parameter('binding_sites_conf', query)


class SetEditBindingSiteValue(Command):
    @classmethod 
    def run(cls, chemem, query):
        chemem.render_site_from_key(query)

class AssignSelectedAtomToIonSite(Command):
    @classmethod
    def js_code(cls, atom_spec):
        return f"addAtomSpecToIonFixer({json.dumps(atom_spec)});"

    @staticmethod
    def _tracked_ligands_by_model_id(chemem):
        tracked = {}
        for ligand_id, record in (getattr(chemem, "_tracked_ligands", {}) or {}).items():
            model = record.get("model")
            model_id = record.get("model_id")
            if model is not None and getattr(model, "id", None) is not None:
                model_id = tuple(model.id)
            if model_id is not None:
                tracked[tuple(model_id)] = (ligand_id, record)
        return tracked

    @staticmethod
    def _protein_atom_spec(atom):
        chain_id = atom.residue.chain.chain_id
        res_name = atom.residue.name
        res_num = atom.residue.number
        atom_name = atom.name
        
        return f"{chain_id}:{res_name}-{res_num}@{atom_name}"

    @staticmethod
    def _model_id_to_spec(model_id):
        if model_id is None:
            return None

        if isinstance(model_id, (tuple, list)):
            if len(model_id) == 0:
                return None
            return ".".join(str(i) for i in model_id)

        model_id_text = str(model_id).strip()
        if not model_id_text:
            return None
        return model_id_text.lstrip("#")

    @staticmethod
    def _ligand_atom_spec(atom, ligand_record):
        matcher = ligand_record.get("atom_matcher")
        if matcher is None:
            return None

        # Keep matcher validation so only tracked ligand atoms are accepted.
        _rd_idx = matcher.get_by_atom(atom)
        if _rd_idx is None:
            return None

        atom_model = getattr(atom, "structure", None)
        model_id = getattr(atom_model, "id", None)
        if model_id is None:
            model_id = ligand_record.get("model_id")

        model_id_spec = AssignSelectedAtomToIonSite._model_id_to_spec(model_id)
        if model_id_spec is None:
            return None

        atom_name = atom.name
        return f"#{model_id_spec}:LIG@{atom_name}"

    @classmethod
    def run(cls, chemem, query):
        current_model = chemem.parameters.get_parameter('current_model')
        tracked_by_model_id = cls._tracked_ligands_by_model_id(chemem)

        selected = all_atoms(chemem.session)
        selected = selected[selected.selected]

        if len(selected) == 0:
            return

        valid_selected = []
        for atom in selected:
            atom_model = getattr(atom, "structure", None)
            atom_model_id = tuple(getattr(atom_model, "id", ())) if atom_model is not None else None

            if atom_model is current_model:
                valid_selected.append(("protein", atom, None))
                continue

            if atom_model_id in tracked_by_model_id:
                _ligand_id, ligand_record = tracked_by_model_id[atom_model_id]
                valid_selected.append(("ligand", atom, ligand_record))

        if len(valid_selected) != 1:
            return

        atom_kind, atom, ligand_record = valid_selected[0]
        if atom_kind == "protein":
            spec_string = cls._protein_atom_spec(atom)
        else:
            spec_string = cls._ligand_atom_spec(atom, ligand_record)
            if spec_string is None:
                return

        site_index = getattr(query, "value", None)
        if site_index is not None:
            payload = json.dumps({"site_index": int(site_index), "atom_spec": spec_string})
            chemem.run_js_code(cls.js_code(payload))
            return

        chemem.run_js_code(cls.js_code(spec_string))

class AddIonSiteParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        #if chemem.current_simualtion_id is not None:
        chemem.ion_site_parameters.add(query)

class AddIonSiteListParameter(Command):
    @classmethod 
    def run(cls, chemem, query):
        #if chemem.current_simualtion_id is not None:
        chemem.ion_site_parameters.add_list_parameter(query.name, query)

class RunIonFixer(Command):
    @classmethod 
    def run(cls, chemem, query):
        backend = chemem.parameters.get_value("chememBackendPath")
        
        if backend is not None:
            
            options = [(k,v.value) for k,v in chemem.ion_site_parameters.parameters.items() if type(v) != list and  v.value != 'protocol' ] 
            protocols = [k for k,v in chemem.ion_site_parameters.parameters.items() if type(v) != list and v.value == "protocol"]
            #should hold atom-sepc and exclude-spec
            list_options = [ v for  k,v in chemem.ion_site_parameters.parameters.items() if type(v) == list]
            flat_list = flatten(list_options)
            converted_list_options, ligand_order = convert_chimerax_atom_spec_to_chemem_atom_spec(flat_list)
            converted_list_options = [(param.name, param.value) for param in converted_list_options]
            options = options + converted_list_options


            chemem_setup = ChemEMSetUp.from_parameters_object(chemem.session, 
                                       chemem.parameters, 
                                       backend,
                                       protocols,
                                       options,
                                       ligand_order=ligand_order,
                                       tracked_ligands=getattr(chemem, "_tracked_ligands", None))
            
            command = chemem_setup.run_command

            launch_chemem_job(chemem, command, IONFIXER_JOB, "Add Ions")

            # Clear so the next run starts fresh (list params would otherwise
            # accumulate across runs).
            chemem.ion_site_parameters._clear()
