#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 12:15:10 2025

@author: aaron.sweeney
"""

#helper functions
import numpy as np 
import subprocess
from scipy.stats import norm
from  scipy import ndimage 
import uuid
import os
import json
import shutil
from rdkit import Chem
from openmm import Platform
from chimerax.core.tasks import Task, TaskState
from chimerax.ChemEM.core.parameters import PathParameter
import datetime
from chimerax.core.commands import run as cx_run
from chimerax.ChemEM.dock.tools import AtomMatcher, mol_from_sdf
import numpy as np


def _build_atom_matcher_from_model_and_sdf(model, sdf_path, tol=1e-3):
    # Keep explicit H atoms for tracked-ligand rewrite output.
    mol = mol_from_sdf(sdf_path, remove_hs=False)
    if mol is None or mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    rd_to_cx = {}
    cx_to_rd = {}
    used_cx_atoms = set()

    cx_atoms = list(model.atoms)

    # Match each RD atom to closest same-element ChimeraX atom
    for rd_idx in range(mol.GetNumAtoms()):
        rd_atom = mol.GetAtomWithIdx(rd_idx)
        rd_elem = rd_atom.GetSymbol().upper()
        rd_xyz = np.asarray(conf.GetAtomPosition(rd_idx), dtype=float)

        best_atom = None
        best_dist = float("inf")

        for cx_atom in cx_atoms:
            if cx_atom in used_cx_atoms:
                continue
            if cx_atom.element.name.upper() != rd_elem:
                continue

            cx_xyz = np.asarray([float(cx_atom.coord[0]), float(cx_atom.coord[1]), float(cx_atom.coord[2])], dtype=float)
            d = float(np.linalg.norm(cx_xyz - rd_xyz))
            if d < best_dist:
                best_dist = d
                best_atom = cx_atom

        if best_atom is not None and best_dist <= tol:
            rd_to_cx[rd_idx] = best_atom
            cx_to_rd[best_atom] = rd_idx
            used_cx_atoms.add(best_atom)

    return AtomMatcher(rd_to_cx, cx_to_rd, rdmol=mol, file=sdf_path)


def _unique_tracked_id(chemem, sdf_path):
    """Generate a tracked-ligand id from the SDF basename, uniquified against
    the existing keys in chemem._tracked_ligands."""
    base = os.path.splitext(os.path.basename(str(sdf_path)))[0] or 'ligand'
    existing = getattr(chemem, '_tracked_ligands', {}) or {}
    if base not in existing:
        return base
    i = 2
    while f'{base}_{i}' in existing:
        i += 1
    return f'{base}_{i}'


def register_tracked_ligand(chemem, sdf_path, model=None, ligand_id=None):
    """Open (or reuse) the SDF as a standalone model, build the AtomMatcher,
    register a tracked-ligand record, refresh the UI, and return ligand_id
    (or None on failure).

    This is the single registration point used by both the Setup ligand-file
    loader and the docked-solution 'Add' action, so every ligand that enters a
    simulation is torsion-capable."""
    if model is None:
        opened = cx_run(chemem.session, f'open "{sdf_path}"')
        model = next((m for m in opened if hasattr(m, "atoms")), None)
    if model is None:
        return None

    matcher = _build_atom_matcher_from_model_and_sdf(model, sdf_path)
    if matcher is None:
        return None

    if ligand_id is None:
        ligand_id = _unique_tracked_id(chemem, sdf_path)
    ligand_id = str(ligand_id).strip()

    chemem._tracked_ligands[ligand_id] = {
        "source_sdf_path": sdf_path,
        "model": model,
        "model_id": tuple(model.id),
        "atom_matcher": matcher,
        "dirty": False,
        "last_saved_path": None,
    }
    if hasattr(chemem, "push_tracked_ligands_to_ui"):
        chemem.push_tracked_ligands_to_ui()
    return ligand_id


#ChemEM Job
CHEMEM_JOB = 'chemem'
ALPHA_MASK_JOB = 'chemem alpha mask'
SIMULATION_JOB = 'chemem-X simulation'
IONFIXER_JOB = 'chemem ion fixer'
EXPORT_SIMULATION = 'export simulation'
EXPORT_LIGAND_TORSIONS = 'export ligand torsions'

class ChemEMJob(Task):
    def __init__(self, session, command, job_type, chemem=None):

        super().__init__(session)
        self.command = command
        self.log_file = []
        self.job_type = job_type #should be able to load the relevent files from the command now
        self.success = None
        self._chemem = chemem  #plugin tool instance, for streaming live output to the HTML view

    def _emit_output(self, text, stream="out"):
        """Stream one output line to the HTML job list.

        ``run()`` executes on a ChimeraX worker thread, so the JS call must be
        marshalled to the UI thread (same pattern as ``thread_safe_log``).
        """
        if self._chemem is None:
            return
        payload = json.dumps({"id": self.id, "line": text, "stream": stream})
        self.session.ui.thread_safe(self._chemem.run_js_code, f"appendJobOutput({payload});")
        
    def terminate(self):
        self.session.tasks.remove(self)
        self.end_time = datetime.datetime.now()
        
        if self._terminate is not None:
            self._terminate.set()
            
        self.state = TaskState.TERMINATING
        
    def run(self, *args, **kw):
        try:
            self.process = subprocess.Popen(self.command, shell=True, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                for line in self.process.stdout:
                    self.log_file.append(f"Output: {line.strip()}\n")
                    self.thread_safe_log(f"Output: {line.strip()}")
                    self._emit_output(line.strip(), stream="out")

                err = self.process.stderr.read()
                if err:
                    self.log_file.append(f"Error: {err.strip()}\n")
                    self.thread_safe_log(f"Error: {err.strip()}")
                    self._emit_output(err.strip(), stream="err")

                    return
                
            finally:
                # Properly close stdout and stderr
                self.process.stdout.close()
                self.process.stderr.close()
                self.process.wait()
                
            if self.process.returncode != 0:
                self.log_file.append(f"Process exited with code {self.process.returncode}\n")
                self.thread_safe_error(f"Process exited with code {self.process.returncode}")
                self.success = False
                return 
            self.success = True
            return 
        
        except Exception as e:
            self.log_file.append(f"Exception while executing command: {e}\n")
            self.thread_safe_error(f"Exception while executing command: {e}")
            self.success = False
            return 
     
    def on_finish(self):
        """Callback method executed after task thread terminates.

        This callback is executed in the UI thread after the
        :py:meth:`run` method returns.  By default, it does nothing.

        """
        self.terminate()
    
    def cancel_job(self):
        """
        Cancels the running job if possible.
        """
        
        if self.process and self.process.poll() is None:
            self.process.terminate()  # Sends SIGTERM
            self.process.wait()
            self.terminate()



class JobHandler:
    def __init__(self):
        
        self.jobs = {}  # Dictionary to store jobs with a unique identifier as the key
    
    
    def add_job(self, job):
        """ Adds a job to the handler. """
        if job.id in self.jobs:
            raise ValueError("A job with this ID already exists.")
        self.jobs[job.id] = job

    def remove_job(self, job_id):
        """ Removes a job from the handler. """
        if job_id in self.jobs:
            self.stop_job(job_id)
            del self.jobs[job_id]


    def stop_job(self, job_id):
        """ Stops a specific job by ID. """
        if job_id in self.jobs:
            self.jobs[job_id].cancel_job()
           

    def get_job_status(self, job_id):
        """ Returns the status of a specific job. """
        if job_id in self.jobs:
            return self.jobs[job_id].status
        return None
    
    def simulation_job_running(self):
        for job in self.jobs.values():
            if job.job_type == SIMULATION_JOB:
                if job.running:
                    return True
        return False


def launch_chemem_job(chemem, command, job_type, summary, load_type=None):
    """Create, surface, start and register a subprocess ChemEMJob.

    Centralises the create -> addJob -> start -> register boilerplate so every
    subprocess job (dock, alpha-mask, add-ions, export-simulation) shows up in
    the unified Jobs tab with a friendly summary and live output streaming.

    ``summary`` is the human-readable label shown as "Job N: <summary>".
    ``load_type`` is the routing key the Load button uses (e.g. 'dock',
    'alpha-mask'); omit it for jobs with no loadable results.
    """
    job = ChemEMJob(chemem.session, command, job_type, chemem=chemem)
    job_data = {
        "id": job.id,
        "status": "running",
        "type": load_type or "",
        "summary": summary,
    }
    chemem.run_js_code(f"addJob({json.dumps(job_data)});")
    job.start()
    chemem.job_handeler.add_job(job)
    return job





def validate_ligand_file(ligand_file):
    valid_file = False
    try:
        mol = mol_from_sdf(ligand_file)
        if mol is not None:
            valid_file = True
    except Exception as e:
        print(f'[ChemEMError] {ligand_file}')
        print(f'[ErrorHint] {e}')
    return valid_file
        

def mol_from_sdf(sdf_file, conf_num = 0, remove_hs = True):
    # Read with removeHs=False so callers can choose whether to keep explicit H atoms.
    suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
    mol = suppl[conf_num]
    if mol is None:
        return None
    if remove_hs:
        mol = Chem.RemoveHs(mol, sanitize=False)
    return mol

def get_platforms():
    import chimerax 
    openmm_plugins_dir = os.path.join(chimerax.app_lib_dir, 'plugins')
    Platform.loadPluginsFromDirectory(openmm_plugins_dir)
    avalible_platforms = []
    for i in range(Platform.getNumPlatforms()):
        plt = Platform.getPlatform(i).getName() 
        
        
        if plt not in avalible_platforms and plt != 'Reference':
            avalible_platforms.append(Platform.getPlatform(i).getName())
    return avalible_platforms


def get_output_from_conf(conf_path):
    #TODO add type helper to get the value types correctly
    if os.path.exists(conf_path):
        with open(conf_path, 'r') as f:
            conf_data = f.read().splitlines()
            data = {}
            for line in conf_data:
                line = line.split('=')
                data[line[0].replace(' ', '')] = line[1].replace(' ', '')
        return data 

def _exe_discovery_log(message):
    print(f"[ChemEMExeDiscovery] {message}")

def _normalise_path(path):
    return os.path.normpath(os.path.realpath(os.path.expanduser(path)))

def _paths_equivalent(path_a, path_b):
    if path_a == path_b:
        return True

    try:
        if os.path.exists(path_a) and os.path.exists(path_b):
            return os.path.samefile(path_a, path_b)
    except OSError:
        return False

    return False

def _append_unique_path(path_list, path, existing_only=False):
    if not path:
        return

    norm_path = _normalise_path(path)
    if existing_only and not os.path.exists(norm_path):
        return

    for existing_path in path_list:
        if _paths_equivalent(existing_path, norm_path):
            return

    path_list.append(norm_path)

def _existing_directories(paths):
    unique = []
    for path in paths:
        _append_unique_path(unique, path, existing_only=True)
    return [p for p in unique if os.path.isdir(p)]

def _known_conda_roots():
    home = os.path.expanduser("~")
    guessed_roots = [
        os.path.join(home, "anaconda3"),
        os.path.join(home, "miniconda3"),
        os.path.join(home, "miniforge3"),
        os.path.join(home, "mambaforge"),
        os.path.join(home, "opt", "anaconda3"),
        os.path.join(home, "opt", "miniconda3"),
        os.path.join(home, "opt", "miniforge3"),
        os.path.join(home, "opt", "mambaforge"),
        os.path.join(home, "Anaconda3"),
        os.path.join(home, "Miniconda3"),
        os.path.join(home, "Miniforge3"),
        os.path.join(home, "Mambaforge"),
        "C:\\ProgramData\\Anaconda3",
        "C:\\ProgramData\\Miniconda3",
        "C:\\ProgramData\\Miniforge3",
        "C:\\ProgramData\\Mambaforge",
        "C:\\Anaconda3",
        "C:\\Miniconda3",
        "C:\\Miniforge3",
        "C:\\Mambaforge",
    ]
    return _existing_directories(guessed_roots)

def _discover_conda_executables():
    executables = []
    set_conda_environment()

    for exe_name in ("conda", "mamba", "micromamba"):
        exe_path = shutil.which(exe_name)
        if exe_path:
            _append_unique_path(executables, exe_path, existing_only=True)

    for root in _known_conda_roots():
        _append_unique_path(executables, os.path.join(root, "bin", "conda"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "bin", "mamba"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "bin", "micromamba"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "Scripts", "conda.exe"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "Scripts", "mamba.exe"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "Scripts", "micromamba.exe"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "condabin", "conda"), existing_only=True)
        _append_unique_path(executables, os.path.join(root, "condabin", "conda.bat"), existing_only=True)

    if executables:
        _exe_discovery_log(f"Conda executables detected: {executables}")
    else:
        _exe_discovery_log("No conda/mamba executables detected on PATH or in known install locations.")
    return executables

def _run_json_command(command):
    try:
        raw = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
        return None

    if not raw:
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None

def get_chemem_paths():
    all_paths = []
    path_chemem = shutil.which("chemem")
    if path_chemem:
        _append_unique_path(all_paths, path_chemem, existing_only=True)
        _exe_discovery_log(f"Accepted executable from PATH: {_normalise_path(path_chemem)}")

    env_paths = get_conda_env_paths()
    if not env_paths:
        if all_paths:
            _exe_discovery_log("No conda-like environments found; using ChemEM executable(s) already on PATH.")
        else:
            _exe_discovery_log("No conda-like environments found to scan for ChemEM.")
        return [PathParameter(i,i) for i in all_paths]

    _exe_discovery_log(f"Scanning {len(env_paths)} environment(s) for ChemEM backend executable.")

    for env_path in env_paths:
        found, chemem_executable = check_chemem_version(env_path)
        if found:
            _append_unique_path(all_paths, chemem_executable, existing_only=True)

    _exe_discovery_log(f"ChemEM executable discovery complete. Found {len(all_paths)} executable(s).")
    return [PathParameter(i,i) for i in all_paths]

def find_anaconda_path():
    # Keep for compatibility: return a best-effort base install path.
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix and os.path.exists(conda_prefix):
        conda_prefix = _normalise_path(conda_prefix)
        if os.path.basename(os.path.dirname(conda_prefix)) == "envs":
            guessed_base = os.path.dirname(os.path.dirname(conda_prefix))
            if os.path.isdir(guessed_base):
                _exe_discovery_log(f"Resolved base from CONDA_PREFIX env path: {guessed_base}")
                return guessed_base
        _exe_discovery_log(f"Using CONDA_PREFIX as conda location: {conda_prefix}")
        return conda_prefix

    for conda_exe in _discover_conda_executables():
        try:
            conda_base = subprocess.check_output(
                [conda_exe, 'info', '--base'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
            continue

        if os.path.isdir(conda_base):
            conda_base = _normalise_path(conda_base)
            _exe_discovery_log(f"Resolved conda base using '{conda_exe}': {conda_base}")
            return conda_base

    known_roots = _known_conda_roots()
    if known_roots:
        _exe_discovery_log(f"Falling back to known conda root: {known_roots[0]}")
        return known_roots[0]

    _exe_discovery_log("No conda-like base installation found.")
    return None

def get_conda_env_paths(conda_base_path=None):
    env_paths = []
    base_paths = []

    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix and os.path.isdir(conda_prefix):
        _append_unique_path(env_paths, conda_prefix, existing_only=True)
        _exe_discovery_log(f"Added active CONDA_PREFIX environment: {_normalise_path(conda_prefix)}")

    if conda_base_path and os.path.isdir(conda_base_path):
        _append_unique_path(base_paths, conda_base_path, existing_only=True)

    for conda_exe in _discover_conda_executables():
        payload = _run_json_command([conda_exe, "env", "list", "--json"])
        if payload is not None and isinstance(payload.get("envs"), list):
            for env in payload.get("envs", []):
                _append_unique_path(env_paths, env, existing_only=True)
            _exe_discovery_log(
                f"Added {len(payload.get('envs', []))} environment path(s) from '{conda_exe} env list --json'."
            )
        else:
            _exe_discovery_log(f"Could not parse environment list from '{conda_exe}'.")

        try:
            base = subprocess.check_output(
                [conda_exe, "info", "--base"],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
            base = None

        if base and os.path.isdir(base):
            _append_unique_path(base_paths, base, existing_only=True)

    for known_root in _known_conda_roots():
        _append_unique_path(base_paths, known_root, existing_only=True)

    for base_path in _existing_directories(base_paths):
        _append_unique_path(env_paths, base_path, existing_only=True)
        envs_path = os.path.join(base_path, "envs")
        if os.path.isdir(envs_path):
            for env_name in os.listdir(envs_path):
                full_env_path = os.path.join(envs_path, env_name)
                if os.path.isdir(full_env_path):
                    _append_unique_path(env_paths, full_env_path, existing_only=True)

    env_paths = _existing_directories(env_paths)
    _exe_discovery_log(f"Total unique conda-like environments discovered: {len(env_paths)}")
    return env_paths

def check_chemem_version(env_path):
    env_path = _normalise_path(env_path)
    candidate_paths = []
    for bin_dir in (os.path.join(env_path, "bin"), os.path.join(env_path, "Scripts")):
        candidate_paths.extend(
            [
                os.path.join(bin_dir, "chemem"),
                os.path.join(bin_dir, "chemem.exe"),
                os.path.join(bin_dir, "chemem.bat"),
                os.path.join(bin_dir, "chemem.cmd"),
            ]
        )

    for chemem_executable in candidate_paths:
        if os.path.isfile(chemem_executable):
            if os.name == "nt" or os.access(chemem_executable, os.X_OK):
                _exe_discovery_log(f"Accepted '{env_path}': found ChemEM executable '{chemem_executable}'.")
                return True, _normalise_path(chemem_executable)

    _exe_discovery_log(f"Rejected '{env_path}': no ChemEM executable found in {candidate_paths}.")
    return False, None

def set_conda_environment():
    conda_path_guess = []
    for root in _known_conda_roots():
        conda_path_guess.extend(
            [
                os.path.join(root, "bin"),
                os.path.join(root, "Scripts"),
                os.path.join(root, "condabin"),
            ]
        )

    existing_path = os.environ.get("PATH", "")
    path_entries = []
    if existing_path:
        for path_entry in existing_path.split(os.pathsep):
            if path_entry:
                _append_unique_path(path_entries, path_entry, existing_only=False)

    for path in conda_path_guess:
        if os.path.isdir(path):
            norm_path = _normalise_path(path)
            old_count = len(path_entries)
            _append_unique_path(path_entries, norm_path, existing_only=True)
            if len(path_entries) > old_count:
                _exe_discovery_log(f"Added to PATH for env discovery: {norm_path}")

    os.environ["PATH"] = os.pathsep.join(path_entries)


def condense_path(path, max_length=25):
    if len(path) <= max_length:
        return path  

    part_length = (max_length - 3) // 2  # 3 characters are reserved for "..."
    start_part = path[:part_length]
    end_part = path[-part_length:]
    return f"{start_part}...{end_part}"

def get_box_vertices(centroid, box_size):
    """
    Calculate the vertices of a box given its center and dimensions.

    :param center: A tuple of (x, y, z) representing the center of the box.
    :param dimensions: A tuple of (length, width, height) of the box.
    :return: A list of tuples, each representing the coordinates of a vertex.
    """
    
    x, y, z = centroid
    xb, yb, zb = box_size

    # Half dimensions
    half_xb = xb / 2
    half_yb = yb / 2
    half_zb = zb / 2

    # Calculate the coordinates of the vertices
    vertices = [
        (x - half_xb, y - half_yb, z - half_zb),
        (x - half_xb, y - half_yb, z + half_zb),
        (x - half_xb, y + half_yb, z - half_zb),
        (x - half_xb, y + half_yb, z + half_zb),
        (x + half_xb, y - half_yb, z - half_zb),
        (x + half_xb, y - half_yb, z + half_zb),
        (x + half_xb, y + half_yb, z - half_zb),
        (x + half_xb, y + half_yb, z + half_zb)
    ]
    
    min_x_vertices = min([i[0] for i in vertices])
    max_x_vertices = max([i[0] for i in vertices])
    min_y_vertices = min([i[1] for i in vertices])
    max_y_vertices = max([i[1] for i in vertices])
    min_z_vertices = min([i[2] for i in vertices])
    max_z_vertices = max([i[2] for i in vertices])

    min_coords = np.array([min_x_vertices,min_y_vertices, min_z_vertices])
    max_coords = np.array([max_x_vertices,max_y_vertices,max_z_vertices])
    
    return min_coords, max_coords

def find_voxel_indices(min_point, max_point, apix, grid_size, origin):
    """
    Determine the minimum and maximum voxel indices within given 3D points.
    
    :param min_point: Tuple of the minimum 3D point coordinates (x, y, z) in Angstroms.
    :param max_point: Tuple of the maximum 3D point coordinates (x, y, z) in Angstroms.
    :param apix: Pixel size in Angstroms.
    :param grid_size: Array of grid sizes [nx, ny, nz].
    :param origin: Origin coordinates (x, y, z).
    :return: Two tuples representing the minimum and maximum voxel indices ((min_ix, min_iy, min_iz), (max_ix, max_iy, max_iz)).
    """
    min_indices = point_to_voxel_index(min_point, apix, origin)
    max_indices = point_to_voxel_index(max_point, apix, origin)
    
    # Ensure indices are within grid size limits
    min_indices = (
        max(0, min_indices[0]), max(0, min_indices[1]), max(0, min_indices[2])
    )
    max_indices = (
        min(grid_size[0] - 1, max_indices[0]),
        min(grid_size[1] - 1, max_indices[1]),
        min(grid_size[2] - 1, max_indices[2])
    )
    
    return min_indices, max_indices

def point_to_voxel_index(point, apix, origin):
    """
    Convert a 3D point in Angstroms to voxel indices.
    
    :param point: Tuple of the 3D point coordinates (x, y, z) in Angstroms.
    :param apix: Pixel size in Angstroms.
    :param origin: Origin coordinates (x, y, z).
    :return: Tuple of voxel indices (ix, iy, iz).
    """
    ix = int((point[0] - origin[0]) / apix[0])
    iy = int((point[1] - origin[1]) / apix[1])
    iz = int((point[2] - origin[2]) / apix[2])
    return ix, iy, iz


def generate_unique_id():
    """
    Generate a unique id of type str

    """
    return str(uuid.uuid4())

def estimate_background_distribution(density_map, cube_size=10, num_cubes=4):
    """
    Estimate the background noise distribution parameters (mean, std).
    For simplicity, we pick `num_cubes` regions along the principal axes far from the particle center.
    Assumes the particle is roughly centered.
    """
    nx, ny, nz = density_map.shape
    half_x, half_y, half_z = nx//2, ny//2, nz//2
    
    # Define cube corners far outside the center. For instance:
    # 1) along +x direction, 2) -x direction, 3) +y direction, 4) -y direction.
    # Adjust these based on known map orientation and empty regions.
    coords = [
        (nx - cube_size, nx, half_y, half_y+cube_size, half_z, half_z+cube_size),
        (0, cube_size, half_y, half_y+cube_size, half_z, half_z+cube_size),
        (half_x, half_x+cube_size, ny - cube_size, ny, half_z, half_z+cube_size),
        (half_x, half_x+cube_size, 0, cube_size, half_z, half_z+cube_size),
    ]
    
    # Extract background voxels
    background_voxels = []
    for (xstart, xend, ystart, yend, zstart, zend) in coords:
        background_voxels.append(density_map[xstart:xend, ystart:yend, zstart:zend].ravel())
    background_voxels = np.concatenate(background_voxels)
    
    # Estimate mean and std from these voxels
    bg_mean = np.mean(background_voxels)
    bg_std = np.std(background_voxels)
    return bg_mean, bg_std

def compute_p_values(density_map, bg_mean, bg_std):
    """
    Compute one-sided p-values for each voxel assuming Gaussian noise distribution.
    p-value = P(X >= value) under N(bg_mean, bg_std^2).
    """
    # Z-score each voxel
    z_map = (density_map - bg_mean) / (bg_std + 1e-12)
    
    # p-values = 1 - CDF(z)
    p_values = 1 - norm.cdf(z_map)
    return p_values

def benjamini_yekutieli(pvals, alpha=0.05):
    """
    Benjamini-Yekutieli procedure for FDR control.
    This method handles arbitrary dependencies.
    
    Returns array of q-values.
    """
    # Flatten p-values and sort
    pvals_flat = pvals.flatten()
    n = len(pvals_flat)
    sort_indices = np.argsort(pvals_flat)
    sorted_pvals = pvals_flat[sort_indices]
    
    # BY correction factor for arbitrary dependency
    # harmonic_n = sum_{i=1}^n (1/i)
    harmonic_n = np.sum(1.0 / np.arange(1, n+1))
    
    # Compute thresholded q-values
    qvals = np.zeros_like(sorted_pvals)
    min_ratio = 1.0
    for i in reversed(range(n)):
        current = sorted_pvals[i] * n * harmonic_n / (i+1)
        if current < min_ratio:
            min_ratio = current
        qvals[i] = min_ratio
    
    # Revert to original ordering
    qvals_original = np.empty_like(qvals)
    qvals_original[sort_indices] = qvals
    return qvals_original.reshape(pvals.shape)

def extract_ligand_density(density_map, 
                           binding_mask,
                           apix,
                           high_threshold_sigma=2.0,
                           ):
    
    
    
    binding_density = density_map *  (binding_mask > 0.0)
    non_zero_values = binding_density[binding_density > 0.0]
    high_threshold = np.std(non_zero_values) * high_threshold_sigma 
    
    thr_map = binding_density * (binding_density > high_threshold)
    
   
    density_smoothed = ndimage.gaussian_filter(binding_density, sigma=1.0)
    mag_grad = sobel_filtered_map(density_smoothed)
    
    labels, num_features = ndimage.label(thr_map, structure = ndimage.generate_binary_structure(3, 1))
    ligand_features = []
    ligand_densities = []
    
    mean_masked = np.mean(binding_density.ravel())
    if num_features > 0:
       
        for i in range(num_features):
            if i == 0:
                continue 
            
            map_features = {}
            dist_map = binding_mask[labels ==i]
            centroid = np.mean(dist_map)
            if centroid > 2.0:
                map_features['centroid'] = centroid 
                
                feature_map = binding_density * (labels == i)
                values_above = feature_map[labels == i]
                
                
                volume = len(values_above) * np.product(apix)
                map_features['volume'] = volume
                map_features['mean_non_zero_density_value'] = values_above.mean() 
                map_features['map_amplitude_thr_no_mask'] = density_map[density_map > 0].mean()
                map_features['map_amplitude_thr_masked'] = mean_masked
                map_features['feature_id'] = i 
                
                if volume <= 10 or values_above.mean() < density_map[density_map > 0].mean():
                    continue
                
                else:

                    full_ligand_density_mask =  grow_ligand_region(feature_map > 0.0,
                                                             binding_density > 0.0,
                                                             mag_grad,
                                                             0.4)
                    full_ligand_density = binding_density * full_ligand_density_mask 
                    ligand_features.append(map_features)
                    ligand_densities.append(full_ligand_density)
                    
    return ligand_densities, ligand_features

def sobel_filtered_map(volume, normalise=True):
    gx = ndimage.sobel(volume, axis=0)
    gy = ndimage.sobel(volume, axis=1)
    gz = ndimage.sobel(volume, axis=2)
    grad_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    if normalise:
        # Normalize the gradient magnitude to be between 0 and 1.
        grad_min = grad_mag.min()
        grad_max = grad_mag.max()
        if grad_max > grad_min:
            grad_mag = (grad_mag - grad_min) / (grad_max - grad_min)
        else:
            # In case the gradient magnitude is constant, return an array of zeros.
            grad_mag = np.zeros_like(grad_mag)
    
    return grad_mag

def grow_ligand_region(seed_mask, binding_mask, gradient_map, grad_threshold):
    """
    Grow a region starting from seed_mask, but only add voxels whose gradient value
    (from gradient_map) is above grad_threshold. This is intended to expand the ligand
    density into the surrounding binding site, but stop growing when the gradient 
    drops below grad_threshold (indicating a less-defined or bridging boundary).
    
    Parameters:
    -----------
    seed_mask : 3D numpy array (boolean)
        Binary mask of the initial ligand seed region.
    binding_mask : 3D numpy array (boolean)
        Binary mask of the overall binding site.
    gradient_map : 3D numpy array (float)
        Normalized gradient map (values between 0 and 1) from the Sobel operator.
    grad_threshold : float
        The threshold for the gradient value. Only neighboring voxels with 
        gradient_map >= grad_threshold will be added to the region.
        
    Returns:
    --------
    grown_region : 3D numpy array (boolean)
        Binary mask of the grown region.
    """
    # Start with the initial seed.
    grown_region = seed_mask.copy()
    structure = ndimage.generate_binary_structure(3, 1)  # 3D connectivity
    changed = True
    
    while changed:
        changed = False
        # Dilate the current region to get its neighbors.
        dilated = ndimage.binary_dilation(grown_region, structure=structure)
        # Restrict to the binding site and only voxels not already included.
        candidates = dilated & binding_mask & (~grown_region)
        # Now, only add candidates that have a sufficiently high gradient.
        valid = candidates & (gradient_map >= grad_threshold)
        if np.any(valid):
            grown_region |= valid
            changed = True
    
    return grown_region
