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
from rdkit import Chem
from openmm import Platform
from chimerax.core.tasks import Task, TaskState
from chimerax.ChemEM.core.parameters import PathParameter
import datetime


#ChemEM Job 
CHEMEM_JOB = 'chemem'
SIMULATION_JOB = 'chemem-X simulation'
EXPORT_SIMULATION = 'export simulation'

class ChemEMJob(Task):
    def __init__(self, session, command, job_type):
        
        super().__init__(session)
        self.command = command 
        self.log_file = []
        self.job_type = job_type #should be able to load the relevent files from the command now 
        self.success = None
        
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

                err = self.process.stderr.read()
                if err:
                    self.log_file.append(f"Error: {err.strip()}\n")
                    self.thread_safe_log(f"Error: {err.strip()}")
                    
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
        

def mol_from_sdf(sdf_file, conf_num = 0):
    suppl = Chem.SDMolSupplier(sdf_file)
    mol = suppl[conf_num]
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

def get_chemem_paths():
    conda_base_path = find_anaconda_path()
    if not conda_base_path:
        print("Anaconda installation not found.")
        return []

    env_paths = get_conda_env_paths(conda_base_path)
    
    all_paths = []
    for env_path in env_paths:
        found, chemem_executable = check_chemem_version(env_path)
        if found:
            all_paths.append(chemem_executable)
            
    return [PathParameter(i,i) for i in all_paths]

def find_anaconda_path():
    # Ensure conda is in the PATH
    set_conda_environment()

    # Check for CONDA_PREFIX environment variable
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix and os.path.exists(conda_prefix):
        return conda_prefix

    # Check if 'conda' command is available in the PATH
    try:
        conda_path = subprocess.check_output(['conda', 'info', '--base'], stderr=subprocess.DEVNULL).decode().strip()
        if os.path.exists(conda_path):
            return conda_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return None

def get_conda_env_paths(conda_base_path):
    envs_path = os.path.join(conda_base_path, 'envs')
    if os.path.exists(envs_path):
        
        return [os.path.join(envs_path, env) for env in os.listdir(envs_path)]
    return []

def check_chemem_version(env_path):
    bin_dir = os.path.join(env_path, 'bin')
    if not os.path.exists(bin_dir):
        bin_dir = os.path.join(env_path, 'Scripts')  # For Windows compatibility
    
    #TODO!!!change to set version 0.4
    #Does not guarentee V2 
    chemem_executable = os.path.join(bin_dir, 'chemem')
    if os.path.exists(chemem_executable):
        
        return True,  os.path.join(bin_dir, 'chemem')
        
    return False, None

def set_conda_environment():
    conda_path_guess = [
        os.path.expanduser("~/anaconda3/bin"),
        os.path.expanduser("~/miniconda3/bin"),
        "C:\\ProgramData\\Anaconda3\\Scripts",
        "C:\\ProgramData\\Miniconda3\\Scripts",
        "C:\\Anaconda3\\Scripts",
        "C:\\Miniconda3\\Scripts"
    ]
    
    for path in conda_path_guess:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path


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

