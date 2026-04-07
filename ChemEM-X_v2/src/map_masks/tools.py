#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:25:20 2025

@author: aaron.sweeney
"""


import numpy as np 
from chimerax.ChemEM.map_masks.parameters import ConfidenceMapParameter
from scipy.stats import norm
from  scipy import ndimage 
import uuid
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data
from chimerax.ChemEM.binding_site.tools import BindingSiteChemEM, RenderBindingSite

class AutoMask:
    
    "run the automated binding site analysis and ligand matching thing from ChemEM2"
    def __init__(self,
                 session,
                 current_model,
                 current_map,
                 protocol="significant-features", #option "alpha-mask"
                 use_selected = False):
        
        self.session = session 
        self.model = current_model 
        self.map = current_map
        self.protocol = protocol
    def run(self):
        
        self.map.region = self.map.full_region() #unpack map
        confidence_map = ConfidenceMap(self.session, self.map).run()
        
        
        new_map = np.array(confidence_map.value.grid_data().matrix().shape)
        new_origin = np.array(confidence_map.value.grid_data().origin)
        new_apix = np.array(confidence_map.value.grid_data().step)
        
        print('#######################')
        
        
        print(confidence_map.value)
        print('#######################')
        
        #i need binding sites facotry for alpha mask
        #sig feats can run with binding sites.run object
        binding_sites_factory = BindingSiteChemEM(self.session,
                                        self.model,
                                        confidence_map.value) #pass conf map here?
        
        
        binding_sites = binding_sites_factory.run()
        
        if self.protocol == "significant-features":
            #create a combined map from features its more informative 
            
            
            for binding_site in binding_sites:
                binding_site.add_to_session(self.session, add_to_session=False)
                render_object = RenderBindingSite(self.session, 
                                            binding_site, 
                                            self.model,
                                            current_map=confidence_map.value, 
                                            add_to_session=True) #don't render yet
                
                binding_site_mask = BindingSiteMask(self.session, 
                                                    render_object,
                                                    confidence_map=confidence_map.value).run()
                
                features = SignificantFeatures(self.session,
                                               render_object,
                                               binding_site_mask).run()
                #reset 
                render_object.reset()
                self.session.models.remove([binding_site_mask])

                if features:
                    print("-!-!-!", features)
                    for density, features, self.apix, self.origin in features:
                        #real to array
                        pass
                        
            confidence_map.value.display = False
            
        elif self.protocol == "alpha-mask":
            pass




class SignificantFeatures:
    
    def __init__(self,
                 session, 
                 binding_site,
                 masked_map
                 ):
        self.session = session 
        self.binding_site = binding_site 
        self.masked_map = masked_map 
    
    def extract_features(self):
        
        ligand_densities, ligand_features = extract_ligand_density(self.grid_data_mask,
                                                          self.grid_data_binding_site_distance_map,
                                                          self.apix)
        
        site_ligands = {}
        if ligand_features:
            max_thr = np.max([i['map_amplitude_thr_no_mask'] for i in ligand_features])
            
            for num, (density, features) in enumerate(zip(ligand_densities, ligand_features)):
                features = {key: float(value) for key, value in features.items()}
                grid_data, volume = self.add_map_to_session(density)
                name = f"Feature map {num}"
                volume.name = name
                #here i think i need to save the features as i don't want to add them to the session individually
                site_ligands[num] = [density, features, self.apix, self.origin]
            
        self.site_ligands = site_ligands
    
    def filter_features(self):
        #TODO!!
        pass
            
            
    def add_map_to_session(self, densmap):
        grid_data = ArrayGridData(densmap,
                                  self.origin, 
                                  self.apix)
        
        
        volume = volume_from_grid_data(grid_data, self.session, style="surface")
        return grid_data, volume        
    
    def run(self):
        self.grid_data_mask = self.masked_map.matrix() 
        
        self.grid_data_binding_site_distance_map = self.binding_site.binding_site.volume.matrix() 
        self.origin, self.apix = self.binding_site.map.data_origin_and_step()
        self.extract_features()
        return self.site_ligands

class BindingSiteMask:
    
    def __init__(self, session, binding_site, confidence_map=None):
        self.session = session 
        self.binding_site = binding_site 
        self.confidence_map = confidence_map
    
    def calculate_binding_site_mask(self):
        
        self.binding_site_masked = self.grid_data_densmap.copy() * (self.grid_data_binding_site > 0.1)
    
    
    def add_map_to_session(self):
        self.grid_data = ArrayGridData(self.binding_site_masked,
                                  self.origin, 
                                  self.apix)
        
        self.volume = volume_from_grid_data(self.grid_data, self.session, style="mesh")
        self.volume.name = "binding site masked"
        
    def match_regions(self):
        
        if self.binding_site.binding_site.volume.region != self.confidence_map.region:
            self.confidence_map.region = self.confidence_map.full_region()
            
            self.confidence_map.region = self.binding_site.binding_site.volume.region
    
    def run(self):
        
        if self.confidence_map is not None:
            #self.match_regions()
            self.grid_data_densmap = self.confidence_map.matrix()
        else:
            self.grid_data_densmap = self.binding_site.map.matrix()
        
        
        self.origin, self.apix = self.binding_site.map.data_origin_and_step()
        
       
        self.grid_data_binding_site = self.binding_site.binding_site.volume.matrix() 
        self.calculate_binding_site_mask()
        self.add_map_to_session()
        return self.volume
    

    
class ConfidenceMap:
    def __init__(self, session, densmap):
        
        self.densmap = densmap
        self.session = session
        self.use_model = False
        self.radius_in_angstrom = 6.0
    
    
        
    def get_full_map(self):
        
        try:
            self.full_map = self.densmap.grid_data().full_data.matrix().copy()
            self.origin = self.densmap.grid_data().full_data.origin
        except AttributeError:
            self.full_map = self.densmap.grid_data().matrix().copy()
            self.origin = self.densmap.grid_data().origin
   
    
        self.apix = self.densmap.grid_data().step
        self.region = self.densmap.region
        
    
    def estimate_background_from_cubes(self):
        self.bg_mean, self.bg_std  = estimate_background_distribution(self.full_map, cube_size=10, num_cubes=4)
    
    def compute_confidence_map(self, cube_size=10, num_cubes=4, desired_ppv=0.99):
        """
        Compute a confidence map from the given density map.
        """
      
        pvals = compute_p_values(self.full_map, self.bg_mean, self.bg_std)
        qvals = benjamini_yekutieli(pvals)
        ppvs = 1 - qvals
        thresholded_map = (ppvs >= desired_ppv).astype(np.float32)
        masked_map = self.full_map.copy() 
        masked_map = masked_map * thresholded_map
        self.confidence_map =  masked_map
        #ConfidenceMapParameter('confidence_map', masked_map)
        
    def add_map_to_session(self):
        self.grid_data = ArrayGridData(self.confidence_map,
                                  self.origin, 
                                  self.apix)
        
        self.volume = volume_from_grid_data(self.grid_data, self.session, style="mesh")
        self.volume.name = "confidence map"
        self.volume.region = self.region
    
    
    def run(self):
        self.get_full_map()
        
        if self.use_model:
            pass
        else:
            self.estimate_background_from_cubes()
        
        self.compute_confidence_map()
        self.add_map_to_session()
        #ConfidenceMapParameter('confidence_map', self.volume)
        return ConfidenceMapParameter('confidence_map', self.volume)
    
    

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
