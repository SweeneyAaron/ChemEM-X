# This file is part of the ChemEM-X software.
#
# Copyright (c) 2024 - Aaron Sweeney
#
# This module was developed by:
#   Aaron Sweeney    <aaron.sweeney AT cssb-hamburg.de>

import numpy as np 
from scipy.spatial import distance_matrix 
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage import label
from scipy.ndimage import  grey_erosion, grey_opening, grey_closing, grey_dilation
from scipy.ndimage import gaussian_filter
from skimage import filters
from chimerax.map_data import ArrayGridData
from chimerax.map import Volume


class SignificantFeatures:
    def __init__(self, session, sig_feat, shape, origin, apix):
        self.session = session
        self.significant_features_data = sig_feat
        self.percent_above_otsu_threshold = 0.0 # values between 0. and 1.
        self.non_zero_voxel_count_threshold = 50 #basically size !!! 
        self.shape = shape 
        self.origin = origin 
        self.apix = apix 
        self.js_code = []
        self.current_filtered_maps = []
        self.reset_map()
        self.chimerax_map_object = self.make_binding_site_map_object(self.current_map_data,
                                          self.origin,
                                          self.apix)
        self.filter_maps()
        
    def reset_map(self):
        
        self.current_map_data = np.zeros(self.shape)
        
    def filter_maps(self):
        current_filtered_maps = []
        for num, sig_feat in enumerate(self.significant_features_data):
            #filter for voxels % above threshold
            if sig_feat[0][2] > self.percent_above_otsu_threshold:
                #filter for voxels % above threshold 
                #TODO! make a maximum_threshold and slider.
                if sig_feat[0][0] > self.non_zero_voxel_count_threshold:
                    current_filtered_maps.append(sig_feat)
        self.current_filtered_maps = current_filtered_maps
        self.add_filtered_maps_to_list()
        #TODO! call filtered_maps from here need to link index from current_filtered_maps and list ids
        
    def add_filtered_maps_to_list(self):
        self.js_code = []
        for num, sig_feat in enumerate(self.current_filtered_maps): 
            string = f'{num} - % > Thr : {sig_feat[0][2]}, n > 0 : {sig_feat[0][0]}'
            self.js_code.append((num, string))
            #TODO!! display these with buttons!
    
    def show_feature(self, idx):
        self.current_map_data += self.current_filtered_maps[idx][1]
        self.show_map()
        
    def hide_feature(self, idx):
        self.current_map_data -= self.current_filtered_maps[idx][1]
        self.show_map()
        
    def show_map(self):
        if self.chimerax_map_object is not None:
            self.session.models.remove([self.chimerax_map_object])
        self.chimerax_map_object = self.make_binding_site_map_object(self.current_map_data,
                                                                     self.origin, 
                                                                     self.apix)
    
    def make_binding_site_map_object(self, densmap, origin, apix, name = 'SigFeatMap'):
        grid_data = ArrayGridData(densmap, 
                                  origin=origin, 
                                  step = apix,
                                  name = name)
        
        volume_object = Volume(self.session, grid_data)
        self.session.models.add([volume_object])
        volume_object.display = False
        volume_object.display = True
        return volume_object
        

class SignificantFeaturesProtocol:
    def __init__(self, session, binding_site, resolution):
        self.session = session 
        self.binding_site = binding_site 
        self.resolution = resolution
        self.percent_above_otsu_threshold = 0.0 # values between 0. and 1.
        self.non_zero_voxel_count_threshold = 50
    
    def process_site(self):
        inside_atoms = self.binding_site.inside_atoms 
        atom_radii = np.array([i.radius for i in inside_atoms])
        origin, apix = self.binding_site.map.data_origin_and_step() 
        shape = self.binding_site.map.matrix().shape 
        
        vectorised_voxels, map_indices = self.vectorise_points(shape, np.array(origin), np.array(apix))
        points_ouside_mask = self.find_points_outside_atoms(inside_atoms, atom_radii, 
                                                         vectorised_voxels)
        map_indices = map_indices[points_ouside_mask]
        densmap = self.build_map(shape, map_indices)
        feature_maps = self.significant_features(densmap)
        
        feature_object = SignificantFeatures(self.session,
                                             feature_maps, 
                                             shape,
                                             origin,
                                             apix)
        #Temporary function
        #self.filter_and_show(feature_maps, origin, apix)

        #chimera_map_object = self.make_binding_site_map_object(densmap, origin, apix)
        return feature_object
    
    def filter_and_show(self, feature_maps, origin, apix):
        for num, sig_feat in enumerate(feature_maps):
            #filter for voxels % above threshold
            if sig_feat[0][2] > self.percent_above_otsu_threshold:
                #filter for voxels % above threshold 
                #TODO! make a maximum_threshold and slider.
                if sig_feat[0][0] > self.non_zero_voxel_count_threshold:
                    name = f'SigFeatTestMap_{num}'
                    mp_obj = self.make_binding_site_map_object(sig_feat[1], origin, apix, name = name)
    
    def significant_features(self, densmap):
        
        feature_maps = []
        
        #get the full map noise approximation!
        full_map = self.binding_site.map.data.matrix()
        full_map = full_map  * (full_map > 0)
        full_map_threshold = filters.threshold_otsu(full_map.ravel())
        
        
        densmap_copy = densmap.copy()
        
        structuring_element = generate_binary_structure(3, 1)
        image_3d = densmap_copy * (densmap_copy  > 0)
        eroded_image = grey_opening(image_3d, footprint=structuring_element)
        flattened = eroded_image.ravel()
        global_thresh = filters.threshold_otsu(flattened)
        segmented_density = self.get_disconected_densities(eroded_image, global_thresh)
        sigma = self.get_sigma(self.resolution)
        for num in np.unique(segmented_density)[1:]:
            mask_1 = (segmented_density == num)
            map_1 = eroded_image * mask_1
            map_1_closed = grey_closing(map_1, footprint=structuring_element)
            smooth_mask = self.smooth_image(map_1_closed , sigma)
            flattened = smooth_mask.ravel()
            smooth_thresh = filters.threshold_otsu(flattened)
            threshold_image = smooth_mask * (smooth_mask > smooth_thresh)
            masked_region = densmap_copy * (threshold_image > 0)
            #feature_maps.append(masked_region)
            n_voxels_above_zero, n_voxels_above_otsu, percent_above_otsu = self.get_percent_over_threshold( masked_region, full_map_threshold)
            feature_maps.append([ ( n_voxels_above_zero, n_voxels_above_otsu, percent_above_otsu),
                                masked_region])
        return feature_maps
            
            
    
    def  get_percent_over_threshold(self, masked_map, threshold):
        
        count_above_zero = np.sum(masked_map > 0)
        
        count_above_otsu = np.sum(masked_map > threshold)
        percent = round(count_above_otsu / count_above_zero , 2)
        
        return count_above_zero, count_above_otsu, percent
        
    
    def smooth_image(self,image, sigma):
        smoothed_image = gaussian_filter(image, sigma=sigma)
        return smoothed_image
    
    def get_sigma(self, resolution, sigma_coeff=0.356):
        return resolution * sigma_coeff
    
    def get_disconected_densities(self, image, threshold, struct = None):
        if struct is None:
            struct = generate_binary_structure(3, 1)
            
        bool_image = (image > threshold)
        labels, num_features = label(bool_image, structure = struct)
        return labels

    
    def min_value_greater_than_zero(self, array_3d):
        """
        Find the minimum value greater than 0 in a 3D numpy array.
        
        Parameters:
        - array_3d: numpy.ndarray, the 3D array to search in.
        
        Returns:
        - The minimum value greater than 0, or None if no such value exists.
        """
        # Filter the array to include only values greater than 0
        filtered_array = array_3d[array_3d > 0]
        
        # If the filtered array is not empty, return the minimum value; otherwise, return None
        if filtered_array.size > 0:
            return filtered_array.min()
        else:
            return None
        
        
    def build_map(self, shape, indices_outside):
        copy_map = np.zeros(shape)#, dtype=np.float64)
        
        map_data = self.binding_site.map.matrix()
        
        for index in indices_outside:
           
            copy_map[index[0],index[1],index[2]] = map_data[index[0],index[1],index[2]]
       
        return copy_map
        
        
        
    
    def make_binding_site_map_object(self, densmap, origin, apix, name = 'SigFeatTestMap'):
        grid_data = ArrayGridData(densmap, 
                                  origin=origin, 
                                  step = apix,
                                  name = name)
        
        volume_object = Volume(self.session, grid_data)
        self.session.models.add([volume_object])
        return volume_object
    
    
    @staticmethod
    def mask(session, rendered_site, resolution):
        instance = SignificantFeaturesProtocol(session, rendered_site, resolution)
        return instance.process_site()
    
    def vectorise_points(self, shape, origin, apix):
        z_indices, y_indices, x_indices = np.indices(shape)
        
        real_world_coords_x = origin[0] + x_indices * apix[0]
        real_world_coords_y = origin[1] + y_indices * apix[1]
        real_world_coords_z = origin[2] + z_indices * apix[2]
        combined_coords = np.stack([real_world_coords_x, real_world_coords_y, real_world_coords_z], axis=-1)
        vectorized_voxels = combined_coords.reshape(-1, 3)
        indices = np.stack([z_indices, y_indices, x_indices], axis=-1).reshape(-1, 3)
        
        return vectorized_voxels, indices
    
    def find_points_outside_atoms(self, inside_atoms, atomic_radii_array,
                                  vectorised_voxels):
        atomic_coords_array = np.array([np.array(i.coord) for i in  inside_atoms])
        dist_matrix = distance_matrix(vectorised_voxels, atomic_coords_array)
        is_outside_matrix = dist_matrix > atomic_radii_array
        points_outside_mask = np.all(is_outside_matrix, axis=1)
        return points_outside_mask
        
        
    
'''     
        

class FloodFillBindingSite:
    def __init__(self, session, atoms, centroid, box_size, map_slice, apix, key,origin):
        
        self.session = session
        self.atoms = atoms
        self.density_map = map_slice
        self._centroid = centroid 
        self._box_size = box_size
        self.apix = apix
        self.grid_spacing = 0.5
        self.contact_tol = 0.0
        self.key = key
        self.color = Color('orange')
        self.origin = origin
        #run
        self.get_box_vertices()
        self.flood_fill_site()
        
    
    
    @property 
    def centroid(self):
        return self._centroid
    
    @centroid.setter 
    def centroid(self, value):
        self._centroid = value 
    
    @property 
    def box_size(self):
        return self._box_size 
    @box_size.setter 
    def box_size(self, value):
        self._box_size = value
        
    
    def flood_fill_site(self):
        self.inside_atoms = [i for i in self.atoms if not self.is_outside(i.coord)]
        self.atom_radii = [i.radius for i in self.inside_atoms]
        #self.atom_radii = [2.0 for i in self.inside_atoms]
        self.vectorise_points()
        self.find_points_outside_atoms()
        
        #show site
        self.add_model()
        self.add_atoms()
        self.add_map()
        copy_map = np.zeros(self.density_map.shape, dtype=np.float64)
    
    
    def add_map(self):
        copy_map = np.zeros(self.density_map.shape)#, dtype=np.float64)
        
        
        for index in self.indices_outside:
           
            copy_map[index[0],index[1],index[2]] = self.density_map[index[0],index[1],index[2]]
       
        
        self.map = copy_map
        self.make_binding_site_map_object(copy_map, self.origin, self.apix)
    
    
        
        

    def make_binding_site_map_object(self, densmap, origin, step):
        grid_data = ArrayGridData(densmap, 
                                  origin=origin, 
                                  step = step,
                                  name = 'M-BSM-test')
        volume_object = Volume(self.session, grid_data)
        self.map_object = volume_object
        self.session.models.add([volume_object])
        
        
    def find_points_outside_atoms(self):
        atomic_coords_array = np.array([np.array(i.coord) for i in  self.inside_atoms])
        atomic_radii_array = np.array(self.atom_radii)
        atomic_radii_array = np.array(self.atom_radii)
        dist_matrix = distance_matrix(self.vectorised_voxels, atomic_coords_array)
        self.dist_matrix = dist_matrix
        is_outside_matrix = dist_matrix > atomic_radii_array
        points_outside_mask = np.all(is_outside_matrix, axis=1)
        self.points_outside  = self.vectorised_voxels[points_outside_mask]
        self.indices_outside = self.map_indices[points_outside_mask]
        
    def find_points_outside_atoms_old(self):
        t1 = time.perf_counter()
        # Convert lists to NumPy arrays for efficient computation
        atomic_coords_array = np.array([np.array(i.coord) for i in  self.inside_atoms])
        atomic_radii_array = np.array(self.atom_radii)[:, np.newaxis]  # Reshape for broadcasting
    
        # Calculate the squared distances between each point and each atom
        # This avoids the sqrt operation, which is unnecessary for comparison
        diff = self.vectorised_voxels[:, np.newaxis, :] - atomic_coords_array
        squared_distances = np.sum(diff ** 2, axis=2)
    
        # Check if any distances are within the squared radii
        within_radii = squared_distances <= atomic_radii_array.T ** 2
    
        # Determine points outside all atoms (no distance within any radius)
        outside_all_atoms = ~np.any(within_radii, axis=1)
    
        # Filter points and indices for those outside all atoms
        self.points_outside = self.vectorised_voxels[outside_all_atoms]
        self.indices_outside = self.map_indices[outside_all_atoms]
        print('rt np', time.perf_counter() - t1)
   
    
    def vectorise_points(self):
        z_indices, y_indices, x_indices = np.indices(self.density_map.shape)
        real_world_coords_x = self.origin[0] + x_indices * self.apix[0]
        real_world_coords_y = self.origin[1] + y_indices * self.apix[1]
        real_world_coords_z = self.origin[2] + z_indices * self.apix[2]
        combined_coords = np.stack([real_world_coords_x, real_world_coords_y, real_world_coords_z], axis=-1)
        vectorized_voxels = combined_coords.reshape(-1, 3)
        indices = np.stack([z_indices, y_indices, x_indices], axis=-1).reshape(-1, 3)
        self.vectorised_voxels = vectorized_voxels
        self.map_indices = indices
    
    def is_outside(self, atom_coord):
        if np.any(np.array(atom_coord) < self.min_coords):
            return True
        elif np.any(np.array(atom_coord) > self.max_coords):
            return True 
        else:
            return False
    
    
    def get_box_vertices(self):
        """
        Calculate the vertices of a box given its center and dimensions.
    
        :param center: A tuple of (x, y, z) representing the center of the box.
        :param dimensions: A tuple of (length, width, height) of the box.
        :return: A list of tuples, each representing the coordinates of a vertex.
        """
        
        x, y, z = self.centroid
        xb, yb, zb = self.box_size
    
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
 
        self.min_coords = np.array([min_x_vertices,min_y_vertices, min_z_vertices])
        self.max_coords = np.array([max_x_vertices,max_y_vertices,max_z_vertices])
        
    def add_model(self):
        
        
        new_model = AtomicStructure(self.session)
        new_model.name = f'Binding site: {self.key}'
        self.session.models.add([new_model])
        self.model = new_model
    
    def add_atoms(self):
       # if len(self.points_outside) > 1000:
           # plot_points = self.points_outside[::10]
       # else:
        plot_points = self.points_outside
        for idx, point in enumerate(plot_points):
            self.add_point(point, idx)
    
    
    def add_point(self, point, idx):
        new_residue = self.get_residue(idx)
        atom_1 = self._add_atom_point(new_residue, point)
        self.set_point_params(atom_1)
    
    def set_point_params(self, atom):
        
        color = self.set_transparency(self.color)
        atom.color = color.uint8x4()
        atom.radius = 0.1
        atom.draw_mode = atom.SPHERE_STYLE
        atom.display = True
    
    
    def get_residue(self, idx):
         
         new_residue_id = 'ProPLID Sphere'
         new_residue = self.model.new_residue(new_residue_id , str( idx ), idx)
         return new_residue
    
    def _add_atom_point(self, residue, point):

        atom = self.model.new_atom('Point', 'C')        
        residue.add_atom(atom)
        atom.coord = point
        return atom

    
    def set_transparency(self, color):
        color = color.rgba.copy()
        color[3] = 0.25
        return Color(color)
    '''
    