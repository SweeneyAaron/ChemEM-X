#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 12:24:31 2025

@author: aaron.sweeney
"""

from collections import defaultdict
import numpy as np 
from scipy.spatial import Delaunay, distance, KDTree, distance_matrix, cKDTree
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fclusterdata
from scipy.ndimage import distance_transform_edt
from skimage import morphology
import networkx as nx
from chimerax.map_data import ArrayGridData
from chimerax.map import volume_from_grid_data
import uuid

class BindingSiteContainer:
    def __init__(self):
        self.binding_sites = []
    
    def __iter__(self):
        return iter(self.binding_sites)
    
    def add_site(self, binding_site):
        if isinstance(binding_site, ChemEM2BindingSite):
            if binding_site in self.binding_sites:
                return self.binding_sites.index(binding_site)
            else:
                self.binding_sites.append(binding_site)
                return self.binding_sites.index(binding_site)
    
    def site_from_centroid(self, centroid):
        centroid = np.array(centroid)
        centroid_reshaped = centroid.reshape(1, -1)
        for site in self.binding_sites:
            
        
            distances = cdist(centroid_reshaped, site.site_centers)
            contacts = distances < site.site_radii[np.newaxis, :]
            centroid_indices, _ = np.where(contacts)
            if len(centroid_indices) > 0:
                #This should be combined if they are is more than 1 site!!
               return site
        return None
        
        

class ChemEM2BindingSite:
    def __init__(self, 
                 site_id,
                 residues, 
                 origin,
                 box_size,
                 apix,
                 grid_data,
                 site_centers,
                 site_radii,
                 centroid,
                 solvent_openings
                 ):
        self.name = site_id
        self.residues = residues 
        self.origin = origin 
        self.box_size = box_size 
        self.apix = apix 
        self.grid_data =grid_data
        self.site_centers = site_centers 
        self.site_radii = site_radii 
        self.centroid = centroid
        self.solvent_openings = solvent_openings
        self.volume = None
        self._type = 'auto'
        self.value = [centroid, box_size]
    def add_to_session(self, session):
        if self.volume is None:
            self.volume = volume_from_grid_data(self.grid_data, session, style="mesh")
            # bind callback to instance.
            #need to do this as volume opens asynchronusly so need to time change right
            self._volume_change_callback = lambda vol, ct: self._on_volume_change(vol, ct)
            self.volume.add_volume_change_callback(self._volume_change_callback)

    def _on_volume_change(self, volume, change_type):
        if getattr(self, '_updating_volume', False):
            return
        self._updating_volume = True
        try:
            if volume.surfaces:
                volume.set_display_style('mesh')
                volume.set_parameters(surface_levels=[1.0])
                volume.remove_volume_change_callback(self._volume_change_callback)
        finally:
            self._updating_volume = False

    def remove_from_session(self, session):
        if self.volume is not None:
            session.models.remove([self.volume])
            self.volume = None
    

            
class AutoSiteChimera:
    
    def __init__(self, 
                 session,
                 current_model,
                 current_map = None
                 ):
        self.session = session 
        self.model = current_model
        self.map = current_map
        #options 
        self.PROBE_SPHERE_MIN = 3.0  # Minimum radius for binding site detection
        self.PROBE_SPHERE_MAX = 6.0  # Maximum radius for binding site detection
        self.FIRST_PASS_THRESHOLD = 1.73  # Distance threshold for first-pass clustering
        self.MIN_CLUSTER_SIZE = 35  # Minimum number of tetrahedra in a cluster
        self.SECOND_PASS_THRESHOLD = 4.5  # Distance threshold for second-pass clustering
        self.padding = 6.0#residue padding around binding site
        self.grid_spacing = 1.0
        self.RADIUS_CUTOFF = 2.5 #third pass cluster radius cutoff
        self.N_OVERLAPS = 2 #thrid pass cluster number of overlaps
        self.n_opening_voxels = 10 # The minimium number of voxels needed to be considered a binding site opening 
        self.voxel_buffer = 1.5 
        self.binding_site_container = BindingSiteContainer()
        
    
    def get_position_input(self):
        
        
        self.atoms = [atom for atom in self.model.atoms if atom.element.number > 1]
        self.positions = np.array([np.array(atom.coord) for atom in self.atoms])
        self.atom_radii = np.array([atom.radius for atom in self.atoms])
        
    def set_grid_spacing(self):
        
        if self.map is not None:
            self.grid_spacing = self.map.data_origin_and_step()[1][0] #apix from map
        else:
            self.grid_spacing = 1.0
    
    def get_delaunay(self):
        
        self.delaunay = Delaunay(self.positions)
        self.tetrahedra = self.delaunay.simplices 
    
    def calculate_circumsphere(self):
        
        self.circumspheres = []
        for tetra in self.tetrahedra:
            vertices = self.positions[tetra]
            center, radius = compute_circumsphere(vertices)
            self.circumspheres.append((center, radius))
    
    def filter_circumspheres(self):
        self.candidate_tetrahedra = []
        self.candidate_centers = []
        self.candidate_radii = []

        for i, (center, radius) in enumerate(self.circumspheres):
            if self.PROBE_SPHERE_MIN <= radius <= self.PROBE_SPHERE_MAX:
                self.candidate_tetrahedra.append(self.tetrahedra[i])
                self.candidate_centers.append(center)
                self.candidate_radii.append(radius)
    
    def first_pass_filter(self):
        '''

        Cluster circumsphere centers of candidate tetrahedra to group nearby tetrahedra.
        
        MIN_CLUSTER_SIZE (Int): cluster size cutoff for significance
        -------

        '''
        labels = fclusterdata(self.candidate_centers, 
                              t=self.FIRST_PASS_THRESHOLD, 
                              criterion='distance')
        
        clusters = defaultdict(list)
        for label, tetrahedron in zip(labels, self.candidate_tetrahedra):
            clusters[label].append(tetrahedron)
        
        # Keep only significant clusters
        self.first_pass_clusters = {label: tetras for label, tetras in clusters.items() if len(tetras) >= self.MIN_CLUSTER_SIZE}
    
    def second_pass_cluster(self):
        '''
        cluster the centroids of first-pass clusters to identify distinct binding sites.
        
        '''
        cluster_centroids = []
        cluster_labels = []
        
        for label, tetras in self.first_pass_clusters.items():
            centers = [compute_circumsphere(self.positions[tetra])[0] for tetra in tetras]
            centroid = np.mean(centers, axis=0)
            cluster_centroids.append(centroid)
            cluster_labels.append(label)
        
        labels_2 = fclusterdata(cluster_centroids, 
                                t=self.SECOND_PASS_THRESHOLD, 
                                criterion='distance')
        
        
        binding_sites = defaultdict(list)
        for new_label, old_label in zip(labels_2, cluster_labels):
            binding_sites[new_label].extend(self.first_pass_clusters[old_label])
        
        
        self.second_pass_clusters = binding_sites
    
    def third_pass_cluster(self):
        
       
        
        N_pockets = len(self.second_pass_clusters)
        
        # Step 1: Extract ordered_labels
        ordered_labels = list(self.second_pass_clusters.keys())
        
        # Create the mapping from original keys to indices
        label_to_index = {label: i for i, label in enumerate(ordered_labels)}
        
        linkage_counts = [{} for _ in range(N_pockets)]
        all_centers = []
        all_labels = []
        
        # Populate all_centers and all_labels with integer indices
        for label, tetras in self.second_pass_clusters.items():
            centers = [compute_circumsphere(self.positions[tetra])[0] for tetra in tetras]
            pocket_index = label_to_index[label]
            all_centers.extend(centers)
            all_labels.extend([pocket_index] * len(centers))
        
        all_centers = np.array(all_centers)
        tree = KDTree(all_centers)
        
        for i, sphere in enumerate(all_centers):
            x, y, z = sphere
            pid = all_labels[i]  # pid is now an integer index
            idxs = tree.query_ball_point([x, y, z], self.RADIUS_CUTOFF)
            for nb_idx in idxs:
                if nb_idx == i:
                    continue
                nb_pid = all_labels[nb_idx]  # nb_pid is also an integer index
                if nb_pid != pid:
                    if nb_pid not in linkage_counts[pid]:
                        linkage_counts[pid][nb_pid] = 0
                    linkage_counts[pid][nb_pid] += 1
        
       
        
        G = nx.Graph()
        G.add_nodes_from(range(N_pockets))
        
        for a in range(N_pockets):
            for b, count in linkage_counts[a].items():
                if count >= self.N_OVERLAPS:
                    G.add_edge(a, b)
        
        clusters = list(nx.connected_components(G))
        

        binding_sites = defaultdict(list)
        for new_label, cluster in enumerate(clusters):
            for idx in cluster:
                old_label = ordered_labels[idx]
                binding_sites[new_label].extend(self.second_pass_clusters[old_label])
        
        
        self.ligand_binding_clusters = binding_sites
    
    
    def get_binding_sites(self):
        #self.binding_sites = AutoBindingSiteGroup()
        self.binding_sites = {} 
        for site_label, tetras in self.ligand_binding_clusters.items():
            
            site_centers = []
            site_radii = []
            for tetra in tetras:
                vertices = self.positions[tetra]
                center, radius = compute_circumsphere(vertices)
                site_centers.append(center)
                site_radii.append(radius)
            
            site_centers = np.array(site_centers)
            site_radii = np.array(site_radii)
            
            # Compute distances between site centers and atom positions
            dist_matrix = distance_matrix(site_centers, self.positions)
            
            # Sum of site sphere radius and atom radius
            radii_sum = site_radii[:, np.newaxis] + self.atom_radii
            
            # Identify contacts where distance is less than radii sum
            contacts = dist_matrix < radii_sum + self.padding
            contact_indices = np.where(contacts)
            unique_atom_indices = np.unique(contact_indices[1])
            
            bounding_residues = [] #key 1
            for index in unique_atom_indices:
                if self.atoms[index].residue not in bounding_residues:
                    bounding_residues.append(self.atoms[index].residue)
            
            # Compute centroid of the binding site 
            binding_site_centroid = np.mean(site_centers, axis=0) #key 2
            
            # Compute bounding box dimensions
            min_coords = np.min(site_centers, axis=0)
            max_coords = np.max(site_centers, axis=0)
            bounding_box_size = max_coords - min_coords #key 3
            
            data = {'residues' : bounding_residues, 
                    'tetrahedrals': tetras,
                    'unique_atom_indices': unique_atom_indices,
                    'binding_site_centroid': binding_site_centroid ,
                    'min_coords': min_coords ,
                    'max_coords' : max_coords ,
                    'bounding_box_size': bounding_box_size,
                    'site_centers' : site_centers,
                    'site_radii' : site_radii 
                    }
            
            origin, box_size, densmap = generate_density_map(site_centers, site_radii, grid_spacing=self.grid_spacing)
            site_mask = create_binding_site_mask(densmap.shape, site_centers, site_radii, tuple([self.grid_spacing]*3), origin)
            
            boundry_mask = find_binding_site_boundary(site_mask)
            distance_map = compute_distance_map(boundry_mask, self.grid_spacing)
            distance_map *= densmap > 0.1
            
            dist_copy = distance_map.copy()
            dist_copy = dist_copy.astype(np.float32)
            
            grid_data = ArrayGridData(distance_map,
                                      origin,
                                      tuple([self.grid_spacing]*3))
            
            #Estimate solvent entry points 
            binding_site_map = densmap * (densmap > 0.1 )
            exterior_shell = create_exterior_shell(binding_site_map, dilation_radius=2)
            
            atom_positions = np.array([self.positions[i] for i in unique_atom_indices])
            atom_radii = np.array([self.atom_radii[i] for i in unique_atom_indices])
            
            exterior_shell = filter_exterior_shell_by_protein_atoms(exterior_shell,
                                                                    atom_positions,
                                                                    atom_radii,
                                                                    origin,
                                                                    tuple([self.grid_spacing]*3),
                                                                    buffer = self.voxel_buffer
                                                                    )
            
            exterior_shell = exterior_shell.astype(np.float32)
            exterior_shell_grid_data = ArrayGridData(exterior_shell,
                                                     origin,
                                                     tuple([self.grid_spacing]*3))
            
            self.binding_site_container.add_site(ChemEM2BindingSite(str(uuid.uuid4()),
                               bounding_residues,
                               origin,
                               box_size,
                               tuple([self.grid_spacing]*3),
                               grid_data,
                               site_centers,
                               site_radii,
                               binding_site_centroid,
                               exterior_shell_grid_data
                               ))
            
            
    def run(self):
        
        self.get_position_input()
        self.set_grid_spacing() 
        self.get_delaunay()
        self.calculate_circumsphere()
        self.filter_circumspheres()
        self.first_pass_filter()
        self.second_pass_cluster()
        self.third_pass_cluster()
        self.get_binding_sites()
        return self.binding_site_container
    
    @classmethod 
    def from_data(cls, 
                  session,
                  current_model,
                  current_map = None):
        
        instance = cls(session, current_model,current_map)
        binding_site_container =  instance.run() 
        return binding_site_container
        

def filter_exterior_shell_by_protein_atoms(
    exterior_shell, 
    protein_positions, 
    protein_radii, 
    origin, 
    grid_spacing, 
    buffer=0.1
):
    """
    Remove voxels from the exterior shell that lie within an exclusion zone 
    around any protein atom.

    Parameters:
      exterior_shell (np.ndarray): 3D boolean array (the exterior shell mask).
      protein_positions (np.ndarray): (N, 3) array of protein atom coordinates (in Å).
      protein_radii (np.ndarray): (N,) array of protein atom van der Waals radii (in Å).
      origin (array-like): The real-space origin of the grid (in Å) given as [x0, y0, z0].
      grid_spacing (float): The spacing between grid points (in Å).
      buffer (float): A small extra distance to add to each atom’s radius (in Å).

    Returns:
      np.ndarray: The updated exterior_shell with voxels too close to any protein atom set to False.
    """
    # Obtain the voxel indices where exterior_shell is True.
    # Note: np.argwhere returns indices in (z, y, x) order.
    voxel_indices = np.argwhere(exterior_shell)  # shape (M, 3)

    # Reorder voxel_indices from (z, y, x) to (x, y, z)
    # so that the first column corresponds to x.
    voxel_indices_xyz = voxel_indices[:, [2, 1, 0]]

    # Convert voxel indices to real-space coordinates:
    # real_coord = origin + (voxel_index_xyz * grid_spacing)
    voxel_coords = origin + voxel_indices_xyz * grid_spacing  # shape (M, 3)

    # Build a KDTree from these voxel coordinates.
    voxel_tree = cKDTree(voxel_coords)

    # Set to keep track of indices (in voxel_coords) to remove.
    voxels_to_remove = set()

    # For each protein atom, remove voxels within the exclusion radius.
    for atom_pos, vdw in zip(protein_positions, protein_radii):
        exclusion_radius = vdw + buffer
        # Query the tree for all voxel indices within the exclusion sphere.
        close_voxel_indices = voxel_tree.query_ball_point(atom_pos, exclusion_radius)
        voxels_to_remove.update(close_voxel_indices)

    # Now update the original exterior_shell.
    # Note: voxel_indices contains the original (z, y, x) indices.
    for idx in voxels_to_remove:
        z, y, x = voxel_indices[idx]
        exterior_shell[z, y, x] = False

    return exterior_shell

def create_exterior_shell(binding_mask, dilation_radius=2):
    """
    Create a thin shell outside the binding site.
    
    Parameters:
        binding_mask (np.ndarray): The binary mask of the binding site.
        dilation_radius (int): Number of voxels for dilation (controls shell thickness).
    
    Returns:
        np.ndarray: A binary mask of the exterior shell.
    """
    # Use a spherical structuring element for 3D dilation
    struct_elem =  morphology.ball(dilation_radius)
    dilated_mask =  morphology.binary_dilation(binding_mask, struct_elem)
    exterior_shell = np.logical_and(dilated_mask, np.logical_not(binding_mask))
    return exterior_shell

def find_binding_site_boundary(binding_mask):
    eroded_mask = morphology.binary_erosion(binding_mask)
    boundary_mask = binding_mask ^ eroded_mask
    #print("Boundary mask extracted.")
    return boundary_mask

def compute_distance_map(boundary_mask, apix):
    inverted_boundary = ~boundary_mask
    distance_map = distance_transform_edt(inverted_boundary, sampling=apix)
    #print("Distance map computed.")
    return distance_map
       

def create_binding_site_mask(density_shape, point, radii, apix, origin):
    binding_mask = np.zeros(density_shape, dtype=bool)
    for center_real, radius in zip(point,radii):
        
        center_voxel = (center_real - origin) / np.array(apix) #xyz
        center_voxel = np.round(center_voxel).astype(int) #xyz
        #print(f"Sphere center (voxel indices): {center_voxel}, Radius (voxels): {radius / np.array(apix)}")
        
        z, y, x = np.ogrid[:density_shape[0], :density_shape[1], :density_shape[2]]
        real_z = z * apix[2] + origin[2]
        real_y = y * apix[1] + origin[1]
        real_x = x * apix[0] + origin[0]
        distance_squared = ((real_x - center_real[0]) ** 2 +
                            (real_y - center_real[1]) ** 2 +
                            (real_z - center_real[2]) ** 2)
        sphere_mask = distance_squared <= radius ** 2
        binding_mask |= sphere_mask
    #print("Binding site mask created.")
    return binding_mask
        
def generate_density_map(points, radii, grid_spacing=1.0):
    """
    Generate a 3D density map from points and radii and save it as an MRC file.

    Parameters:
        points (numpy.ndarray): An (N, 3) array of points [x, y, z].
        radii (numpy.ndarray): An array of radii for each point.
        grid_spacing (float): The spacing of the grid in Å. Default is 1.0 Å.
    """
    # Determine the bounding box of the points based on the max radius
    min_coords = np.min(points - radii[:, np.newaxis], axis=0)
    max_coords = np.max(points + radii[:, np.newaxis], axis=0)
    
    # Calculate grid dimensions
    grid_shape = np.ceil((max_coords - min_coords) / grid_spacing).astype(int) + 1
    density_grid = np.zeros(grid_shape, dtype=np.float32)

    # Define the grid origin (bottom-left corner)
    origin = min_coords
    
    # Iterate over each point and add spherical density to the grid
    
    for point, radius in zip(points, radii):
        
      
        # Calculate grid limits for the current sphere
        lower = np.floor((point - radius - origin) / grid_spacing).astype(int)
        upper = np.ceil((point + radius - origin) / grid_spacing).astype(int)

        # Ensure the indices are within the grid bounds
        lower = np.maximum(lower, 0)
        upper = np.minimum(upper, np.array(grid_shape))
        
        # Generate grid coordinates within the bounding box of the sphere
        for x in range(lower[0], upper[0]):
            for y in range(lower[1], upper[1]):
                for z in range(lower[2], upper[2]):
                    # Calculate the real position of the grid point
                    grid_point = origin + np.array([x, y, z]) * grid_spacing
                    distance = np.linalg.norm(grid_point - point)
                    if distance <= radius:
                        density_grid[x, y, z] += np.exp(-distance**2 / (2 * (radius / 2.0)**2))

    # Transpose the density grid to match MRC file format (if necessary)
    density_grid = density_grid.astype(np.float32)
    density_grid = np.transpose(density_grid, (2, 1, 0)).astype(np.float32)
    
    
    return origin, density_grid.shape, density_grid

        
def compute_circumsphere(vertices):
    A, B, C, D = vertices
    # Setup matrices for solving the circumcenter
    lhs = 2 * np.array([A - D, B - D, C - D])
    rhs = np.array([
        np.dot(A, A) - np.dot(D, D),
        np.dot(B, B) - np.dot(D, D),
        np.dot(C, C) - np.dot(D, D)
    ])
    # Solve for the circumcenter
    circumcenter = np.linalg.solve(lhs, rhs) #+ D
    # Calculate the radius
    #radius = np.linalg.norm(circumcenter - A)
    radius = distance.euclidean(circumcenter, A)
    return circumcenter, radius
        
        
        