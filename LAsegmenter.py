#!/usr/bin/env python3
"""
Left Atrium Segmenter - Version 906
Based on v17 with two fixes:
- MA region smaller: bounded by the 3 picked points, not extending beyond them
- Lateral wall: moved to between LAA/LIPV and MA (not left of LSPV/LIPV)
"""

import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import networkx as nx
import argparse
from collections import deque
import heapq
import pickle
import os
import time
import csv
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None

class LASegmenter:
    def __init__(self, vtk_file):
        self.vtk_file = vtk_file
        self.mesh = None
        self.points = None
        self.faces = None
        self.graph = None
        self.markers = {}
        self.centering_offset = None
        self.exterior_segmenter = None
        self.exterior_mesh = None
        self.default_ostium_radius = 10.0
        
        self.landmark_sequence = [
            ('MA', 'point1', 'MITRAL ANNULUS - MA1: First point on major axis'),
            ('MA', 'point2', 'MITRAL ANNULUS - MA2: Second point on major axis'),
            ('MA', 'point3', 'MITRAL ANNULUS - MA3: First point on minor axis'),
            ('MA', 'point4', 'MITRAL ANNULUS - MA4: Second point on minor axis'),
            ('RSPV', 'vein', 'RSPV - Click TIP, then position cutting plane'),
            ('LSPV', 'vein', 'LSPV - Click TIP, then position cutting plane'),
            ('RIPV', 'vein', 'RIPV - Click TIP, then position cutting plane'),
            ('LIPV', 'vein', 'LIPV - Click TIP, then position cutting plane'),
            ('LAA', 'vein', 'LAA - Click TIP, then position cutting plane'),
        ]
        
        self.colors = {
            'RSPV': (1, 0, 0), 'LSPV': (0, 0, 1), 'RIPV': (1, 0, 1),
            'LIPV': (0, 1, 0), 'MA': (0.5, 0.5, 0.5), 'LAA': (1, 1, 0)
        }
        
        self.extended_region_names = [
            'Background', 'RSPV', 'LSPV', 'RIPV', 'LIPV', 'MA', 'LAA',
            'Posterior_Wall', 'Roof', 'Inferior_Wall', 'Lateral_Wall',
            'Septal_Wall', 'Anterior_Wall', 'RSPV_Ostium', 'LSPV_Ostium',
            'RIPV_Ostium', 'LIPV_Ostium'
        ]
        
        self.extended_color_map = {
            0: (200, 200, 200), 1: (255, 0, 0), 2: (0, 0, 255),
            3: (255, 0, 255), 4: (0, 255, 0), 5: (80, 80, 80),
            6: (255, 255, 0), 7: (255, 128, 0), 8: (0, 255, 255),
            9: (128, 0, 255), 10: (255, 192, 203), 11: (0, 128, 128),
            12: (128, 255, 128), 13: (200, 50, 50), 14: (50, 50, 200),
            15: (200, 50, 200), 16: (50, 200, 50)
        }
        
    def load_mesh(self):
        print(f"\nLoading {self.vtk_file}...")
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(self.vtk_file)
        reader.Update()
        
        self.mesh = reader.GetOutput()
        self.points = vtk_to_numpy(self.mesh.GetPoints().GetData())
        
        polys = self.mesh.GetPolys()
        polys.InitTraversal()
        self.faces = []
        id_list = vtk.vtkIdList()
        while polys.GetNextCell(id_list):
            face = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
            if len(face) == 3:
                self.faces.append(face)
            elif len(face) == 4:
                self.faces.append([face[0], face[1], face[2]])
                self.faces.append([face[0], face[2], face[3]])
        self.faces = np.array(self.faces)
        print(f"✓ {len(self.points)} vertices, {len(self.faces)} faces")
        
    def center_mesh(self, offset=None):
        """Center the mesh. If offset is provided, use it instead of computing centroid.
        Returns the centering offset used (for applying same transform to other meshes).
        """
        if offset is None:
            offset = np.mean(self.points, axis=0)
        self.centering_offset = offset.copy()
        self.points -= offset
        vtk_points = vtk.vtkPoints()
        for p in self.points:
            vtk_points.InsertNextPoint(p)
        self.mesh.SetPoints(vtk_points)
        return offset
        
    def build_graph(self):
        print("Building graph...")
        self.graph = nx.Graph()
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1) % 3]
                if not self.graph.has_edge(v1, v2):
                    d = np.linalg.norm(self.points[v1] - self.points[v2])
                    self.graph.add_edge(v1, v2, weight=d)
        print(f"✓ {self.graph.number_of_nodes()} nodes")
    
    def compute_vertex_normal(self, vid):
        normals = []
        for face in self.faces:
            if vid in face:
                v0, v1, v2 = self.points[face[0]], self.points[face[1]], self.points[face[2]]
                n = np.cross(v1 - v0, v2 - v0)
                if np.linalg.norm(n) > 1e-6:
                    normals.append(n / np.linalg.norm(n))
        if normals:
            avg = np.mean(normals, axis=0)
            if np.linalg.norm(avg) > 1e-6:
                return avg / np.linalg.norm(avg)
        return np.array([0, 0, 1])
    
    def compute_all_vertex_normals(self):
        """Compute normals for all vertices at once using vectorized operations."""
        num_vertices = len(self.points)
        vertex_normals = np.zeros((num_vertices, 3))
        
        # Compute face normals
        v0 = self.points[self.faces[:, 0]]  # (F, 3)
        v1 = self.points[self.faces[:, 1]]  # (F, 3)
        v2 = self.points[self.faces[:, 2]]  # (F, 3)
        
        face_normals = np.cross(v1 - v0, v2 - v0)  # (F, 3)
        face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)  # (F, 1)
        face_norms = np.maximum(face_norms, 1e-10)  # Avoid division by zero
        face_normals = face_normals / face_norms  # (F, 3) normalized
        
        # Accumulate normals at each vertex
        for f_idx, face in enumerate(self.faces):
            for vid in face:
                vertex_normals[vid] += face_normals[f_idx]
        
        # Normalize vertex normals
        vertex_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_norms = np.maximum(vertex_norms, 1e-10)
        vertex_normals = vertex_normals / vertex_norms
        
        # Store as instance variable
        self.vertex_normals = vertex_normals
        
        return vertex_normals
    
    def estimate_vein_direction(self, tip_id):
        tip = self.points[tip_id]
        normal = self.compute_vertex_normal(tip_id)
        vein_dir = -normal
        
        dists = np.linalg.norm(self.points - tip, axis=1)
        nearby = (dists > 5) & (dists < 25)
        if np.sum(nearby) > 10:
            centroid = np.mean(self.points[nearby], axis=0)
            cdir = centroid - tip
            if np.linalg.norm(cdir) > 1e-6:
                cdir /= np.linalg.norm(cdir)
                vein_dir = 0.5 * vein_dir + 0.5 * cdir
                vein_dir /= np.linalg.norm(vein_dir)
        return vein_dir
    
    def find_connected_component(self, start_id, mask):
        """Find connected component containing start_id within mask"""
        if not mask[start_id]:
            return set()
        
        visited = set()
        queue = deque([start_id])
        
        while queue:
            vid = queue.popleft()
            if vid in visited:
                continue
            if not mask[vid]:
                continue
            
            visited.add(vid)
            
            for neighbor in self.graph.neighbors(vid):
                if neighbor not in visited and mask[neighbor]:
                    queue.append(neighbor)
        
        return visited
    
    def create_plane_actors(self, center, normal, radius, color):
        normal = np.array(normal) / np.linalg.norm(normal)
        
        circle = vtk.vtkRegularPolygonSource()
        circle.SetNumberOfSides(64)
        circle.SetRadius(radius)
        circle.SetCenter(0, 0, 0)
        circle.SetGeneratePolygon(False)
        
        transform = vtk.vtkTransform()
        transform.Translate(*center)
        
        default = np.array([0, 0, 1])
        if not np.allclose(normal, default) and not np.allclose(normal, -default):
            axis = np.cross(default, normal)
            axis /= np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(default, normal), -1, 1))
            transform.RotateWXYZ(np.degrees(angle), *axis)
        elif np.allclose(normal, -default):
            transform.RotateX(180)
        
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(circle.GetOutputPort())
        tf.SetTransform(transform)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputConnection(tf.GetOutputPort())
        tube.SetRadius(0.5)
        tube.SetNumberOfSides(12)
        
        ring_mapper = vtk.vtkPolyDataMapper()
        ring_mapper.SetInputConnection(tube.GetOutputPort())
        ring_actor = vtk.vtkActor()
        ring_actor.SetMapper(ring_mapper)
        ring_actor.GetProperty().SetColor(*color)
        
        disk = vtk.vtkDiskSource()
        disk.SetInnerRadius(0)
        disk.SetOuterRadius(radius)
        disk.SetCircumferentialResolution(64)
        
        dtf = vtk.vtkTransformPolyDataFilter()
        dtf.SetInputConnection(disk.GetOutputPort())
        dtf.SetTransform(transform)
        
        disk_mapper = vtk.vtkPolyDataMapper()
        disk_mapper.SetInputConnection(dtf.GetOutputPort())
        disk_actor = vtk.vtkActor()
        disk_actor.SetMapper(disk_mapper)
        disk_actor.GetProperty().SetColor(*color)
        disk_actor.GetProperty().SetOpacity(0.3)
        
        return ring_actor, disk_actor
    
    def create_posterior_wall_plane_actors(self, p1, p2, p3, plane_normal, color, plane_size=15):
        """Create visualization actors for a plane defined by 3 points"""
        # Normalize normal
        plane_normal = np.array(plane_normal) / (np.linalg.norm(plane_normal) + 1e-10)
        
        # Create two orthogonal vectors in the plane
        v1 = p2 - p1
        v1 = v1 / (np.linalg.norm(v1) + 1e-10)
        v2 = np.cross(plane_normal, v1)
        v2 = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # Create a rectangle in the plane
        plane_center = (p1 + p2 + p3) / 3.0
        
        # 4 corners of the plane rectangle
        corners = [
            plane_center - v1 * plane_size - v2 * plane_size,
            plane_center + v1 * plane_size - v2 * plane_size,
            plane_center + v1 * plane_size + v2 * plane_size,
            plane_center - v1 * plane_size + v2 * plane_size,
        ]
        
        # Create quad from corners
        quad = vtk.vtkQuad()
        points = vtk.vtkPoints()
        for i, corner in enumerate(corners):
            points.InsertNextPoint(*corner)
            quad.GetPointIds().SetId(i, i)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)
        polydata.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(0.2)
        
        # Add wireframe edges
        edges = vtk.vtkExtractEdges()
        edges.SetInputData(polydata)
        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(edges.GetOutputPort())
        edge_actor = vtk.vtkActor()
        edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetColor(*color)
        edge_actor.GetProperty().SetLineWidth(2)
        
        return actor, edge_actor
    

    
    def select_landmarks_interactive(self):
        print("\n" + "="*60)
        print("  LANDMARK SELECTION")
        print("="*60)
        print("\nVEINS: Click tip, then adjust plane:")
        print("  W/S: Tilt forward/back | A/D: Tilt left/right")
        print("  UP/DOWN: Move along normal | I/K: Move up/down | J/L: Move left/right")
        print("  +/-: Radius | R: Reset | SPACE: Confirm")
        print("\nMA: Click 3 points on rim\n")
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.9, 0.9, 0.9)
        actor.GetProperty().SetOpacity(0.85)
        
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.2)
        
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(1200, 900)
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        state = {
            'idx': 0, 'tip_id': None, 'tip_pos': None, 'base_normal': None,
            'plane_pos': None, 'plane_normal': None, 'radius': self.default_ostium_radius,
            'offset': 0, 'tilt_fb': 0, 'tilt_lr': 0,
            'marker': None, 'ring': None, 'disk': None, 'permanent': [],
            'ma_point_id': None, 'ma_coords': None, 'done': False,
            'regions': np.zeros(len(self.points), dtype=int), 'region_actor': None,
            'exit_reason': None,  # Track why we're exiting
            'review_mode': False   # Track if we're reviewing a landmark
        }
        
        text = vtk.vtkTextActor()
        text.GetTextProperty().SetFontSize(18)
        text.GetTextProperty().SetColor(1, 1, 0)
        text.SetPosition(10, 10)
        renderer.AddViewProp(text)
        
        info = vtk.vtkTextActor()
        info.GetTextProperty().SetFontSize(14)
        info.GetTextProperty().SetColor(0.5, 1, 0.5)
        info.SetPosition(10, 40)
        renderer.AddViewProp(info)
        
        def update_text():
            region, ltype, desc = self.landmark_sequence[state['idx']]
            text.SetInput(f"[{state['idx']+1}/{len(self.landmark_sequence)}] {desc}")
        
        def update_region_visualization():
            """Update the 3D visualization to show current regions"""
            if state['region_actor']:
                renderer.RemoveActor(state['region_actor'])
                state['region_actor'] = None
            
            # Create colored mesh based on regions
            region_colors = vtk.vtkUnsignedCharArray()
            region_colors.SetNumberOfComponents(3)
            region_colors.SetName("Colors")
            
            for vid in range(len(self.points)):
                rid = state['regions'][vid]
                color = self.extended_color_map.get(rid, (200, 200, 200))
                region_colors.InsertNextTuple3(int(color[0]), int(color[1]), int(color[2]))
            
            self.mesh.GetPointData().SetScalars(region_colors)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(self.mesh)
            mapper.SetScalarModeToUsePointData()
            
            region_actor = vtk.vtkActor()
            region_actor.SetMapper(mapper)
            renderer.RemoveActor(actor)  # Remove the original gray actor
            renderer.AddActor(region_actor)
            state['region_actor'] = region_actor
            window.Render()

        
        def update_info():
            if state['plane_pos'] is not None:
                info.SetInput(f"Offset: {state['offset']:.1f}mm | Radius: {state['radius']:.1f}mm | Tilt: FB={state['tilt_fb']:.0f}° LR={state['tilt_lr']:.0f}°")
            else:
                info.SetInput("")
        
        def compute_tilted_normal(cam_right=None, cam_up=None):
            """Compute tilted normal relative to viewport if camera vectors provided, else relative to vein"""
            base = state['base_normal'].copy()
            
            if cam_right is not None and cam_up is not None:
                # Viewport-relative tilting using camera vectors
                # tilt_lr: rotate around camera up axis (left-right tilt)
                # tilt_fb: rotate around camera right axis (forward-backward tilt)
                n = base.copy()
                
                if state['tilt_lr'] != 0:
                    angle = np.radians(state['tilt_lr'])
                    c, s = np.cos(angle), np.sin(angle)
                    # Rodrigues' rotation formula around cam_up
                    n = n * c + np.cross(cam_up, n) * s + cam_up * np.dot(cam_up, n) * (1 - c)
                
                if state['tilt_fb'] != 0:
                    angle = np.radians(state['tilt_fb'])
                    c, s = np.cos(angle), np.sin(angle)
                    # Rodrigues' rotation formula around cam_right
                    n = n * c + np.cross(cam_right, n) * s + cam_right * np.dot(cam_right, n) * (1 - c)
                
                return n / np.linalg.norm(n)
            else:
                # Vein-relative tilting (original behavior)
                if abs(base[2]) < 0.9:
                    right = np.cross(base, [0, 0, 1])
                else:
                    right = np.cross(base, [0, 1, 0])
                right /= np.linalg.norm(right)
                up = np.cross(right, base)
                up /= np.linalg.norm(up)
                
                n = base.copy()
                if state['tilt_fb'] != 0:
                    angle = np.radians(state['tilt_fb'])
                    c, s = np.cos(angle), np.sin(angle)
                    n = n * c + np.cross(right, n) * s + right * np.dot(right, n) * (1 - c)
                
                if state['tilt_lr'] != 0:
                    angle = np.radians(state['tilt_lr'])
                    c, s = np.cos(angle), np.sin(angle)
                    n = n * c + np.cross(up, n) * s + up * np.dot(up, n) * (1 - c)
                
                return n / np.linalg.norm(n)
        
        def update_plane(tilt_only=False, cam_right=None, cam_up=None):
            if state['plane_pos'] is None:
                return
            
            region = self.landmark_sequence[state['idx']][0]
            
            if state['ring']:
                renderer.RemoveActor(state['ring'])
            if state['disk']:
                renderer.RemoveActor(state['disk'])
            
            state['plane_normal'] = compute_tilted_normal(cam_right, cam_up)
            
            # If only tilting (not changing offset), keep disk center fixed
            # If changing offset, recalculate position along the new normal
            if not tilt_only:
                state['plane_pos'] = state['tip_pos'] + state['plane_normal'] * state['offset']
            
            state['ring'], state['disk'] = self.create_plane_actors(
                state['plane_pos'], state['plane_normal'], state['radius'], self.colors[region]
            )
            renderer.AddActor(state['ring'])
            renderer.AddActor(state['disk'])
            
            update_info()
            window.Render()
        
        def on_click(obj, event):
            # Don't process clicks during review mode
            if state['review_mode']:
                return
            
            pos = interactor.GetEventPosition()
            picker = vtk.vtkCellPicker()
            picker.SetTolerance(0.005)
            
            if not picker.Pick(pos[0], pos[1], 0, renderer):
                return
            
            picked = np.array(picker.GetPickPosition())
            dists = np.linalg.norm(self.points - picked, axis=1)
            vid = np.argmin(dists)
            vpos = self.points[vid].copy()
            
            region, ltype, desc = self.landmark_sequence[state['idx']]
            
            if ltype == 'vein':
                state['tip_id'] = vid
                state['tip_pos'] = vpos.copy()
                state['base_normal'] = self.estimate_vein_direction(vid)
                state['plane_normal'] = state['base_normal'].copy()
                state['plane_pos'] = vpos.copy()
                state['radius'] = self.default_ostium_radius
                state['offset'] = 0
                state['tilt_fb'] = 0
                state['tilt_lr'] = 0
                
                if state['marker']:
                    renderer.RemoveActor(state['marker'])
                
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(*vpos)
                sphere.SetRadius(1.5)
                sm = vtk.vtkPolyDataMapper()
                sm.SetInputConnection(sphere.GetOutputPort())
                state['marker'] = vtk.vtkActor()
                state['marker'].SetMapper(sm)
                state['marker'].GetProperty().SetColor(1, 1, 1)
                renderer.AddActor(state['marker'])
                
                # Get camera vectors to pass to update_plane
                camera = renderer.GetActiveCamera()
                cam_pos = np.array(camera.GetPosition())
                cam_focal = np.array(camera.GetFocalPoint())
                cam_view = cam_focal - cam_pos
                cam_view = cam_view / np.linalg.norm(cam_view)
                cam_up_raw = np.array(camera.GetViewUp())
                cam_up = cam_up_raw / np.linalg.norm(cam_up_raw)
                cam_right = np.cross(cam_view, cam_up)
                cam_right = cam_right / np.linalg.norm(cam_right)
                cam_up = np.cross(cam_right, cam_view)
                cam_up = cam_up / np.linalg.norm(cam_up)
                
                update_plane(cam_right=cam_right, cam_up=cam_up)
                text.SetInput(f"[{state['idx']+1}/{len(self.landmark_sequence)}] WSAD=tilt, IJKL=move, UP/DOWN=offset, SPACE=confirm")
                text.GetTextProperty().SetColor(0, 1, 0)
            else:
                state['ma_point_id'] = vid
                state['ma_coords'] = vpos.copy()
                
                if state['marker']:
                    renderer.RemoveActor(state['marker'])
                
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(*vpos)
                sphere.SetRadius(2)
                sm = vtk.vtkPolyDataMapper()
                sm.SetInputConnection(sphere.GetOutputPort())
                state['marker'] = vtk.vtkActor()
                state['marker'].SetMapper(sm)
                state['marker'].GetProperty().SetColor(*self.colors[region])
                renderer.AddActor(state['marker'])
                
                text.SetInput(f"[{state['idx']+1}/{len(self.landmark_sequence)}] SPACE to confirm")
                text.GetTextProperty().SetColor(0, 1, 0)
            
            window.Render()
        
        def on_key(obj, event):
            key = interactor.GetKeySym()
            region, ltype, desc = self.landmark_sequence[state['idx']]
            
            # Handle review mode key presses
            if state['review_mode']:
                if key == 'space':
                    # Accept the current landmark
                    state['review_mode'] = False
                    print(f"DEBUG: Review mode SPACE - idx={state['idx']}, tip_id={state['tip_id']}")
                    
                    # If reviewing MA region (entered after idx=3), move to RSPV selection
                    # MA review is entered with idx=4, which is RSPV in landmark_sequence
                    # After accepting MA, we stay at idx=4 to start RSPV selection
                    if state['idx'] == 4:
                        # Check if we were reviewing MA or selecting RSPV
                        # If we just came from MA acceptance, we'll be selecting RSPV (no state vars set yet)
                        # If we confirmed RSPV plane, we'll have state['tip_id'] set
                        if state['tip_id'] is None:
                            # Just accepted MA review, now ready for RSPV selection
                            update_text()
                            text.GetTextProperty().SetColor(1, 1, 0)
                            window.Render()
                        else:
                            # Just accepted RSPV review (manual), move to LSPV
                            state['idx'] += 1
                            state['tip_id'] = None
                            state['tip_pos'] = None
                            state['plane_pos'] = None
                            state['plane_normal'] = None
                            state['base_normal'] = None
                            state['radius'] = self.default_ostium_radius
                            state['offset'] = 0
                            state['tilt_fb'] = 0
                            state['tilt_lr'] = 0
                            if state['marker']:
                                renderer.RemoveActor(state['marker'])
                                state['marker'] = None
                            if state['ring']:
                                renderer.RemoveActor(state['ring'])
                                state['ring'] = None
                            if state['disk']:
                                renderer.RemoveActor(state['disk'])
                                state['disk'] = None
                            update_text()
                            text.GetTextProperty().SetColor(1, 1, 0)
                            window.Render()
                    # If reviewing a PV (idx 5-7: RSPV, LSPV, RIPV, LIPV), move to next
                    elif state['idx'] >= 5 and state['idx'] <= 7:
                        print(f"DEBUG: Accepting PV review - idx={state['idx']} -> {state['idx']+1}")
                        state['idx'] += 1
                        print(f"DEBUG: After increment, idx={state['idx']}")
                        # Reset PV state variables for next PV selection
                        state['tip_id'] = None
                        state['tip_pos'] = None
                        state['plane_pos'] = None
                        state['plane_normal'] = None
                        state['base_normal'] = None
                        state['radius'] = self.default_ostium_radius
                        state['offset'] = 0
                        state['tilt_fb'] = 0
                        state['tilt_lr'] = 0
                        if state['marker']:
                            renderer.RemoveActor(state['marker'])
                            state['marker'] = None
                        if state['ring']:
                            renderer.RemoveActor(state['ring'])
                            state['ring'] = None
                        if state['disk']:
                            renderer.RemoveActor(state['disk'])
                            state['disk'] = None
                        update_text()
                        text.GetTextProperty().SetColor(1, 1, 0)
                        window.Render()
                    # If reviewing LAA (idx=8), we're done
                    elif state['idx'] == 8:
                        text.SetInput("✓ ALL LANDMARKS SELECTED! Close window.")
                        text.GetTextProperty().SetColor(0, 1, 0)
                        window.Render()
                        print("\n✓ All landmarks selected!")
                        state['exit_reason'] = 'all_done'
                        window.Finalize()
                        interactor.TerminateApp()
                    return
                elif key == 'Escape':
                    # Undo the current landmark
                    if state['idx'] == 4:
                        # Check if we're undoing MA or RSPV (both at idx=4)
                        if state['tip_id'] is None:
                            # Undo MA - delete MA region and all MA markers
                            print("Undoing MA selection...")
                            # Reset MA region (region_id 5 for MA)
                            state['regions'][state['regions'] == 5] = 0
                            
                            # Remove MA markers from display
                            for permanent_actor in state['permanent']:
                                renderer.RemoveActor(permanent_actor)
                            state['permanent'] = []
                            
                            # Reset to idx 0 to re-select MA points
                            state['idx'] = 0
                            state['ma_point_id'] = None
                            state['ma_coords'] = None
                            if state['marker']:
                                renderer.RemoveActor(state['marker'])
                                state['marker'] = None
                            
                            state['review_mode'] = False
                            update_text()
                            text.GetTextProperty().SetColor(1, 1, 0)
                            update_region_visualization()
                            window.Render()
                        else:
                            # Undo RSPV
                            region = 'RSPV'
                            print(f"Undoing {region} selection...")
                            
                            # Reset PV region (RSPV = 1)
                            state['regions'][state['regions'] == 1] = 0
                            
                            # Remove ostium region (RSPV_Ostium = 13)
                            state['regions'][state['regions'] == 13] = 0
                            
                            # Remove the marker and ring from permanent list if they're there
                            if state['permanent']:
                                if len(state['permanent']) >= 2:
                                    # Remove last 2: ring and marker
                                    removed_ring = state['permanent'].pop()
                                    removed_marker = state['permanent'].pop()
                                    renderer.RemoveActor(removed_ring)
                                    renderer.RemoveActor(removed_marker)
                                elif len(state['permanent']) >= 1:
                                    # Remove last 1: marker
                                    removed_marker = state['permanent'].pop()
                                    renderer.RemoveActor(removed_marker)
                            
                            # Remove ring and disk if they exist
                            if state['ring']:
                                renderer.RemoveActor(state['ring'])
                                state['ring'] = None
                            if state['disk']:
                                renderer.RemoveActor(state['disk'])
                                state['disk'] = None
                            
                            # Reset PV state variables for manual selection
                            state['tip_id'] = None
                            state['tip_pos'] = None
                            state['plane_pos'] = None
                            state['plane_normal'] = None
                            state['base_normal'] = None
                            state['radius'] = self.default_ostium_radius
                            state['offset'] = 0
                            state['tilt_fb'] = 0
                            state['tilt_lr'] = 0
                            
                            if state['marker']:
                                renderer.RemoveActor(state['marker'])
                                state['marker'] = None
                            
                            # Exit review mode to allow manual placement
                            state['review_mode'] = False
                            update_text()
                            text.GetTextProperty().SetColor(1, 1, 0)
                            update_region_visualization()
                            window.Render()
                    
                            
                            # Reset PV state variables (stay at idx=4 to re-select RSPV)
                            state['tip_id'] = None
                            state['tip_pos'] = None
                            state['plane_pos'] = None
                            state['plane_normal'] = None
                            state['base_normal'] = None
                            state['radius'] = self.default_ostium_radius
                            state['offset'] = 0
                            state['tilt_fb'] = 0
                            state['tilt_lr'] = 0
                            
                            if state['marker']:
                                renderer.RemoveActor(state['marker'])
                                state['marker'] = None
                            if state['ring']:
                                renderer.RemoveActor(state['ring'])
                                state['ring'] = None
                            if state['disk']:
                                renderer.RemoveActor(state['disk'])
                                state['disk'] = None
                            
                            # Stay at idx=4 to re-select RSPV
                            state['review_mode'] = False
                            update_text()
                            text.GetTextProperty().SetColor(1, 1, 0)
                            update_region_visualization()
                            window.Render()
                    elif state['idx'] >= 5 and state['idx'] <= 9:
                        # Undo PV - delete PV region and markers
                        region = self.landmark_sequence[state['idx']][0]
                        print(f"Undoing {region} selection...")
                        
                        # Reset PV region
                        pv_region_id = {'RSPV': 1, 'LSPV': 2, 'RIPV': 3, 'LIPV': 4, 'LAA': 6}[region]
                        state['regions'][state['regions'] == pv_region_id] = 0
                        
                        # Remove ostium region (LAA has no ostium)
                        if region != 'LAA':
                            ostium_region_id = {'RSPV': 13, 'LSPV': 14, 'RIPV': 15, 'LIPV': 16}[region]
                            if ostium_region_id in state['regions']:
                                state['regions'][state['regions'] == ostium_region_id] = 0
                        
                        # Remove the marker and ring from permanent list (last 2 actors added for this PV)
                        # Note: disk was removed from renderer immediately and not added to permanent
                        to_remove_count = 0
                        if state['permanent']:
                            # Check if last actor is the ring (was it added?)
                            if len(state['permanent']) >= 2:
                                # Remove last 2: ring and marker
                                removed_ring = state['permanent'].pop()
                                removed_marker = state['permanent'].pop()
                                renderer.RemoveActor(removed_ring)
                                renderer.RemoveActor(removed_marker)
                                to_remove_count = 2
                            elif len(state['permanent']) >= 1:
                                # Remove last 1: marker
                                removed_marker = state['permanent'].pop()
                                renderer.RemoveActor(removed_marker)
                                to_remove_count = 1
                        
                        # Remove ring and disk if they exist in current state
                        if state['ring']:
                            renderer.RemoveActor(state['ring'])
                            state['ring'] = None
                        if state['disk']:
                            renderer.RemoveActor(state['disk'])
                            state['disk'] = None
                        
                        # Reset PV state variables
                        state['tip_id'] = None
                        state['tip_pos'] = None
                        state['plane_pos'] = None
                        state['plane_normal'] = None
                        state['base_normal'] = None
                        state['radius'] = self.default_ostium_radius
                        state['offset'] = 0
                        state['tilt_fb'] = 0
                        state['tilt_lr'] = 0
                        
                        if state['marker']:
                            renderer.RemoveActor(state['marker'])
                            state['marker'] = None
                        
                        # Stay at same idx to re-select same PV (don't decrement)
                        state['review_mode'] = False
                        update_text()
                        text.GetTextProperty().SetColor(1, 1, 0)
                        update_region_visualization()
                        window.Render()
                    return
                else:
                    return  # Ignore other keys in review mode
            
            if ltype == 'vein' and state['plane_pos'] is not None:
                # Get camera vectors for viewport-relative controls
                camera = renderer.GetActiveCamera()
                cam_pos = np.array(camera.GetPosition())
                cam_focal = np.array(camera.GetFocalPoint())
                cam_view = cam_focal - cam_pos
                cam_view = cam_view / np.linalg.norm(cam_view)
                cam_up_raw = np.array(camera.GetViewUp())
                cam_up = cam_up_raw / np.linalg.norm(cam_up_raw)
                cam_right = np.cross(cam_view, cam_up)
                cam_right = cam_right / np.linalg.norm(cam_right)
                cam_up = np.cross(cam_right, cam_view)  # Recompute to ensure orthogonal
                cam_up = cam_up / np.linalg.norm(cam_up)
                
                def update_marker_display():
                    """Update tip marker visualization"""
                    if state['marker']:
                        renderer.RemoveActor(state['marker'])
                    sphere = vtk.vtkSphereSource()
                    sphere.SetCenter(*state['tip_pos'])
                    sphere.SetRadius(1.5)
                    sm = vtk.vtkPolyDataMapper()
                    sm.SetInputConnection(sphere.GetOutputPort())
                    marker = vtk.vtkActor()
                    marker.SetMapper(sm)
                    marker.GetProperty().SetColor(1, 1, 1)
                    renderer.AddActor(marker)
                    return marker
                
                if key == 'w':
                    state['tilt_fb'] += 5
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    state['tip_pos'] = state['plane_pos'] - state['plane_normal'] * state['offset']
                    state['marker'] = update_marker_display()
                    window.Render()
                    return
                elif key == 's':
                    state['tilt_fb'] -= 5
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    state['tip_pos'] = state['plane_pos'] - state['plane_normal'] * state['offset']
                    state['marker'] = update_marker_display()
                    window.Render()
                    return
                elif key == 'a':
                    state['tilt_lr'] -= 5
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    state['tip_pos'] = state['plane_pos'] - state['plane_normal'] * state['offset']
                    state['marker'] = update_marker_display()
                    window.Render()
                    return
                elif key == 'd':
                    state['tilt_lr'] += 5
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    state['tip_pos'] = state['plane_pos'] - state['plane_normal'] * state['offset']
                    state['marker'] = update_marker_display()
                    window.Render()
                    return
                elif key == 'Up':
                    state['offset'] += 2
                    # Adjust offset only - do NOT recompute normal, just move plane along existing normal
                    state['plane_pos'] = state['tip_pos'] + state['plane_normal'] * state['offset']
                    # Recreate actors with updated plane_pos but same normal
                    region = self.landmark_sequence[state['idx']][0]
                    if state['ring']:
                        renderer.RemoveActor(state['ring'])
                    if state['disk']:
                        renderer.RemoveActor(state['disk'])
                    state['ring'], state['disk'] = self.create_plane_actors(
                        state['plane_pos'], state['plane_normal'], state['radius'], self.colors[region]
                    )
                    renderer.AddActor(state['ring'])
                    renderer.AddActor(state['disk'])
                    update_info()
                    window.Render()
                    return
                elif key == 'Down':
                    state['offset'] = max(0, state['offset'] - 2)
                    # Adjust offset only - do NOT recompute normal, just move plane along existing normal
                    state['plane_pos'] = state['tip_pos'] + state['plane_normal'] * state['offset']
                    # Recreate actors with updated plane_pos but same normal
                    region = self.landmark_sequence[state['idx']][0]
                    if state['ring']:
                        renderer.RemoveActor(state['ring'])
                    if state['disk']:
                        renderer.RemoveActor(state['disk'])
                    state['ring'], state['disk'] = self.create_plane_actors(
                        state['plane_pos'], state['plane_normal'], state['radius'], self.colors[region]
                    )
                    renderer.AddActor(state['ring'])
                    renderer.AddActor(state['disk'])
                    update_info()
                    window.Render()
                    return
                elif key in ['plus', 'equal']:
                    state['radius'] += 1
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    return
                elif key == 'minus':
                    state['radius'] = max(3, state['radius'] - 1)
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    return
                elif key == 'r':
                    state['offset'] = 0
                    state['tilt_fb'] = 0
                    state['tilt_lr'] = 0
                    state['radius'] = self.default_ostium_radius
                    update_plane(tilt_only=False, cam_right=cam_right, cam_up=cam_up)
                    return
                elif key in ['i', 'j', 'k', 'l']:
                    # Move disk center in viewport coordinates
                    # I: up, K: down, J: left, L: right
                    move_distance = 1.0  # mm per keystroke
                    
                    move_vector = np.array([0.0, 0.0, 0.0])
                    if key == 'i':
                        move_vector = cam_up * move_distance
                    elif key == 'k':
                        move_vector = -cam_up * move_distance
                    elif key == 'j':
                        move_vector = -cam_right * move_distance
                    elif key == 'l':
                        move_vector = cam_right * move_distance
                    
                    state['plane_pos'] += move_vector
                    state['tip_pos'] += move_vector
                    
                    update_plane(tilt_only=True, cam_right=cam_right, cam_up=cam_up)
                    # Update tip marker visualization
                    if state['marker']:
                        renderer.RemoveActor(state['marker'])
                    sphere = vtk.vtkSphereSource()
                    sphere.SetCenter(*state['tip_pos'])
                    sphere.SetRadius(1.5)
                    sm = vtk.vtkPolyDataMapper()
                    sm.SetInputConnection(sphere.GetOutputPort())
                    state['marker'] = vtk.vtkActor()
                    state['marker'].SetMapper(sm)
                    state['marker'].GetProperty().SetColor(1, 1, 1)
                    renderer.AddActor(state['marker'])
                    window.Render()
                    return
            
            if key == 'space':
                if ltype == 'vein' and state['plane_pos'] is not None:
                    self.markers[f"{region}_distal"] = {
                        'point_id': state['tip_id'],
                        'coords': state['tip_pos'].copy(),
                    }
                    self.markers[f"{region}_ostium"] = {
                        'coords': state['plane_pos'].copy(),
                        'normal': state['plane_normal'].copy(),
                        'radius': state['radius'],
                        'point_id': None,
                    }
                    print(f"  ✓ {region}: offset={state['offset']:.1f}mm, r={state['radius']:.1f}mm")
                    
                    if state['marker']:
                        state['marker'].GetProperty().SetOpacity(0.5)
                        state['permanent'].append(state['marker'])
                        state['marker'] = None
                    if state['ring']:
                        state['ring'].GetProperty().SetOpacity(0.5)
                        state['permanent'].append(state['ring'])
                        state['ring'] = None
                    if state['disk']:
                        renderer.RemoveActor(state['disk'])
                        state['disk'] = None
                    
                    # NOTE: Keep state['tip_id'] set so on_key handler can distinguish PV review from MA review
                    # state['tip_id'] will be cleared when accepting PV review in on_key handler
                    state['plane_pos'] = None
                    
                    # After confirming a PV, enter review mode in the same window
                    if region in ['RSPV', 'LSPV', 'RIPV', 'LIPV', 'LAA']:
                        print(f"\nCreating {region} region and ostium...")
                        self.create_pv_regions(state['regions'], region)
                        update_region_visualization()
                        
                        # Enter review mode - just change key bindings
                        state['review_mode'] = True
                        text.SetInput(f"{region} REVIEW: SPACE=accept, ESC=undo")
                        text.GetTextProperty().SetColor(0, 1, 1)
                        window.Render()
                        return
                    
                elif ltype != 'vein' and state['ma_coords'] is not None:
                    self.markers[f"{region}_{ltype}"] = {
                        'point_id': state['ma_point_id'],
                        'coords': state['ma_coords'].copy(),
                    }
                    print(f"  ✓ {region}_{ltype}")
                    
                    if state['marker']:
                        state['marker'].GetProperty().SetOpacity(0.6)
                        state['permanent'].append(state['marker'])
                        state['marker'] = None
                    
                    state['ma_point_id'] = None
                    state['ma_coords'] = None
                    
                    # Increment idx only for MA points, not PVs
                    # PV idx increment happens when accepting review
                    state['idx'] += 1
                else:
                    return
                
                info.SetInput("")
                
                # After confirming 4th MA point, enter review mode in same window
                if state['idx'] == 4:
                    print("\nComputing MA region...")
                    self.create_ma_region(state['regions'])
                    update_region_visualization()
                    
                    # Enter MA review mode - just change key bindings
                    state['review_mode'] = True
                    text.SetInput("MA REVIEW: SPACE=accept, ESC=undo")
                    text.GetTextProperty().SetColor(0, 1, 1)
                    window.Render()
                    return
                
                if state['idx'] < len(self.landmark_sequence):
                    update_text()
                    text.GetTextProperty().SetColor(1, 1, 0)
                    # Reset camera only when transitioning TO MA (not between MA points)
                    if self.landmark_sequence[state['idx']][0] == 'MA' and (state['idx'] == 0 or self.landmark_sequence[state['idx']-1][0] != 'MA'):
                        renderer.GetActiveCamera().SetPosition(0, 100, 50)
                        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
                    window.Render()
                else:
                    text.SetInput("✓ ALL LANDMARKS SELECTED! Close window.")
                    text.GetTextProperty().SetColor(0, 1, 0)
                    window.Render()
                    print("\n✓ All landmarks selected!")
                    state['exit_reason'] = 'all_done'
                    window.Finalize()
                    interactor.TerminateApp()
                    return
        
        update_text()
        interactor.AddObserver('LeftButtonPressEvent', on_click)
        interactor.AddObserver('KeyPressEvent', on_key)
        
        # Close window to terminate program
        def on_window_close(obj, event):
            print("\n✗ Window closed. Terminating program.")
            state['exit_reason'] = 'window_closed'
            window.Finalize()
            interactor.TerminateApp()
        
        interactor.AddObserver('WinCloseEvent', on_window_close)
        
        window.SetWindowName("LA Segmenter - Landmarks")
        interactor.Initialize()
        window.Render()
        interactor.Start()
        
        # If window was closed, ensure we return window_closed regardless of other state
        if state['exit_reason'] is None:
            # Window closed without going through normal exit paths
            return state['regions'], 'window_closed'
        
        # Check why we exited
        if state['exit_reason'] == 'window_closed':
            return state['regions'], 'window_closed'
        elif state['exit_reason'] == 'MA':
            return state['regions'], 'MA'
        elif state['exit_reason'] == 'all_done':
            return state['regions'], None
        elif state['exit_reason'] is not None:
            # A PV was selected
            return state['regions'], state['exit_reason']
        else:
            return state['regions'], None
    
    def compute_geodesic_distance_from_set(self, source_verts):
        """Compute geodesic distance from a set of source vertices using multi-source Dijkstra"""
        dist = np.full(len(self.points), np.inf)
        
        heap = []
        for sv in source_verts:
            dist[sv] = 0
            heapq.heappush(heap, (0, sv))
        
        visited = set()
        while heap:
            d, vid = heapq.heappop(heap)
            if vid in visited:
                continue
            visited.add(vid)
            
            for neighbor in self.graph.neighbors(vid):
                if neighbor in visited:
                    continue
                edge_len = self.graph[vid][neighbor]['weight']
                new_dist = d + edge_len
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(heap, (new_dist, neighbor))
        
        return dist
    
    def get_pv_border_vertices(self, regions, pv_rid):
        """Get border vertices of a PV region"""
        pv_verts = set(np.where(regions == pv_rid)[0])
        border_verts = set()
        
        for vid in pv_verts:
            for neighbor in self.graph.neighbors(vid):
                if neighbor not in pv_verts:
                    border_verts.add(vid)
                    break
        
        return border_verts
    
    def signed_distance_to_plane(self, points, plane_point, plane_normal):
        """Calculate signed distance from points to a plane"""
        return np.dot(points - plane_point, plane_normal)
    
    def find_connected_components_for_region(self, regions, region_id):
        """Find connected components using BFS on mesh faces"""
        mask = regions == region_id
        vertex_indices = set(np.where(mask)[0])
        
        if len(vertex_indices) == 0:
            return []
        
        adjacency = {v: set() for v in vertex_indices}
        
        for face in self.faces:
            face_verts_in_region = [v for v in face if v in vertex_indices]
            if len(face_verts_in_region) >= 2:
                for i, v1 in enumerate(face_verts_in_region):
                    for v2 in face_verts_in_region[i+1:]:
                        adjacency[v1].add(v2)
                        adjacency[v2].add(v1)
        
        visited = set()
        components = []
        
        for start_vertex in vertex_indices:
            if start_vertex in visited:
                continue
            
            component = set()
            queue = deque([start_vertex])
            
            while queue:
                v = queue.popleft()
                if v in visited:
                    continue
                visited.add(v)
                component.add(v)
                
                for neighbor in adjacency[v]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            if component:
                components.append(component)
        
        return components
    
    def keep_largest_connected_component(self, regions, region_id):
        """Keep only the largest connected component"""
        components = self.find_connected_components_for_region(regions, region_id)
        
        if len(components) <= 1:
            return 0
        
        components.sort(key=len, reverse=True)
        largest = components[0]
        
        removed = 0
        for comp in components[1:]:
            for idx in comp:
                regions[idx] = 0
                removed += 1
        
        return removed
    
    def enforce_continuity(self, regions):
        """Ensure each region is continuous"""
        print("\n  Enforcing region continuity...")
        
        total_removed = 0
        
        for pass_num in range(3):
            pass_removed = 0
            
            for rid in range(1, 17):
                count = np.sum(regions == rid)
                if count > 0:
                    removed = self.keep_largest_connected_component(regions, rid)
                    if removed > 0:
                        print(f"    Region {rid} ({self.extended_region_names[rid]}): removed {removed}")
                        pass_removed += removed
            
            orphaned = np.where(regions == 0)[0]
            if len(orphaned) > 0:
                print(f"    Reassigning {len(orphaned)} orphaned vertices...")
                assigned = np.where(regions > 0)[0]
                
                if len(assigned) > 0:
                    for idx in orphaned:
                        dists = np.linalg.norm(self.points[assigned] - self.points[idx], axis=1)
                        nearest = assigned[np.argmin(dists)]
                        regions[idx] = regions[nearest]
            
            total_removed += pass_removed
            
            if pass_removed == 0:
                break
        
        print(f"    Total: {total_removed} vertices reassigned")
    
    def smooth_boundaries(self, regions, iterations=3):
        """Smooth region boundaries by majority voting"""
        print(f"\n  Smoothing boundaries ({iterations} iterations)...")
        
        for it in range(iterations):
            new_regions = regions.copy()
            changed = 0
            
            for i in range(len(self.points)):
                neighbors = list(self.graph.neighbors(i))
                if len(neighbors) < 2:
                    continue
                
                neighbor_regions = [regions[n] for n in neighbors]
                unique_neighbors = set(neighbor_regions)
                if len(unique_neighbors) == 1:
                    continue
                
                from collections import Counter
                counts = Counter(neighbor_regions)
                most_common = counts.most_common(1)[0][0]
                
                if counts[most_common] / len(neighbors) > 0.7:
                    if new_regions[i] != most_common:
                        new_regions[i] = most_common
                        changed += 1
            
            regions[:] = new_regions
            
            if changed == 0:
                break
    
    def create_ma_region(self, regions):
        """Create MA region from MA points as a 3D ellipsoid.
        
        The ellipsoid is defined by:
        - Axis 1: Major axis from the 4 points (length = distance from center to p2)
        - Axis 2: Minor axis from the 4 points (length = distance from center to p4)
        - Axis 3: Perpendicular to the 4-point plane
                  (length = (major_axis_length + minor_axis_length)/4.0)
        """
        print("\n1. Creating MA region (3D Ellipsoid)...")
        
        ma_p1 = self.markers['MA_point1']['coords']
        ma_p2 = self.markers['MA_point2']['coords']
        ma_p3 = self.markers['MA_point3']['coords']
        ma_p4 = self.markers['MA_point4']['coords']
        
        # All 4 points define the plane
        v1 = ma_p2 - ma_p1
        v2 = ma_p3 - ma_p1
        ma_plane_norm = np.cross(v1, v2)
        if np.linalg.norm(ma_plane_norm) < 1e-6:
            v2 = ma_p4 - ma_p1
            ma_plane_norm = np.cross(v1, v2)
        
        if np.linalg.norm(ma_plane_norm) < 1e-6:
            ma_plane_norm = np.array([0, 0, 1])
        else:
            ma_plane_norm = ma_plane_norm / np.linalg.norm(ma_plane_norm)
        
        # Compute ellipse center
        d1 = ma_p2 - ma_p1
        d2 = ma_p4 - ma_p3
        A = np.column_stack([d1, -d2])
        b = ma_p3 - ma_p1
        try:
            params, _ = np.linalg.lstsq(A, b, rcond=None)
            t = params[0]
            ellipse_center = ma_p1 + t * d1
        except:
            ellipse_center = (ma_p1 + ma_p2 + ma_p3 + ma_p4) / 4.0
        
        # Project diagonals onto plane and compute axes
        diag1 = ma_p2 - ma_p1
        diag2 = ma_p4 - ma_p3
        diag1_in_plane = diag1 - np.dot(diag1, ma_plane_norm) * ma_plane_norm
        diag2_in_plane = diag2 - np.dot(diag2, ma_plane_norm) * ma_plane_norm
        
        if np.linalg.norm(diag1_in_plane) > 1e-6:
            axis1 = diag1_in_plane / np.linalg.norm(diag1_in_plane)
        else:
            axis1 = d1 / np.linalg.norm(d1) if np.linalg.norm(d1) > 1e-6 else np.array([1, 0, 0])
        
        # Orthogonalize axis2 with respect to axis1 (Gram-Schmidt)
        if np.linalg.norm(diag2_in_plane) > 1e-6:
            axis2_candidate = diag2_in_plane / np.linalg.norm(diag2_in_plane)
        else:
            axis2_candidate = d2 / np.linalg.norm(d2) if np.linalg.norm(d2) > 1e-6 else np.array([0, 1, 0])
        
        # Remove component along axis1 to ensure orthogonality
        axis2 = axis2_candidate - np.dot(axis2_candidate, axis1) * axis1
        if np.linalg.norm(axis2) > 1e-6:
            axis2 = axis2 / np.linalg.norm(axis2)
        else:
            # Fallback: create perpendicular vector in the plane
            axis2 = np.cross(ma_plane_norm, axis1)
            if np.linalg.norm(axis2) > 1e-6:
                axis2 = axis2 / np.linalg.norm(axis2)
            else:
                axis2 = np.array([0, 1, 0])
        
        # The third axis is perpendicular to the plane
        axis3 = ma_plane_norm
        
        # Semi-axes lengths
        semi_axis1 = np.linalg.norm(ma_p2 - ellipse_center)
        semi_axis2 = np.linalg.norm(ma_p4 - ellipse_center)
        
        # Third axis length = 2 * (major_axis_length + minor_axis_length)
        major_axis_length = 2 * semi_axis1
        minor_axis_length = 2 * semi_axis2
        #semi_axis3 = (major_axis_length + minor_axis_length)/4.0
        semi_axis3 = 10.0  # Fixed thickness for MA region
        
        # Add 2mm margin to all axes
        margin = 2.0
        semi_axis1 += margin
        semi_axis2 += margin
                
        print(f"  MA ellipsoid parameters (with 1mm margin):")
        print(f"    Center: ({ellipse_center[0]:.2f}, {ellipse_center[1]:.2f}, {ellipse_center[2]:.2f})")
        print(f"    Semi-axis1 (major, in-plane): {semi_axis1:.2f} mm (full length: {2*semi_axis1:.2f} mm)")
        print(f"    Semi-axis2 (minor, in-plane): {semi_axis2:.2f} mm (full length: {2*semi_axis2:.2f} mm)")
        print(f"    Semi-axis3 (perpendicular): {semi_axis3:.2f} mm (full length: {2*semi_axis3:.2f} mm)")
        
        # Assign MA region using 3D ellipsoid equation
        for i, pt in enumerate(self.points):
            if regions[i] != 0:
                continue
            v = pt - ellipse_center
            
            # Compute coordinates along the three axes
            comp1 = np.dot(v, axis1)
            comp2 = np.dot(v, axis2)
            comp3 = np.dot(v, axis3)
            
            # Check 3D ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1
            if semi_axis1 > 1e-6 and semi_axis2 > 1e-6 and semi_axis3 > 1e-6:
                ellipsoid_param = (comp1 / semi_axis1) ** 2 + (comp2 / semi_axis2) ** 2 + (comp3 / semi_axis3) ** 2
                if ellipsoid_param <= 1.0:
                    regions[i] = 5
                elif ellipsoid_param <= 1.05:  # Slight tolerance for boundary vertices
                    regions[i] = 5
        
        print(f"  MA: {np.sum(regions == 5)}")
        return ellipse_center
    
    
    def create_pv_regions(self, regions, pv_name):
        """Create PV region and ostium ring for a single PV."""
        pv_rid_map = {'RSPV': 1, 'LSPV': 2, 'RIPV': 3, 'LIPV': 4, 'LAA': 6}
        ost_rid_map = {'RSPV': 13, 'LSPV': 14, 'RIPV': 15, 'LIPV': 16}
        
        pv_rid = pv_rid_map[pv_name]
        
        # Create PV region
        ost = self.markers[f'{pv_name}_ostium']
        center, normal, radius = ost['coords'], ost['normal'], ost['radius']
        tip_id = self.markers[f'{pv_name}_distal']['point_id']
        distal_pt = self.points[tip_id]
        distal_side = np.sign(np.dot(distal_pt - center, normal))
        
        candidates = np.zeros(len(self.points), dtype=bool)
        for i, pt in enumerate(self.points):
            if np.sign(np.dot(pt - center, normal)) == distal_side:
                candidates[i] = True
        
        connected = self.find_connected_component(tip_id, candidates)
        for vid in connected:
            if regions[vid] == 0:
                regions[vid] = pv_rid
        
        print(f"  {pv_name}: {len(connected)}")
        
        # Create ostium ring (if not LAA)
        if pv_name != 'LAA':
            ost_rid = ost_rid_map[pv_name]
            border_verts = self.get_pv_border_vertices(regions, pv_rid)
            
            if len(border_verts) > 0:
                dist_from_border = self.compute_geodesic_distance_from_set(border_verts)
                candidate_ostium = set()
                for i in range(len(self.points)):
                    if regions[i] != 0:
                        continue
                    if dist_from_border[i] <= 5.0:
                        candidate_ostium.add(i)
                
                if candidate_ostium:
                    adjacency = {v: set() for v in candidate_ostium}
                    for v in candidate_ostium:
                        for neighbor in self.graph.neighbors(v):
                            if neighbor in candidate_ostium:
                                adjacency[v].add(neighbor)
                            elif neighbor in border_verts:
                                adjacency[v].add(-1)
                    
                    start_verts = [v for v in candidate_ostium if -1 in adjacency[v]]
                    if start_verts:
                        visited = set()
                        queue = deque(start_verts)
                        while queue:
                            v = queue.popleft()
                            if v in visited or v == -1:
                                continue
                            visited.add(v)
                            for neighbor in adjacency[v]:
                                if neighbor not in visited and neighbor != -1:
                                    queue.append(neighbor)
                        
                        for v in visited:
                            regions[v] = ost_rid
                        
                        print(f"  {pv_name}_Ostium: {len(visited)}")
    
    def compute_wall_geometry(self, regions, ellipse_center):
        """Precompute all wall geometry parameters. Returns a dict with planes and axes."""
        # Get ostia centers
        ostia_centers = {}
        for pv in ['RSPV', 'LSPV', 'RIPV', 'LIPV']:
            pv_rid = {'RSPV': 1, 'LSPV': 2, 'RIPV': 3, 'LIPV': 4}[pv]
            border_verts = self.get_pv_border_vertices(regions, pv_rid)
            if len(border_verts) > 0:
                border_positions = np.array([self.points[v] for v in border_verts])
                ostia_centers[pv] = np.mean(border_positions, axis=0)
            else:
                ostia_centers[pv] = self.markers[f'{pv}_ostium']['coords']
        
        # Compute coordinate system
        rspv_ost_center = ostia_centers['RSPV']
        lspv_ost_center = ostia_centers['LSPV']
        ripv_ost_center = ostia_centers['RIPV']
        lipv_ost_center = ostia_centers['LIPV']
        
        pv_center = (rspv_ost_center + lspv_ost_center + ripv_ost_center + lipv_ost_center) / 4.0
        sup_center = (rspv_ost_center + lspv_ost_center) / 2.0
        inf_center = (ripv_ost_center + lipv_ost_center) / 2.0
        si_axis = sup_center - inf_center
        si_axis = si_axis / np.linalg.norm(si_axis)
        
        left_pv_center = (lspv_ost_center + lipv_ost_center) / 2.0
        right_pv_center = (rspv_ost_center + ripv_ost_center) / 2.0
        lr_axis = right_pv_center - left_pv_center
        lr_axis = lr_axis / np.linalg.norm(lr_axis)
        
        ap_axis = np.cross(lr_axis, si_axis)
        ap_axis = ap_axis / np.linalg.norm(ap_axis)
        
        pv_ap_component = np.dot(pv_center, ap_axis)
        ellipse_ap_component = np.dot(ellipse_center, ap_axis)
        
        heart_center_ap = (pv_ap_component*0.7 + ellipse_ap_component*0.3) 
        heart_center = pv_center + (heart_center_ap - pv_ap_component) * ap_axis
        
        pv_quadrangle_center = (rspv_ost_center + lspv_ost_center + ripv_ost_center + lipv_ost_center) / 4.0
        
        def make_plane(p1, p2, p3):
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            to_center = pv_quadrangle_center - p1
            if np.dot(normal, to_center) < 0:
                normal = -normal
            return p1, normal
        
        post_top_plane_pt, post_top_plane_normal = make_plane(rspv_ost_center, lspv_ost_center, heart_center)
        post_bottom_plane_pt, post_bottom_plane_normal = make_plane(ripv_ost_center, lipv_ost_center, heart_center)
        post_left_plane_pt, post_left_plane_normal = make_plane(lspv_ost_center, lipv_ost_center, heart_center)
        post_right_plane_pt, post_right_plane_normal = make_plane(rspv_ost_center, ripv_ost_center, heart_center)
        
        # Find bottommost vertices of RIPV and LIPV rings (PV regions, not ostium)
        ripv_ring = np.where(regions == 3)[0]  # RIPV region
        lipv_ring = np.where(regions == 4)[0]  # LIPV region
        ripv_bottom_vid = None
        lipv_bottom_vid = None
        
        if len(ripv_ring) > 0:
            si_scores = np.dot(self.points[ripv_ring] - ostia_centers['RIPV'], si_axis)
            ripv_bottom_vid = ripv_ring[np.argmin(si_scores)]  # Minimum SI = bottommost
        
        if len(lipv_ring) > 0:
            si_scores = np.dot(self.points[lipv_ring] - ostia_centers['LIPV'], si_axis)
            lipv_bottom_vid = lipv_ring[np.argmin(si_scores)]  # Minimum SI = bottommost
        
        # Inferior bottom plane - connects bottommost vertices of RIPV and LIPV, parallel to AP axis
        inf_bottom_plane_pt = None
        inf_bottom_plane_normal = None
        if ripv_bottom_vid is not None and lipv_bottom_vid is not None:
            p1 = self.points[ripv_bottom_vid]
            p2 = self.points[lipv_bottom_vid]
            # Plane parallel to AP axis: normal perpendicular to both (p2-p1) and AP axis
            line_vec = p2 - p1
            line_vec = line_vec / (np.linalg.norm(line_vec) + 1e-10)
            normal = np.cross(line_vec, ap_axis)
            normal = normal / (np.linalg.norm(normal) + 1e-10)
            # Ensure normal points toward PV quadrangle center
            to_center = pv_quadrangle_center - p1
            if np.dot(normal, to_center) < 0:
                normal = -normal
            inf_bottom_plane_pt = p1
            inf_bottom_plane_normal = normal
        else:
            # Fallback to ostium centers if ring vertices not available
            inf_bottom_plane_pt, inf_bottom_plane_normal = make_plane(ripv_ost_center, lipv_ost_center, heart_center)
        
        # Inferior wall side planes - use RIPV and LIPV ostia centers (symmetric with anterior)
        # These planes go through ostia center → MA center → heart center
        inf_right_plane_pt = None
        inf_right_plane_normal = None
        inf_left_plane_pt = None
        inf_left_plane_normal = None
        
        ma_center = ellipse_center
        
        # Right inferior plane: through RIPV ostia center, MA center, and heart center
        inf_right_plane_pt, inf_right_plane_normal = make_plane(ripv_ost_center, ma_center, heart_center)
        # Normal should point toward the LEFT (inward, toward center of inferior region)
        to_left = lipv_ost_center - ripv_ost_center
        if np.dot(inf_right_plane_normal, to_left) < 0:
            inf_right_plane_normal = -inf_right_plane_normal
        
        # Left inferior plane: through LIPV ostia center, MA center, and heart center
        inf_left_plane_pt, inf_left_plane_normal = make_plane(lipv_ost_center, ma_center, heart_center)
        # Normal should point toward the RIGHT (inward, toward center of inferior region)
        to_right = ripv_ost_center - lipv_ost_center
        if np.dot(inf_left_plane_normal, to_right) < 0:
            inf_left_plane_normal = -inf_left_plane_normal
        
        # Roof anterior plane - use most anterior vertices from the LOWER boundary of RSPV and LSPV ostium rings
        # Ostium rings have 2 boundaries:
        #   - Upper boundary: vertices adjacent to the PV (vein) region
        #   - Lower boundary: vertices adjacent to other regions (not this ostium, not this vein)
        # Only the lower boundary vertices should be candidates for the roof anterior plane
        rspv_ostium = np.where(regions == 13)[0]  # RSPV_Ostium
        lspv_ostium = np.where(regions == 14)[0]  # LSPV_Ostium
        rspv_anterior_vid = None
        lspv_anterior_vid = None
        
        def get_lower_boundary_vertices(ostium_verts, ostium_rid, pv_rid):
            """Get vertices on the lower boundary of an ostium ring.
            Lower boundary = vertices with neighbors that are NOT this ostium and NOT the corresponding PV.
            """
            lower_boundary = []
            ostium_set = set(ostium_verts)
            for vid in ostium_verts:
                for neighbor in self.graph.neighbors(vid):
                    neighbor_region = regions[neighbor]
                    # If neighbor is not the ostium itself AND not the PV, this is a lower boundary vertex
                    if neighbor_region != ostium_rid and neighbor_region != pv_rid:
                        lower_boundary.append(vid)
                        break
            return np.array(lower_boundary) if lower_boundary else None
        
        if len(rspv_ostium) > 0:
            # Get lower boundary vertices (touching regions other than RSPV_Ostium=13 and RSPV=1)
            rspv_lower = get_lower_boundary_vertices(rspv_ostium, 13, 1)
            if rspv_lower is not None and len(rspv_lower) > 0:
                ap_scores = np.dot(self.points[rspv_lower] - ostia_centers['RSPV'], ap_axis)
                rspv_anterior_vid = rspv_lower[np.argmin(ap_scores)]
            else:
                # Fallback: use any ostium vertex with minimum AP score
                ap_scores = np.dot(self.points[rspv_ostium] - ostia_centers['RSPV'], ap_axis)
                rspv_anterior_vid = rspv_ostium[np.argmin(ap_scores)]
        
        if len(lspv_ostium) > 0:
            # Get lower boundary vertices (touching regions other than LSPV_Ostium=14 and LSPV=2)
            lspv_lower = get_lower_boundary_vertices(lspv_ostium, 14, 2)
            if lspv_lower is not None and len(lspv_lower) > 0:
                ap_scores = np.dot(self.points[lspv_lower] - ostia_centers['LSPV'], ap_axis)
                lspv_anterior_vid = lspv_lower[np.argmin(ap_scores)]
            else:
                # Fallback: use any ostium vertex with minimum AP score
                ap_scores = np.dot(self.points[lspv_ostium] - ostia_centers['LSPV'], ap_axis)
                lspv_anterior_vid = lspv_ostium[np.argmin(ap_scores)]
        
        roof_ant_plane_pt = None
        roof_ant_plane_normal = None
        if rspv_anterior_vid is not None and lspv_anterior_vid is not None:
            p1 = self.points[rspv_anterior_vid]
            p2 = self.points[lspv_anterior_vid]
            roof_ant_plane_pt, roof_ant_plane_normal = make_plane(p1, p2, heart_center)
            # Normal should point TOWARD MA/anterior (away from roof/PVs)
            # Use MA center as reference - normal should point toward it
            to_ma = ellipse_center - p1
            if np.dot(roof_ant_plane_normal, to_ma) < 0:
                roof_ant_plane_normal = -roof_ant_plane_normal
        
        ant_plane_right_pt = None
        ant_plane_right_normal = None
        ant_plane_left_pt = None
        ant_plane_left_normal = None
        
        ma_center = ellipse_center
        # Use ostia centers directly for more reliable plane positioning
        # (instead of most anterior ostium vertex which can be mispositioned)
        rspv_ost_center = ostia_centers['RSPV']
        lspv_ost_center = ostia_centers['LSPV']
        
        # Right anterior plane: through RSPV ostia center, MA center, and heart center
        ant_plane_right_pt, ant_plane_right_normal = make_plane(rspv_ost_center, ma_center, heart_center)
        # Normal should point toward the LEFT (inward, toward center of anterior region)
        to_left = lspv_ost_center - rspv_ost_center
        if np.dot(ant_plane_right_normal, to_left) < 0:
            ant_plane_right_normal = -ant_plane_right_normal
        
        # Left anterior plane: through LSPV ostia center, MA center, and heart center
        ant_plane_left_pt, ant_plane_left_normal = make_plane(lspv_ost_center, ma_center, heart_center)
        # Normal should point toward the RIGHT (inward, toward center of anterior region)
        to_right = rspv_ost_center - lspv_ost_center
        if np.dot(ant_plane_left_normal, to_right) < 0:
            ant_plane_left_normal = -ant_plane_left_normal
        
        # Septal wall plane - parallel to SI axis, passing through MA center
        # Normal is perpendicular to SI axis (use LR axis as the plane normal)
        septal_wall_plane_pt = ma_center
        septal_wall_plane_normal = lr_axis
        
        return {
            'ostia_centers': ostia_centers,
            'si_axis': si_axis,
            'ap_axis': ap_axis,
            'lr_axis': lr_axis,
            'heart_center': heart_center,
            'pv_quadrangle_center': pv_quadrangle_center,
            'rspv_ost_center': rspv_ost_center,
            'lspv_ost_center': lspv_ost_center,
            'ripv_ost_center': ripv_ost_center,
            'lipv_ost_center': lipv_ost_center,
            'post_top_plane_pt': post_top_plane_pt,
            'post_top_plane_normal': post_top_plane_normal,
            'post_bottom_plane_pt': post_bottom_plane_pt,
            'post_bottom_plane_normal': post_bottom_plane_normal,
            'post_left_plane_pt': post_left_plane_pt,
            'post_left_plane_normal': post_left_plane_normal,
            'post_right_plane_pt': post_right_plane_pt,
            'post_right_plane_normal': post_right_plane_normal,
            'inf_bottom_plane_pt': inf_bottom_plane_pt,
            'inf_bottom_plane_normal': inf_bottom_plane_normal,
            'inf_right_plane_pt': inf_right_plane_pt,
            'inf_right_plane_normal': inf_right_plane_normal,
            'inf_left_plane_pt': inf_left_plane_pt,
            'inf_left_plane_normal': inf_left_plane_normal,
            'roof_ant_plane_pt': roof_ant_plane_pt,
            'roof_ant_plane_normal': roof_ant_plane_normal,
            'roof_ant_rspv_vid': rspv_anterior_vid,  # Debug: vertex ID used for roof_ant_plane (RSPV side)
            'roof_ant_lspv_vid': lspv_anterior_vid,  # Debug: vertex ID used for roof_ant_plane (LSPV side)
            'ant_plane_right_pt': ant_plane_right_pt,
            'ant_plane_right_normal': ant_plane_right_normal,
            'ant_plane_left_pt': ant_plane_left_pt,
            'ant_plane_left_normal': ant_plane_left_normal,
            'septal_wall_plane_pt': septal_wall_plane_pt,
            'septal_wall_plane_normal': septal_wall_plane_normal,
        }
    
    def create_posterior_wall(self, regions, geom):
        """Create posterior wall region (rid=7)."""
        dist_post_top = self.signed_distance_to_plane(self.points, geom['post_top_plane_pt'], geom['post_top_plane_normal'])
        dist_post_bottom = self.signed_distance_to_plane(self.points, geom['post_bottom_plane_pt'], geom['post_bottom_plane_normal'])
        dist_post_right = self.signed_distance_to_plane(self.points, geom['post_right_plane_pt'], geom['post_right_plane_normal'])
        dist_post_left = self.signed_distance_to_plane(self.points, geom['post_left_plane_pt'], geom['post_left_plane_normal'])
        
        unassigned = regions == 0
        post_mask = (unassigned & (dist_post_top > 0) & (dist_post_bottom > 0) & (dist_post_right > 0) & (dist_post_left > 0))
        regions[post_mask] = 7
        print(f"  Posterior: {np.sum(regions == 7)}")
    
    def create_roof_wall(self, regions, geom):
        """Create roof wall region (rid=8)."""
        dist_post_top = self.signed_distance_to_plane(self.points, geom['post_top_plane_pt'], geom['post_top_plane_normal'])
        dist_post_right = self.signed_distance_to_plane(self.points, geom['post_right_plane_pt'], geom['post_right_plane_normal'])
        dist_post_left = self.signed_distance_to_plane(self.points, geom['post_left_plane_pt'], geom['post_left_plane_normal'])
        
        dist_roof_ant = np.zeros(len(self.points))
        if geom['roof_ant_plane_pt'] is not None:
            dist_roof_ant = self.signed_distance_to_plane(self.points, geom['roof_ant_plane_pt'], geom['roof_ant_plane_normal'])
        
        dist_ant_right = self.signed_distance_to_plane(self.points, geom['rspv_ost_center'], geom['lr_axis'])
        dist_ant_left = self.signed_distance_to_plane(self.points, geom['lspv_ost_center'], geom['lr_axis'])
        
        unassigned = regions == 0
        roof_mask = (unassigned & (dist_post_top < 0) & (dist_roof_ant < 0) & (dist_ant_right < 0) & (dist_ant_left > 0))
        regions[roof_mask] = 8
        print(f"  Roof: {np.sum(regions == 8)}")
    
    def create_inferior_wall(self, regions, geom):
        """Create inferior wall region (rid=9)."""
        # Use post_bottom plane to separate inferior from posterior (symmetric with roof_ant for anterior)
        dist_post_bottom = self.signed_distance_to_plane(self.points, geom['post_bottom_plane_pt'], geom['post_bottom_plane_normal'])
        
        dist_inf_right = np.zeros(len(self.points))
        dist_inf_left = np.zeros(len(self.points))
        if geom['inf_right_plane_pt'] is not None:
            dist_inf_right = self.signed_distance_to_plane(self.points, geom['inf_right_plane_pt'], geom['inf_right_plane_normal'])
        if geom['inf_left_plane_pt'] is not None:
            dist_inf_left = self.signed_distance_to_plane(self.points, geom['inf_left_plane_pt'], geom['inf_left_plane_normal'])
        
        unassigned = regions == 0
        # Inferior wall is (symmetric with anterior):
        # - Below post_bottom plane (dist_post_bottom < 0, since normal points toward PV center)
        # - On the inward side of both left and right planes (dist > 0, since normals point inward)
        inferior_mask = (unassigned & (dist_post_bottom < 0) & (dist_inf_right > 0) & (dist_inf_left > 0))
        regions[inferior_mask] = 9
        print(f"  Inferior: {np.sum(regions == 9)}")
    
    def create_anterior_wall(self, regions, geom):
        """Create anterior wall region (rid=12)."""
        dist_roof_ant = np.zeros(len(self.points))
        if geom['roof_ant_plane_pt'] is not None:
            dist_roof_ant = self.signed_distance_to_plane(self.points, geom['roof_ant_plane_pt'], geom['roof_ant_plane_normal'])
        
        dist_ant_right = np.zeros(len(self.points))
        dist_ant_left = np.zeros(len(self.points))
        if geom['ant_plane_right_pt'] is not None:
            dist_ant_right = self.signed_distance_to_plane(self.points, geom['ant_plane_right_pt'], geom['ant_plane_right_normal'])
        if geom['ant_plane_left_pt'] is not None:
            dist_ant_left = self.signed_distance_to_plane(self.points, geom['ant_plane_left_pt'], geom['ant_plane_left_normal'])
        
        unassigned = regions == 0
        # Anterior wall is:
        # - In front of roof_ant plane (dist_roof_ant > 0, since normal points toward MA/anterior)
        # - On the inward side of both left and right planes (dist > 0, since normals point inward)
        anterior_mask = (unassigned & (dist_roof_ant > 0) & (dist_ant_right > 0) & (dist_ant_left > 0))

        regions[anterior_mask] = 12
        print(f"  Anterior: {np.sum(regions == 12)}")
    
    def create_septal_wall(self, regions, geom):
        """Create septal wall region (rid=11)."""
        dist_septal_wall = self.signed_distance_to_plane(self.points, geom['septal_wall_plane_pt'], geom['septal_wall_plane_normal'])
        
        unassigned = regions == 0
        septal_mask = (unassigned & (dist_septal_wall > 0))
        regions[septal_mask] = 11
        print(f"  Septal: {np.sum(regions == 11)}")
    
    def create_lateral_wall(self, regions, geom):
        """Create lateral wall region (rid=10)."""
        dist_septal_wall = self.signed_distance_to_plane(self.points, geom['septal_wall_plane_pt'], geom['septal_wall_plane_normal'])        

        unassigned = regions == 0
        lateral_mask = (unassigned & (dist_septal_wall < 0))
        regions[lateral_mask] = 10
        print(f"  Lateral: {np.sum(regions == 10)}")
    
    def create_laa_and_walls(self, regions, ellipse_center):
        """Create wall regions (LAA already created during landmark selection)."""
        print("\n2. Creating wall regions...")
        geom = self.compute_wall_geometry(regions, ellipse_center)
        
        self.create_posterior_wall(regions, geom)
        self.create_roof_wall(regions, geom)
        self.create_inferior_wall(regions, geom)
        self.create_anterior_wall(regions, geom)
        self.create_lateral_wall(regions, geom)
        self.create_septal_wall(regions, geom)
    

    def review_segmentation(self, regions, title="SEGMENTATION REVIEW", step=None, wall_rid=None, geom=None):
        """Review segmentation with navigation controls. Returns 'next' to proceed, 'undo' to go back."""
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
        
        is_final_review = (step == 'FINAL')
        is_veins_review = (step == 'VEINS')
        is_walls_review = (step == 'WALLS')
        is_wall_single = (step == 'WALL_SINGLE')
        
        if is_final_review:
            help_text = "SPACE=save results, ESC=discard"
            print(f"\n{help_text}")
        elif is_veins_review:
            help_text = "SPACE=continue, 's'=checkpoint"
            print(f"\n{help_text}\n")
        elif is_walls_review:
            help_text = "SPACE=continue, ESC=exit the program"
            print(f"\n{help_text}\n")
        elif is_wall_single:
            help_text = "SPACE=accept, ESC=redo this wall"
            print(f"\n{help_text}\n")
        else:
            help_text = "SPACE=continue, ESC=undo"
            print(f"\n{help_text}\n")
        
        # Setup VTK viewing
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.1, 0.1)
        
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(1000, 1000)
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        # Create colored mesh based on regions
        region_colors = vtk.vtkUnsignedCharArray()
        region_colors.SetNumberOfComponents(3)
        region_colors.SetName("Colors")
        
        for vid in range(len(self.points)):
            rid = int(regions[vid])
            c = self.extended_color_map.get(rid, (200, 200, 200))
            region_colors.InsertNextTuple3(*c)
        
        self.mesh.GetPointData().SetScalars(region_colors)
        
        # Add mesh visualization
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.mesh)
        mapper.SetScalarModeToUsePointData()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Make mesh fully opaque for wall visualization
        actor.GetProperty().SetOpacity(1.0)
        
        renderer.AddActor(actor)

        # Add exterior mesh visualization on final review
        if is_final_review and self.exterior_mesh:
            print("  visualizing epicardium...")
            ext_mapper = vtk.vtkPolyDataMapper()
            ext_mapper.SetInputData(self.exterior_mesh)
            ext_actor = vtk.vtkActor()
            ext_actor.SetMapper(ext_mapper)
            ext_actor.GetProperty().SetColor(0.9, 0.9, 0.9)  # Light gray
            ext_actor.GetProperty().SetOpacity(0.3)  # Semi-transparent
            renderer.AddActor(ext_actor)
        
        # Debug visualization: show roof_ant_plane anchor points during wall review
        if is_walls_review and geom is not None:
            # Visualize the two points used for roof_ant_plane creation
            rspv_vid = geom.get('roof_ant_rspv_vid')
            lspv_vid = geom.get('roof_ant_lspv_vid')
            
            if rspv_vid is not None:
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(*self.points[rspv_vid])
                sphere.SetRadius(2.0)
                sphere.SetPhiResolution(16)
                sphere.SetThetaResolution(16)
                sm = vtk.vtkPolyDataMapper()
                sm.SetInputConnection(sphere.GetOutputPort())
                sa = vtk.vtkActor()
                sa.SetMapper(sm)
                sa.GetProperty().SetColor(1, 0, 0)  # Red for RSPV side
                renderer.AddActor(sa)
                print(f"  DEBUG: RSPV roof_ant point vid={rspv_vid} at {self.points[rspv_vid]}")
            
            if lspv_vid is not None:
                sphere = vtk.vtkSphereSource()
                sphere.SetCenter(*self.points[lspv_vid])
                sphere.SetRadius(2.0)
                sphere.SetPhiResolution(16)
                sphere.SetThetaResolution(16)
                sm = vtk.vtkPolyDataMapper()
                sm.SetInputConnection(sphere.GetOutputPort())
                sa = vtk.vtkActor()
                sa.SetMapper(sm)
                sa.GetProperty().SetColor(0, 0, 1)  # Blue for LSPV side
                renderer.AddActor(sa)
                print(f"  DEBUG: LSPV roof_ant point vid={lspv_vid} at {self.points[lspv_vid]}")
            
            # Also draw a line between them to show the roof_ant_plane edge
            if rspv_vid is not None and lspv_vid is not None:
                line_pts = vtk.vtkPoints()
                line_pts.InsertNextPoint(self.points[rspv_vid])
                line_pts.InsertNextPoint(self.points[lspv_vid])
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, 0)
                line.GetPointIds().SetId(1, 1)
                lines = vtk.vtkCellArray()
                lines.InsertNextCell(line)
                line_pd = vtk.vtkPolyData()
                line_pd.SetPoints(line_pts)
                line_pd.SetLines(lines)
                tube = vtk.vtkTubeFilter()
                tube.SetInputData(line_pd)
                tube.SetRadius(0.5)
                tube.SetNumberOfSides(8)
                lm = vtk.vtkPolyDataMapper()
                lm.SetInputConnection(tube.GetOutputPort())
                la = vtk.vtkActor()
                la.SetMapper(lm)
                la.GetProperty().SetColor(1, 1, 0)  # Yellow line
                renderer.AddActor(la)
        
        # Add text display for title and keybindings
        text_actor = vtk.vtkTextActor()
        text_actor.GetTextProperty().SetFontSize(14)
        text_actor.GetTextProperty().SetColor(1, 1, 0)
        text_actor.SetPosition(10, 10)
        text_actor.SetInput(f"{title}\n\n{help_text}")
        renderer.AddViewProp(text_actor)
        
        # Status text for messages like checkpoint saved
        status_text = vtk.vtkTextActor()
        status_text.GetTextProperty().SetFontSize(12)
        status_text.GetTextProperty().SetColor(0, 1, 0)
        status_text.SetPosition(10, window.GetSize()[1] - 50)
        status_text.SetInput("")
        renderer.AddViewProp(status_text)
        
        # State for key handling
        state = {'action': None}
        
        def on_key(obj, event):
            key = interactor.GetKeySym()
            if key == 'space':
                state['action'] = 'next'
                window.Finalize()
                interactor.TerminateApp()
            elif key == 'Escape':
                # ESC is blocked during veins review
                if is_veins_review:
                    return
                # ESC behavior depends on review type
                if is_final_review:
                    state['action'] = 'discard'
                else:
                    state['action'] = 'undo'
                window.Finalize()
                interactor.TerminateApp()
            elif is_final_review and key == 'q':
                state['action'] = 'quit'
                window.Finalize()
                interactor.TerminateApp()
            elif is_veins_review and key == 's':
                # Save checkpoint
                base_path = self.vtk_file.replace('.vtk', '')
                self.save_checkpoint(base_path)
                status_text.SetInput("✓ Checkpoint saved")
                window.Render()
        
        interactor.AddObserver('KeyPressEvent', on_key)
        
        # Close window to terminate program
        def on_window_close(obj, event):
            print("\n✗ Window closed. Terminating program.")
            state['action'] = 'window_closed'
            window.Finalize()
            interactor.TerminateApp()
        
        interactor.AddObserver('WinCloseEvent', on_window_close)
        
        window.SetWindowName(title)
        interactor.Initialize()
        window.Render()
        interactor.Start()
        
        # If window was closed, ensure we return window_closed regardless of state
        if state['action'] is None:
            print("DEBUG: state['action'] is None after interactor.Start()")
            return 'window_closed'
        
        # If window was closed, return window_closed regardless
        if state['action'] == 'window_closed':
            return 'window_closed'
        return state['action'] if state['action'] is not None else 'next'
    
    def create_boundary_actor(self, regions):
        edges = set()
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i+1) % 3]
                if regions[v1] != regions[v2]:
                    edges.add((min(v1, v2), max(v1, v2)))
        
        if not edges:
            return None
        
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for v1, v2 in edges:
            i1 = pts.InsertNextPoint(self.points[v1])
            i2 = pts.InsertNextPoint(self.points[v2])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i1)
            line.GetPointIds().SetId(1, i2)
            lines.InsertNextCell(line)
        
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(pd)
        tube.SetRadius(0.3)
        tube.SetNumberOfSides(8)
        
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(tube.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(1, 1, 1)
        return a
    
    def create_coordinate_system_actor(self, origin, si_axis, ap_axis, lr_axis, scale=100.0):
        """Create actors for SI, AP, LR axes with arrows"""
        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")
        
        # SI axis (green) - Superior-Inferior
        p0 = pts.InsertNextPoint(origin)
        p1 = pts.InsertNextPoint(origin + si_axis * scale)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p0)
        line.GetPointIds().SetId(1, p1)
        lines.InsertNextCell(line)
        colors.InsertNextTuple3(0, 255, 0)
        colors.InsertNextTuple3(0, 255, 0)
        
        # AP axis (red) - Anterior-Posterior
        p0 = pts.InsertNextPoint(origin)
        p1 = pts.InsertNextPoint(origin + ap_axis * scale)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p0)
        line.GetPointIds().SetId(1, p1)
        lines.InsertNextCell(line)
        colors.InsertNextTuple3(255, 0, 0)
        colors.InsertNextTuple3(255, 0, 0)
        
        # LR axis (blue) - Left-Right
        p0 = pts.InsertNextPoint(origin)
        p1 = pts.InsertNextPoint(origin + lr_axis * scale)
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p0)
        line.GetPointIds().SetId(1, p1)
        lines.InsertNextCell(line)
        colors.InsertNextTuple3(0, 0, 255)
        colors.InsertNextTuple3(0, 0, 255)
        
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(lines)
        pd.GetPointData().SetScalars(colors)
        
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(pd)
        tube.SetRadius(1.0)
        tube.SetNumberOfSides(8)
        
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(tube.GetOutputPort())
        m.SetScalarModeToUsePointData()
        a = vtk.vtkActor()
        a.SetMapper(m)
        
        return a
    
    def save_checkpoint(self, base_path):
        """Save LASegmenter state to .wrk file"""
        wrk_file = base_path + '.wrk'
        try:
            with open(wrk_file, 'wb') as f:
                pickle.dump(self, f)
            print(f"\n✓ Checkpoint saved: {wrk_file}")
            return True
        except Exception as e:
            print(f"\n✗ Failed to save checkpoint: {e}")
            return False
    
    @staticmethod
    def load_from_checkpoint(wrk_file):
        """Load LASegmenter state from .wrk file"""
        try:
            with open(wrk_file, 'rb') as f:
                segmenter = pickle.load(f)
            print(f"✓ Checkpoint loaded: {wrk_file}")
            return segmenter
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return None
    
    def save_results(self, regions, prefix):
        print("\n" + "="*60)
        print("  SAVING")
        print("="*60 + "\n")
        
        # np.save(f"{prefix}_regions.npy", regions)
        # print(f"✓ {prefix}_regions.npy")
        
        with open(f"{prefix}_region_names.txt", 'w') as f:
            f.write("ID,Name,R,G,B\n")
            for i, n in enumerate(self.extended_region_names):
                c = self.extended_color_map.get(i, (200,200,200))
                f.write(f"{i},{n},{c[0]},{c[1]},{c[2]}\n")
        print(f"✓ {prefix}_region_names.txt")
        
        arr = vtk.vtkIntArray()
        arr.SetName("Regions")
        for r in regions:
            arr.InsertNextValue(int(r))
        
        out = vtk.vtkPolyData()
        out.DeepCopy(self.mesh)
        out.GetPointData().AddArray(arr)
        out.GetPointData().SetActiveScalars("Regions")
        
        if self.exterior_mesh:
            print("  Appending exterior mesh to output...")
            ext_arr = vtk.vtkIntArray()
            ext_arr.SetName("Epicardium")
            num_ext = self.exterior_mesh.GetNumberOfPoints()
            for _ in range(num_ext):
                ext_arr.InsertNextValue(0)
            
            ext_copy = vtk.vtkPolyData()
            ext_copy.DeepCopy(self.exterior_mesh)
            ext_copy.GetPointData().AddArray(ext_arr)
             # Ensure "Regions" is active on exterior copy too
            ext_copy.GetPointData().SetActiveScalars("Epicardium")

            # Append
            append = vtk.vtkAppendPolyData()
            append.AddInputData(out)
            append.AddInputData(ext_copy)
            append.Update()
            out = append.GetOutput()

        w = vtk.vtkPolyDataWriter()
        w.SetFileName(f"{prefix}_regions.vtk")
        w.SetInputData(out)
        w.Write()
        print(f"✓ {prefix}_regions.vtk")
        
        # colors = np.array([self.extended_color_map.get(int(r), (200,200,200)) for r in regions], dtype=np.uint8)
        # ca = vtk.vtkUnsignedCharArray()
        # ca.SetNumberOfComponents(3)
        # ca.SetName("Colors")
        # for c in colors:
        #     ca.InsertNextTuple3(*c)
        
        # ply = vtk.vtkPolyData()
        # ply.DeepCopy(self.mesh)
        # ply.GetPointData().SetScalars(ca)
        
        # pw = vtk.vtkPLYWriter()
        # pw.SetFileName(f"{prefix}_regions.ply")
        # pw.SetInputData(ply)
        # pw.Write()
        # print(f"✓ {prefix}_regions.ply")
        
        # with open(f"{prefix}_landmarks.txt", 'w') as f:
        #     f.write("Region,Type,VertexID,X,Y,Z,Radius\n")
        #     for k, v in sorted(self.markers.items()):
        #         parts = k.rsplit('_', 1)
        #         reg, typ = parts[0], parts[1]
        #         c = v['coords']
        #         vid = v.get('point_id', -1) or -1
        #         rad = v.get('radius', 0) or 0
        #         f.write(f"{reg},{typ},{vid},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{rad:.2f}\n")
        # print(f"✓ {prefix}_landmarks.txt")
    
    def run(self):
        # Load from checkpoint if this is a .wrk file
        if self.vtk_file.endswith('.wrk'):
            segmenter = LASegmenter.load_from_checkpoint(self.vtk_file)
            if segmenter is None:
                return None
            # Copy loaded state to self
            self.mesh = segmenter.mesh
            self.points = segmenter.points
            self.faces = segmenter.faces
            self.graph = segmenter.graph
            self.markers = segmenter.markers
            # Copy centering offset (critical for aligning exterior mesh later)
            if hasattr(segmenter, 'centering_offset'):
                self.centering_offset = segmenter.centering_offset
            regions = np.zeros(len(self.points), dtype=int)
            
            # Recreate all regions from saved markers
            print("\nRecreating regions from checkpoint...")
            
            # 1. Create MA region
            self.create_ma_region(regions)
            
            # 2. Create PV regions (RSPV, LSPV, RIPV, LIPV, LAA)
            for pv_name in ['RSPV', 'LSPV', 'RIPV', 'LIPV', 'LAA']:
                if f'{pv_name}_ostium' in self.markers:
                    self.create_pv_regions(regions, pv_name)
            
        else:
            # Normal flow: load mesh, select landmarks interactively
            # (Reviews are now integrated into select_landmarks_interactive with review_mode)
            self.load_mesh()
            self.center_mesh()
            self.build_graph()
            regions, last_pv = self.select_landmarks_interactive()
        
        # Initialize last_pv for checkpoint loading path
        if 'last_pv' not in locals():
            last_pv = None
        
        # Check if window was closed during landmark selection
        if last_pv == 'window_closed':
            print("\n✗ Program terminated by user during landmark selection.")
            return None
        
        print(f"\nDEBUG: After landmarks, last_pv = {last_pv}")
        action = self.review_segmentation(regions, "VEINS FINAL REVIEW - for save", step='VEINS')
        print(f"DEBUG: After veins review, action = {action}")
        
        # Check if window was closed during veins review
        if action == 'window_closed':
            print("\n✗ Program terminated by user.")
            return None

        # Extract ellipse center from MA points for wall creation
        ma_p1 = self.markers['MA_point1']['coords']
        ma_p2 = self.markers['MA_point2']['coords']
        ma_p3 = self.markers['MA_point3']['coords']
        ma_p4 = self.markers['MA_point4']['coords']
        
        d1 = ma_p2 - ma_p1
        d2 = ma_p4 - ma_p3
        A = np.column_stack([d1, -d2])
        b = ma_p3 - ma_p1
        try:
            params, _ = np.linalg.lstsq(A, b, rcond=None)
            t = params[0]
            ellipse_center = ma_p1 + t * d1
        except:
            ellipse_center = (ma_p1 + ma_p2 + ma_p3 + ma_p4) / 4.0
        
        # Create walls with per-wall review
        wall_sequence = [
            ('Posterior', self.create_posterior_wall, 7),
            ('Roof', self.create_roof_wall, 8),
            ('Inferior', self.create_inferior_wall, 9),
            ('Anterior', self.create_anterior_wall, 12),
            ('Lateral', self.create_lateral_wall, 10),
            ('Septal', self.create_septal_wall, 11)
        ]
        
        geom = self.compute_wall_geometry(regions, ellipse_center)
        
        # Create all walls without per-wall review (single wall review disabled for now)
        # TODO: Re-enable individual wall reviews in future for debugging purposes
        for wall_idx, (wall_name, create_func, wall_rid) in enumerate(wall_sequence):
            print("\n" + "="*60)
            print(f"  CREATE {wall_name.upper()} WALL")
            print("="*60)
            
            # Create the wall
            create_func(regions, geom)

        
        action = self.review_segmentation(regions, "WALL REVIEW - before smoothing", step='WALLS', geom=geom)
        
        # Check if user pressed ESC to quit or closed window
        if action == 'undo' or action == 'window_closed':
            print("\n✗ Program terminated by user.")
            return None

        # Assign remaining vertices to nearest region
        unassigned = regions == 0
        remaining = np.sum(unassigned)
        if remaining > 0:
            print(f"\n  Assigning {remaining} remaining vertices...")
            assigned = np.where(regions > 0)[0]
            for idx in np.where(unassigned)[0]:
                dists = np.linalg.norm(self.points[assigned] - self.points[idx], axis=1)
                regions[idx] = regions[assigned[np.argmin(dists)]]
            
        
        # Smoothing and continuity enforcement
        print("\n" + "="*60)
        print("  SMOOTHING AND CONTINUITY")
        print("="*60)
        self.smooth_boundaries(regions, iterations=3)
        self.enforce_continuity(regions)
        self.smooth_boundaries(regions, iterations=2)
        self.enforce_continuity(regions)
        
        # Final review and save
        print("\n" + "="*60)
        print("  FINAL SUMMARY")
        print("="*60)
        for i, n in enumerate(self.extended_region_names):
            c = np.sum(regions == i)
            if c > 0:
                print(f"  {i:2d}. {n:20s}: {c:6d}")
        
        # Load Epicardium
        print("\n" + "="*60)
        print("  LOAD EPICARDIUM (EXTERIOR) MESH")
        print("="*60)
        
        epi_path = None
        if tk:
            try:
                # Create a hidden root window if not already existing
                root = tk.Tk()
                root.withdraw()
                print("Please select the EPICARDIUM (exterior) mesh file...")
                epi_path = filedialog.askopenfilename(defaultextension=".vtk", title="Select epicardium mesh file")
                root.destroy()
            except Exception as e:
                print(f"Error opening file dialog: {e}")
        
        if epi_path:
            print(f"Loading epicardium: {epi_path}")
            # Use LASegmenter dict/class or just load mesh directly?
            # Reusing LASegmenter class is easy for loading/centering
            self.exterior_segmenter = LASegmenter(epi_path)
            self.exterior_segmenter.load_mesh()
            
            # Apply same centering offset as interior
            if self.centering_offset is not None:
                print(f"Applying centering offset to exterior mesh: {self.centering_offset}")
                self.exterior_segmenter.center_mesh(offset=self.centering_offset)
            else:
                 print("⚠ Interior has no centering offset. Assuming meshes are pre-aligned.")
            
            # Build graph if needed (maybe not needed for thickness calc if only using KDTree/RayCast on mesh?)
            # But the user request said "loading epicardium and building its' graph". So I will build it.
            self.exterior_segmenter.build_graph()
            self.exterior_mesh = self.exterior_segmenter.mesh
            print("✓ Epicardium loaded and aligned.")
        else:
            print("⚠ No epicardium selected. Results will only contain endocardium.")

        action = self.review_segmentation(regions, "FINAL REVIEW", step='FINAL')
        
        if action == 'window_closed':
            print("\n✗ Program terminated by user.")
            return None
        elif action == 'discard' or action == 'quit':
            print("\n✗ Not saved")
            return None
        else:
            # SPACE was pressed - save results
            self.save_results(regions, self.vtk_file.replace('.vtk', '').replace('.wrk', ''))
            print("\n✓ Segmentation complete!")
            return regions





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    args = parser.parse_args()
    
    if args.file is None:
        try:
            if tk:
                root = tk.Tk()
                root.withdraw()
                f = filedialog.askopenfilename(filetypes=[("VTK", "*.vtk"), ("Checkpoint", "*.wrk")])
                root.destroy()
                if not f:
                    return
                args.file = f
        except:
            print("Usage: python LAsegmenter.py [file.vtk|file.wrk]")
            return
    
    interior = LASegmenter(args.file)
    interior.run()

if __name__ == '__main__':
    main()