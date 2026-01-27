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

class LASegmenter:
    def __init__(self, vtk_file):
        self.vtk_file = vtk_file
        self.mesh = None
        self.points = None
        self.faces = None
        self.graph = None
        self.markers = {}
        self.default_ostium_radius = 10.0
        
        self.landmark_sequence = [
            ('RSPV', 'vein', 'RSPV - Click TIP, then position cutting plane'),
            ('LSPV', 'vein', 'LSPV - Click TIP, then position cutting plane'),
            ('RIPV', 'vein', 'RIPV - Click TIP, then position cutting plane'),
            ('LIPV', 'vein', 'LIPV - Click TIP, then position cutting plane'),
            ('LAA', 'vein', 'LAA - Click TIP, then position cutting plane'),
            ('MA', 'point1', 'MITRAL ANNULUS - MA1: First point on major axis'),
            ('MA', 'point2', 'MITRAL ANNULUS - MA2: Second point on major axis'),
            ('MA', 'point3', 'MITRAL ANNULUS - MA3: First point on minor axis'),
            ('MA', 'point4', 'MITRAL ANNULUS - MA4: Second point on minor axis'),
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
        
    def center_mesh(self):
        center = np.mean(self.points, axis=0)
        self.points -= center
        vtk_points = vtk.vtkPoints()
        for p in self.points:
            vtk_points.InsertNextPoint(p)
        self.mesh.SetPoints(vtk_points)
        
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
            'ma_point_id': None, 'ma_coords': None, 'done': False
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
                
                update_plane()
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
                    
                    state['tip_id'] = None
                    state['plane_pos'] = None
                    
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
                else:
                    return
                
                state['idx'] += 1
                info.SetInput("")
                
                if state['idx'] < len(self.landmark_sequence):
                    update_text()
                    text.GetTextProperty().SetColor(1, 1, 0)
                    # Reset camera only when transitioning TO MA (not between MA points)
                    if self.landmark_sequence[state['idx']][0] == 'MA' and (state['idx'] == 0 or self.landmark_sequence[state['idx']-1][0] != 'MA'):
                        renderer.GetActiveCamera().SetPosition(0, 100, 50)
                        renderer.GetActiveCamera().SetFocalPoint(0, 0, 0)
                    window.Render()
                else:
                    text.SetInput("✓ ALL DONE! Close window.")
                    text.GetTextProperty().SetColor(0, 1, 0)
                    window.Render()
                    print("\n✓ All landmarks selected!")
                    window.Finalize()
                    interactor.TerminateApp()
        
        update_text()
        interactor.AddObserver('LeftButtonPressEvent', on_click)
        interactor.AddObserver('KeyPressEvent', on_key)
        
        window.SetWindowName("LA Segmenter - Landmarks")
        interactor.Initialize()
        window.Render()
        interactor.Start()
    
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
    
    def define_regions(self):
        """Define regions with improved straight-line boundaries"""
        print("\n" + "="*60)
        print("  COMPUTING REGIONS")
        print("="*60)
        
        regions = np.zeros(len(self.points), dtype=int)
        mesh_center = np.mean(self.points, axis=0)
        
        # Get landmark positions
        rspv_c = self.markers['RSPV_ostium']['coords']
        lspv_c = self.markers['LSPV_ostium']['coords']
        ripv_c = self.markers['RIPV_ostium']['coords']
        lipv_c = self.markers['LIPV_ostium']['coords']
        laa_c = self.markers['LAA_ostium']['coords']
        
        ma_p1 = self.markers['MA_point1']['coords']
        ma_p2 = self.markers['MA_point2']['coords']
        ma_p3 = self.markers['MA_point3']['coords']
        ma_p4 = self.markers['MA_point4']['coords']
        
        # =====================================================
        # 1. Define PV regions (limited to single outpouching)
        # =====================================================
        print("\n1. Vein regions (single outpouching)...")
        for pv, rid in [('RSPV', 1), ('LSPV', 2), ('RIPV', 3), ('LIPV', 4), ('LAA', 6)]:
            ost = self.markers[f'{pv}_ostium']
            center, normal, radius = ost['coords'], ost['normal'], ost['radius']
            tip_id = self.markers[f'{pv}_distal']['point_id']
            distal_pt = self.points[tip_id]
            distal_side = np.sign(np.dot(distal_pt - center, normal))
            
            # Find candidates on distal side
            candidates = np.zeros(len(self.points), dtype=bool)
            for i, pt in enumerate(self.points):
                if np.sign(np.dot(pt - center, normal)) == distal_side:
                    candidates[i] = True
            
            # Keep only connected component containing tip
            connected = self.find_connected_component(tip_id, candidates)
            
            for vid in connected:
                regions[vid] = rid
            
            print(f"  {pv}: {len(connected)}")
        
        # =====================================================
        # 2. Define MA region - ELLIPSE bounded by 4 points
        # =====================================================
        print("\n2. MA region (ellipse defined by 4 points)...")
        
        # Get the 4 MA points
        ma_p1 = self.markers['MA_point1']['coords']
        ma_p2 = self.markers['MA_point2']['coords']
        ma_p3 = self.markers['MA_point3']['coords']
        ma_p4 = self.markers['MA_point4']['coords']
        
        # All 4 points define the plane
        # Use first 3 points to compute plane normal
        v1 = ma_p2 - ma_p1
        v2 = ma_p3 - ma_p1
        ma_plane_norm = np.cross(v1, v2)
        if np.linalg.norm(ma_plane_norm) < 1e-6:
            # Use alternative vectors if first 3 are collinear
            v2 = ma_p4 - ma_p1
            ma_plane_norm = np.cross(v1, v2)
        
        if np.linalg.norm(ma_plane_norm) < 1e-6:
            # Points are collinear, use default
            ma_plane_norm = np.array([0, 0, 1])
        else:
            ma_plane_norm = ma_plane_norm / np.linalg.norm(ma_plane_norm)
        
        # Compute intersection of diagonals MA1-MA2 and MA3-MA4 (ellipse center)
        # Line 1: P = ma_p1 + t * (ma_p2 - ma_p1)
        # Line 2: Q = ma_p3 + s * (ma_p4 - ma_p3)
        d1 = ma_p2 - ma_p1
        d2 = ma_p4 - ma_p3
        
        # Solve: ma_p1 + t*d1 = ma_p3 + s*d2 using least squares
        A = np.column_stack([d1, -d2])
        b = ma_p3 - ma_p1
        try:
            params, _ = np.linalg.lstsq(A, b, rcond=None)
            t = params[0]
            ellipse_center = ma_p1 + t * d1
        except:
            ellipse_center = (ma_p1 + ma_p2 + ma_p3 + ma_p4) / 4.0
        
        # Diagonal directions (axes of the ellipse)
        diag1 = ma_p2 - ma_p1  # MA1-MA2 diagonal
        diag2 = ma_p4 - ma_p3  # MA3-MA4 diagonal
        
        # Project diagonals onto the plane
        diag1_in_plane = diag1 - np.dot(diag1, ma_plane_norm) * ma_plane_norm
        diag2_in_plane = diag2 - np.dot(diag2, ma_plane_norm) * ma_plane_norm
        
        # Normalize to get unit vectors
        if np.linalg.norm(diag1_in_plane) > 1e-6:
            axis1 = diag1_in_plane / np.linalg.norm(diag1_in_plane)
        else:
            axis1 = d1 / np.linalg.norm(d1) if np.linalg.norm(d1) > 1e-6 else np.array([1, 0, 0])
        
        if np.linalg.norm(diag2_in_plane) > 1e-6:
            axis2 = diag2_in_plane / np.linalg.norm(diag2_in_plane)
        else:
            axis2 = d2 / np.linalg.norm(d2) if np.linalg.norm(d2) > 1e-6 else np.array([0, 1, 0])
        
        # Semi-axes lengths (half the diagonal lengths)
        semi_axis1 = np.linalg.norm(ma_p2 - ellipse_center)
        semi_axis2 = np.linalg.norm(ma_p4 - ellipse_center)
        
        # Tolerance for perpendicular distance from plane
        perp_tolerance = 10.0  # Allow 10mm perpendicular to the plane
        
        for i, pt in enumerate(self.points):
            if regions[i] != 0:
                continue
            
            v = pt - ellipse_center
            
            # Get perpendicular distance from MA plane
            perp_dist = abs(np.dot(v, ma_plane_norm))
            
            # Project point onto MA plane
            v_in_plane = v - np.dot(v, ma_plane_norm) * ma_plane_norm
            
            # Measure components along the two diagonal directions
            comp1 = np.dot(v_in_plane, axis1)
            comp2 = np.dot(v_in_plane, axis2)
            
            # Ellipse equation: (comp1/semi_axis1)^2 + (comp2/semi_axis2)^2 <= 1
            if semi_axis1 > 1e-6 and semi_axis2 > 1e-6:
                ellipse_param = (comp1 / semi_axis1) ** 2 + (comp2 / semi_axis2) ** 2
                
                # Include points within ellipse if perp_dist < 10mm
                # OR points up to ~5mm outside ellipse if perp_dist < 5mm
                if ellipse_param <= 1.0 and perp_dist <= perp_tolerance:
                    regions[i] = 5
                elif ellipse_param <= 1.1 and perp_dist <= 5.0:
                    # Allow points slightly outside ellipse (within ~5mm) if closer to plane
                    regions[i] = 5
        
        print(f"  MA: {np.sum(regions == 5)}")
        
        # =====================================================
        # 3. Define ostia rings (5mm from PV border) - FIXED to be proper rings
        # =====================================================
        print("\n3. Ostia rings (5mm from PV border)...")
        ostium_ring_width = 5.0
        
        # Store ostia centers and most anterior points for later use
        ostia_centers = {}
        ostia_anterior_pts = {}
        
        for pv, pv_rid, ost_rid in [('RSPV', 1, 13), ('LSPV', 2, 14), ('RIPV', 3, 15), ('LIPV', 4, 16)]:
            border_verts = self.get_pv_border_vertices(regions, pv_rid)
            
            if len(border_verts) == 0:
                print(f"  {pv}_Ostium: 0 (no border)")
                ostia_centers[pv] = self.markers[f'{pv}_ostium']['coords']
                ostia_anterior_pts[pv] = self.markers[f'{pv}_ostium']['coords']
                continue
            
            # Compute center of border vertices
            border_positions = np.array([self.points[v] for v in border_verts])
            ostia_centers[pv] = np.mean(border_positions, axis=0)
            
            # Compute geodesic distance from border
            dist_from_border = self.compute_geodesic_distance_from_set(border_verts)
            
            # For ostia_anterior_pts, use a default (will be computed from AP axis later)
            most_anterior_pt = ostia_centers[pv]
            ostia_anterior_pts[pv] = most_anterior_pt
            
            # Assign ostium ring - ONLY vertices within 5mm of border AND adjacent to PV or other ostium vertices
            # This prevents "outpouching" by requiring connectivity
            candidate_ostium = set()
            for i in range(len(self.points)):
                if regions[i] != 0:
                    continue
                if dist_from_border[i] <= ostium_ring_width:
                    candidate_ostium.add(i)
            
            # Keep only the connected component that touches the PV border
            if candidate_ostium:
                # Build adjacency within candidates
                adjacency = {v: set() for v in candidate_ostium}
                for v in candidate_ostium:
                    for neighbor in self.graph.neighbors(v):
                        if neighbor in candidate_ostium:
                            adjacency[v].add(neighbor)
                        elif neighbor in border_verts:
                            # Mark this candidate as touching the border
                            adjacency[v].add(-1)  # Special marker
                
                # BFS from any vertex touching the border
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
                    
                    # Assign only connected vertices
                    for v in visited:
                        regions[v] = ost_rid
                    
                    print(f"  {pv}_Ostium: {len(visited)}")
                else:
                    print(f"  {pv}_Ostium: 0 (no connection to PV)")
            else:
                print(f"  {pv}_Ostium: 0 (no candidates)")
        
        self.review_segmentation(regions)

        # =====================================================
        # 4. Define wall regions using straight lines
        # =====================================================
        print("\n4. Wall regions...")
        
        # Key points for boundaries
        rspv_ost_center = ostia_centers['RSPV']
        lspv_ost_center = ostia_centers['LSPV']
        ripv_ost_center = ostia_centers['RIPV']
        lipv_ost_center = ostia_centers['LIPV']
        
        # Most anterior points of superior PV ostia (for roof anterior border)
        rspv_anterior = ostia_anterior_pts['RSPV']
        lspv_anterior = ostia_anterior_pts['LSPV']
        
        # Compute coordinate system
        pv_center = (rspv_ost_center + lspv_ost_center + ripv_ost_center + lipv_ost_center) / 4.0
        
        # Superior-Inferior axis (from inf PVs to sup PVs)
        sup_center = (rspv_ost_center + lspv_ost_center) / 2.0
        inf_center = (ripv_ost_center + lipv_ost_center) / 2.0
        si_axis = sup_center - inf_center
        si_axis = si_axis / np.linalg.norm(si_axis)
        
        # Left-Right axis (from Left PVs to Right PVs)
        left_pv_center = (lspv_ost_center + lipv_ost_center) / 2.0
        right_pv_center = (rspv_ost_center + ripv_ost_center) / 2.0
        lr_axis = right_pv_center - left_pv_center
        lr_axis = lr_axis / np.linalg.norm(lr_axis)
        
        # Anterior-Posterior axis (perpendicular to SI/LR plane)
        ap_axis = np.cross(lr_axis, si_axis)
        ap_axis = ap_axis / np.linalg.norm(ap_axis)
        
        # Heart's center: midpoint between pv_center and ellipse_center along AP axis
        # Project both centers onto AP axis, then find midpoint
        pv_ap_component = np.dot(pv_center, ap_axis)
        ellipse_ap_component = np.dot(ellipse_center, ap_axis)
        heart_center_ap = (pv_ap_component + ellipse_ap_component) / 2.0
        
        # Construct heart center: average of pv_center and ellipse_center, then project to AP axis
        heart_center = (pv_center + ellipse_center) / 2.0
        heart_center = pv_center + (heart_center_ap - pv_ap_component) * ap_axis
        
        # ===================
        # BOUNDARY PLANES
        # ===================
        
        # POSTERIOR WALL: Quadrangle defined by 4 planes
        # Each plane is defined by 3 points forming the posterior wall boundary
        # Use the center of 4 PV ostia as reference for normal orientation
        pv_quadrangle_center = (rspv_ost_center + lspv_ost_center + ripv_ost_center + lipv_ost_center) / 4.0
        
        # Top border: plane defined by RSPV center, LSPV center, and heart center
        p1_top = rspv_ost_center
        p2_top = lspv_ost_center
        p3_top = heart_center
        v1_top = p2_top - p1_top
        v2_top = p3_top - p1_top
        post_top_plane_normal = np.cross(v1_top, v2_top)
        post_top_plane_normal = post_top_plane_normal / (np.linalg.norm(post_top_plane_normal) + 1e-10)
        # Ensure normal points toward PV quadrangle center
        to_center = pv_quadrangle_center - p1_top
        if np.dot(post_top_plane_normal, to_center) < 0:
            post_top_plane_normal = -post_top_plane_normal
        post_top_plane_pt = p1_top
        
        # Bottom border: plane defined by RIPV center, LIPV center, and heart center
        p1_bottom = ripv_ost_center
        p2_bottom = lipv_ost_center
        p3_bottom = heart_center
        v1_bottom = p2_bottom - p1_bottom
        v2_bottom = p3_bottom - p1_bottom
        post_bottom_plane_normal = np.cross(v1_bottom, v2_bottom)
        post_bottom_plane_normal = post_bottom_plane_normal / (np.linalg.norm(post_bottom_plane_normal) + 1e-10)
        # Ensure normal points toward PV quadrangle center
        to_center = pv_quadrangle_center - p1_bottom
        if np.dot(post_bottom_plane_normal, to_center) < 0:
            post_bottom_plane_normal = -post_bottom_plane_normal
        post_bottom_plane_pt = p1_bottom
        
        # Left border: plane defined by LSPV center, LIPV center, and heart center
        p1_left = lspv_ost_center
        p2_left = lipv_ost_center
        p3_left = heart_center
        v1_left = p2_left - p1_left
        v2_left = p3_left - p1_left
        post_left_plane_normal = np.cross(v1_left, v2_left)
        post_left_plane_normal = post_left_plane_normal / (np.linalg.norm(post_left_plane_normal) + 1e-10)
        # Ensure normal points toward PV quadrangle center
        to_center = pv_quadrangle_center - p1_left
        if np.dot(post_left_plane_normal, to_center) < 0:
            post_left_plane_normal = -post_left_plane_normal
        post_left_plane_pt = p1_left
        
        # Right border: plane defined by RSPV center, RIPV center, and heart center
        p1_right = rspv_ost_center
        p2_right = ripv_ost_center
        p3_right = heart_center
        v1_right = p2_right - p1_right
        v2_right = p3_right - p1_right
        post_right_plane_normal = np.cross(v1_right, v2_right)
        post_right_plane_normal = post_right_plane_normal / (np.linalg.norm(post_right_plane_normal) + 1e-10)
        # Ensure normal points toward PV quadrangle center
        to_center = pv_quadrangle_center - p1_right
        if np.dot(post_right_plane_normal, to_center) < 0:
            post_right_plane_normal = -post_right_plane_normal
        post_right_plane_pt = p1_right
        
        # ROOF anterior border: Plane defined by foremost vertices of RSPV and LSPV ostium rings and heart center
        rspv_ostium = np.where(regions == 13)[0]  # RSPV ostium ring
        lspv_ostium = np.where(regions == 14)[0]  # LSPV ostium ring
        
        rspv_anterior_vid = None
        lspv_anterior_vid = None
        
        if len(rspv_ostium) > 0:
            # Find ostium vertex with minimum component along AP axis (most anterior)
            ap_scores = np.dot(self.points[rspv_ostium] - ostia_centers['RSPV'], ap_axis)
            min_idx = np.argmin(ap_scores)
            rspv_anterior_vid = rspv_ostium[min_idx]
        
        if len(lspv_ostium) > 0:
            # Find ostium vertex with minimum component along AP axis (most anterior)
            ap_scores = np.dot(self.points[lspv_ostium] - ostia_centers['LSPV'], ap_axis)
            min_idx = np.argmin(ap_scores)
            lspv_anterior_vid = lspv_ostium[min_idx]
        
        # Create plane from foremost vertices and heart center
        roof_ant_plane_pt = None
        roof_ant_plane_normal = None
        
        if rspv_anterior_vid is not None and lspv_anterior_vid is not None:
            p1_roof_ant = self.points[rspv_anterior_vid]
            p2_roof_ant = self.points[lspv_anterior_vid]
            p3_roof_ant = heart_center
            
            v1_roof_ant = p2_roof_ant - p1_roof_ant
            v2_roof_ant = p3_roof_ant - p1_roof_ant
            roof_ant_plane_normal = np.cross(v1_roof_ant, v2_roof_ant)
            roof_ant_plane_normal = roof_ant_plane_normal / (np.linalg.norm(roof_ant_plane_normal) + 1e-10)
            
            # Ensure normal points toward anterior (away from posterior wall)
            # Use pv_quadrangle_center as reference - normal should point away from it
            to_pv_center = pv_quadrangle_center - p1_roof_ant
            if np.dot(roof_ant_plane_normal, to_pv_center) > 0:
                roof_ant_plane_normal = -roof_ant_plane_normal
            
            roof_ant_plane_pt = p1_roof_ant

        # INFERIOR: Same width as posterior, between posterior bottom and MA
        # Right border: vertical plane through RIPV center (same as posterior right extended)
        # Left border: vertical plane through LIPV center (same as posterior left extended)
        
        # ANTERIOR: Between roof anterior border and MA
        # Right border: line from RSPV ostia center to MA
        # Left border: line from LSPV ostia center to MA (or LAA to MA)
        
        # For anterior borders, we use the same right/left borders as roof
        # Plane through RSPV center, perpendicular to LR axis
        ant_right_plane_pt = rspv_ost_center
        ant_right_plane_normal = lr_axis
        
        # Plane through LSPV center, perpendicular to LR axis
        ant_left_plane_pt = lspv_ost_center
        ant_left_plane_normal = lr_axis
        
        # Anterior boundary plane: through heart center, perpendicular to AP axis
        # Prevents posterior wall from extending forward beyond this plane
        ostia_anterior_plane_pt = heart_center
        ostia_anterior_plane_normal = ap_axis
        
        unassigned = regions == 0
        
        # Calculate signed distances for all points
        dist_post_top = self.signed_distance_to_plane(self.points, post_top_plane_pt, post_top_plane_normal)
        dist_post_bottom = self.signed_distance_to_plane(self.points, post_bottom_plane_pt, post_bottom_plane_normal)
        dist_post_right = self.signed_distance_to_plane(self.points, post_right_plane_pt, post_right_plane_normal)
        dist_post_left = self.signed_distance_to_plane(self.points, post_left_plane_pt, post_left_plane_normal)
        
        # POSTERIOR WALL: Quadrangle bounded by 4 planes through PV centers and heart center
        post_mask = (unassigned &
                     (dist_post_top > 0) &      # On posterior side of top plane
                     (dist_post_bottom > 0) &   # On posterior side of bottom plane
                     (dist_post_right > 0) &    # On posterior side of right plane
                     (dist_post_left > 0))      # On posterior side of left plane
        regions[post_mask] = 7
        unassigned = regions == 0
        print(f"  Posterior: {np.sum(regions == 7)}")

        # Calculate distances for anterior/roof boundaries
        dist_ant_right = self.signed_distance_to_plane(self.points, ant_right_plane_pt, ant_right_plane_normal)
        dist_ant_left = self.signed_distance_to_plane(self.points, ant_left_plane_pt, ant_left_plane_normal)
        
        # Roof anterior boundary plane
        dist_roof_ant = np.zeros(len(self.points))
        if roof_ant_plane_pt is not None and roof_ant_plane_normal is not None:
            dist_roof_ant = self.signed_distance_to_plane(self.points, roof_ant_plane_pt, roof_ant_plane_normal)
        
        # ROOF: Between sup PVs and roof anterior line, between RSPV-LSPV width
        roof_mask = (unassigned &
                     (dist_post_top < 0) &      # Anterior of sup PVs line (opposite side from posterior wall)
                     (dist_roof_ant < 0) &      # Posterior of roof anterior plane (behind the foremost ostium vertices)
                     (dist_ant_right < 0) &     # Left of RSPV (LR axis now points left->right)
                     (dist_ant_left > 0))       # Right of LSPV (LR axis now points left->right)
        regions[roof_mask] = 8
        unassigned = regions == 0
        print(f"  Roof: {np.sum(regions == 8)}")
        
        # INFERIOR: Below posterior (below inf PVs line), same width as posterior
        inferior_mask = (unassigned &
                         (dist_post_bottom < 0) &   # Anterior of bottom plane (not in posterior interior)
                         (dist_post_right > 0) &    # Posterior side of right plane (same as posterior interior)
                         (dist_post_left > 0))      # Posterior side of left plane (same as posterior interior)
        regions[inferior_mask] = 9
        unassigned = regions == 0
        print(f"  Inferior: {np.sum(regions == 9)}")

        # ANTERIOR WALL: Defined by 3 planes
        # Plane 1: RSPV foremost vertex, MA center, heart center (right boundary)
        # Plane 2: LSPV foremost vertex, MA center, heart center (left boundary)
        # Plane 3: RSPV foremost vertex, LSPV foremost vertex, heart center (posterior boundary)
        
        ma_center = ellipse_center  # MA center from ellipse (mitral annulus)
        
        ant_plane_right_pt = None
        ant_plane_right_normal = None
        ant_plane_left_pt = None
        ant_plane_left_normal = None
        
        if rspv_anterior_vid is not None and lspv_anterior_vid is not None:
            rspv_ant_pt = self.points[rspv_anterior_vid]
            lspv_ant_pt = self.points[lspv_anterior_vid]
            
            # Right boundary plane: RSPV foremost, MA center, heart center
            p1_ant_r = rspv_ant_pt
            p2_ant_r = ma_center
            p3_ant_r = heart_center
            v1_ant_r = p2_ant_r - p1_ant_r
            v2_ant_r = p3_ant_r - p1_ant_r
            ant_plane_right_normal = np.cross(v1_ant_r, v2_ant_r)
            ant_plane_right_normal = ant_plane_right_normal / (np.linalg.norm(ant_plane_right_normal) + 1e-10)
            
            # Ensure normal points away from pv_quadrangle_center (toward anterior/lateral)
            to_pv_center = pv_quadrangle_center - p1_ant_r
            if np.dot(ant_plane_right_normal, to_pv_center) > 0:
                ant_plane_right_normal = -ant_plane_right_normal
            ant_plane_right_pt = p1_ant_r
            
            # Left boundary plane: LSPV foremost, MA center, heart center
            p1_ant_l = lspv_ant_pt
            p2_ant_l = ma_center
            p3_ant_l = heart_center
            v1_ant_l = p2_ant_l - p1_ant_l
            v2_ant_l = p3_ant_l - p1_ant_l
            ant_plane_left_normal = np.cross(v1_ant_l, v2_ant_l)
            ant_plane_left_normal = ant_plane_left_normal / (np.linalg.norm(ant_plane_left_normal) + 1e-10)
            
            # Ensure normal points away from pv_quadrangle_center (toward anterior/lateral)
            to_pv_center = pv_quadrangle_center - p1_ant_l
            if np.dot(ant_plane_left_normal, to_pv_center) > 0:
                ant_plane_left_normal = -ant_plane_left_normal
            ant_plane_left_pt = p1_ant_l
        
        # Calculate distances for anterior region
        dist_ant_right = np.zeros(len(self.points))
        dist_ant_left = np.zeros(len(self.points))
        
        if ant_plane_right_pt is not None and ant_plane_right_normal is not None:
            dist_ant_right = self.signed_distance_to_plane(self.points, ant_plane_right_pt, ant_plane_right_normal)
        
        if ant_plane_left_pt is not None and ant_plane_left_normal is not None:
            dist_ant_left = self.signed_distance_to_plane(self.points, ant_plane_left_pt, ant_plane_left_normal)
        
        # ANTERIOR: Region in front of foremost ostia, bounded by RSPV/LAA plane and LSPV/LAA plane
        anterior_mask = (unassigned &
                         (dist_roof_ant > 0) &      # Anterior of roof anterior plane (foremost ostia vertices)
                         (dist_ant_right < 0) &     # Anterior side of RSPV/MA plane
                         (dist_ant_left > 0))       # Anterior side of LSPV/MA plane
        regions[anterior_mask] = 12
        unassigned = regions == 0
        print(f"  Anterior: {np.sum(regions == 12)}")

        # SEPTAL: Right side (right of RSPV/RIPV line)
        septal_mask = (unassigned &
                       (dist_post_right < 0))       # Right of right PVs
        regions[septal_mask] = 11
        unassigned = regions == 0
        print(f"  Septal: {np.sum(regions == 11)}")
        
        # Define LAA boundary for lateral region
        laa_center = self.markers['LAA_ostium']['coords']
        dist_laa_boundary = self.signed_distance_to_plane(self.points, laa_center, lr_axis)

        # LATERAL: Between LAA and left PVs (LSPV/LIPV), down to MA
        # Right boundary: LAA center
        # Left boundary: left PV line (LSPV/LIPV)
        lateral_mask = (unassigned &
                        (dist_laa_boundary > 0) &   # Left of LAA
                        (dist_post_left < 0))       # Right of left PVs (still within the left side)
        # If nothing matches that narrow band, just take what's left of LAA
        if np.sum(lateral_mask) == 0:
            lateral_mask = (unassigned & (dist_laa_boundary > 0))
        regions[lateral_mask] = 10
        unassigned = regions == 0
        print(f"  Lateral: {np.sum(regions == 10)}")

        self.review_segmentation(regions)

        # Assign any remaining vertices
        remaining = np.sum(unassigned)
        if remaining > 0:
            print(f"\n  Assigning {remaining} remaining vertices...")
            assigned = np.where(regions > 0)[0]
            for idx in np.where(unassigned)[0]:
                dists = np.linalg.norm(self.points[assigned] - self.points[idx], axis=1)
                regions[idx] = regions[assigned[np.argmin(dists)]]

        
        # Smooth and enforce continuity
        self.smooth_boundaries(regions, iterations=3)
        self.enforce_continuity(regions)
        self.smooth_boundaries(regions, iterations=2)
        self.enforce_continuity(regions)
        
        # Verify
        print("\n  Verifying continuity...")
        all_ok = True
        for rid in range(1, 17):
            count = np.sum(regions == rid)
            if count > 0:
                components = self.find_connected_components_for_region(regions, rid)
                if len(components) > 1:
                    print(f"    WARNING: {self.extended_region_names[rid]} has {len(components)} components")
                    all_ok = False
        if all_ok:
            print("    ✓ All regions continuous")
        
        return regions
    
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
    
    def review_segmentation(self, regions):
        """Simple review without editing"""
        print("\n" + "="*60)
        print("  REVIEW SEGMENTATION")
        print("="*60)
        print("\n'a'=all, 0-9=region, 'n'=next, 'Left/Right'=prev/next, 'b'=boundaries")
        print("'p'=posterior planes, 'o'=anterior planes, 'c'=coord system, 's'=save, 'q'=quit\n")
        
        colors_arr = np.array([self.extended_color_map.get(int(r), (200,200,200)) for r in regions], dtype=np.uint8)
        
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        for c in colors_arr:
            vtk_colors.InsertNextTuple3(*c)
        
        mesh = vtk.vtkPolyData()
        mesh.DeepCopy(self.mesh)
        mesh.GetPointData().SetScalars(vtk_colors)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mesh_actor = vtk.vtkActor()
        mesh_actor.SetMapper(mapper)
        mesh_actor.GetProperty().SetRepresentationToSurface()
        mesh_actor.GetProperty().LightingOn()
        
        renderer = vtk.vtkRenderer()
        renderer.AddActor(mesh_actor)
        renderer.SetBackground(0.1, 0.1, 0.2)
        
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(1200, 900)
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        state = {'view': 'all', 'view_idx': 0, 'bounds': True, 'save': False, 'checkpoint': False, 'show_posterior_planes': False, 'show_anterior_planes': False, 'posterior_actors': [], 'anterior_actors': []}
        
        boundary = [self.create_boundary_actor(regions)]
        if boundary[0]:
            renderer.AddActor(boundary[0])
        
        text = vtk.vtkTextActor()
        text.GetTextProperty().SetFontSize(18)
        text.GetTextProperty().SetColor(1, 1, 0)
        text.SetPosition(10, 10)
        renderer.AddViewProp(text)
        
        def update_view(region=None):
            ca = np.array([self.extended_color_map.get(int(r), (200,200,200)) for r in regions], dtype=np.uint8)
            if region is None:
                state['view'] = 'all'
                for i, c in enumerate(ca):
                    vtk_colors.SetTuple3(i, *c)
                text.SetInput("VIEW: All | 's'=save | 'q'=quit")
            else:
                state['view'] = region
                for i, c in enumerate(ca):
                    if regions[i] == region:
                        vtk_colors.SetTuple3(i, *c)
                    else:
                        vtk_colors.SetTuple3(i, 50, 50, 50)
                name = self.extended_region_names[region] if region < len(self.extended_region_names) else f"R{region}"
                text.SetInput(f"VIEW: {name} ({np.sum(regions == region)}) | 'a'=all")
            mesh.GetPointData().Modified()
            window.Render()
        
        def update_bounds():
            if boundary[0]:
                renderer.RemoveActor(boundary[0])
                boundary[0] = None
            if state['bounds']:
                boundary[0] = self.create_boundary_actor(regions)
                if boundary[0]:
                    renderer.AddActor(boundary[0])
            window.Render()
        
        def toggle_posterior_planes():
            """Toggle visibility of posterior wall bounding planes"""
            state['show_posterior_planes'] = not state['show_posterior_planes']
            
            # Remove existing actors if any
            for actor in state['posterior_actors']:
                renderer.RemoveActor(actor)
            state['posterior_actors'] = []
            
            if state['show_posterior_planes']:
                # Retrieve PV centers from markers
                rspv_center = self.markers.get('RSPV_ostium', {}).get('coords')
                lspv_center = self.markers.get('LSPV_ostium', {}).get('coords')
                ripv_center = self.markers.get('RIPV_ostium', {}).get('coords')
                lipv_center = self.markers.get('LIPV_ostium', {}).get('coords')
                
                if all([rspv_center is not None, lspv_center is not None, 
                        ripv_center is not None, lipv_center is not None]):
                    
                    # Approximate heart_center and plane normals from segment data
                    # For visualization, we'll recalculate them simply
                    pv_center = (rspv_center + lspv_center + ripv_center + lipv_center) / 4.0
                    
                    # Top plane: RSPV, LSPV, pv_center
                    v1 = lspv_center - rspv_center
                    v2 = pv_center - rspv_center
                    normal_top = np.cross(v1, v2)
                    a_top, e_top = self.create_posterior_wall_plane_actors(
                        rspv_center, lspv_center, pv_center, normal_top, (1, 0, 0))  # Red
                    renderer.AddActor(a_top)
                    renderer.AddActor(e_top)
                    state['posterior_actors'].extend([a_top, e_top])
                    
                    # Bottom plane: RIPV, LIPV, pv_center
                    v1 = lipv_center - ripv_center
                    v2 = pv_center - ripv_center
                    normal_bottom = np.cross(v2, v1)
                    a_bot, e_bot = self.create_posterior_wall_plane_actors(
                        ripv_center, lipv_center, pv_center, normal_bottom, (0, 1, 0))  # Green
                    renderer.AddActor(a_bot)
                    renderer.AddActor(e_bot)
                    state['posterior_actors'].extend([a_bot, e_bot])
                    
                    # Left plane: LSPV, LIPV, pv_center
                    v1 = lipv_center - lspv_center
                    v2 = pv_center - lspv_center
                    normal_left = np.cross(v2, v1)
                    a_left, e_left = self.create_posterior_wall_plane_actors(
                        lspv_center, lipv_center, pv_center, normal_left, (0, 0, 1))  # Blue
                    renderer.AddActor(a_left)
                    renderer.AddActor(e_left)
                    state['posterior_actors'].extend([a_left, e_left])
                    
                    # Right plane: RSPV, RIPV, pv_center
                    v1 = ripv_center - rspv_center
                    v2 = pv_center - rspv_center
                    normal_right = np.cross(v1, v2)
                    a_right, e_right = self.create_posterior_wall_plane_actors(
                        rspv_center, ripv_center, pv_center, normal_right, (1, 1, 0))  # Yellow
                    renderer.AddActor(a_right)
                    renderer.AddActor(e_right)
                    state['posterior_actors'].extend([a_right, e_right])
            
            window.Render()
        
        def toggle_anterior_planes():
            """Toggle visibility of anterior wall bounding planes"""
            state['show_anterior_planes'] = not state['show_anterior_planes']
            
            # Remove existing actors if any
            for actor in state['anterior_actors']:
                renderer.RemoveActor(actor)
            state['anterior_actors'] = []
            
            if state['show_anterior_planes']:
                # Retrieve necessary data from segmentation
                rspv_center = self.markers.get('RSPV_ostium', {}).get('coords')
                lspv_center = self.markers.get('LSPV_ostium', {}).get('coords')
                ma_center_vis = self.markers.get('MA_point1', {}).get('coords')
                
                if all([rspv_center is not None, lspv_center is not None, ma_center_vis is not None]):
                    # Approximate heart_center for visualization
                    ma_p1 = self.markers.get('MA_point1', {}).get('coords')
                    ma_p2 = self.markers.get('MA_point2', {}).get('coords')
                    if ma_p1 is not None and ma_p2 is not None:
                        ma_center_vis = (ma_p1 + ma_p2) / 2.0
                    else:
                        ma_center_vis = ma_center_vis
                    
                    pv_center = (rspv_center + lspv_center) / 2.0
                    heart_center_approx = (pv_center + ma_center_vis) / 2.0
                    
                    # Plane 1 (Right): RSPV foremost, MA center, heart center
                    # For visualization, use approximated foremost RSPV
                    p1_r = rspv_center  # Approximate foremost as ostium center
                    p2_r = ma_center_vis
                    p3_r = heart_center_approx
                    v1_r = p2_r - p1_r
                    v2_r = p3_r - p1_r
                    normal_ant_right = np.cross(v1_r, v2_r)
                    a_ant_r, e_ant_r = self.create_posterior_wall_plane_actors(
                        p1_r, p2_r, p3_r, normal_ant_right, (1, 0, 1), plane_size=15)  # Magenta
                    renderer.AddActor(a_ant_r)
                    renderer.AddActor(e_ant_r)
                    state['anterior_actors'].extend([a_ant_r, e_ant_r])
                    
                    # Plane 2 (Left): LSPV foremost, MA center, heart center
                    p1_l = lspv_center  # Approximate foremost as ostium center
                    p2_l = ma_center_vis
                    p3_l = heart_center_approx
                    v1_l = p2_l - p1_l
                    v2_l = p3_l - p1_l
                    normal_ant_left = np.cross(v1_l, v2_l)
                    a_ant_l, e_ant_l = self.create_posterior_wall_plane_actors(
                        p1_l, p2_l, p3_l, normal_ant_left, (0, 1, 1), plane_size=15)  # Cyan
                    renderer.AddActor(a_ant_l)
                    renderer.AddActor(e_ant_l)
                    state['anterior_actors'].extend([a_ant_l, e_ant_l])
                    
                    # Plane 3 (Posterior boundary): RSPV foremost, LSPV foremost, heart center
                    p1_post = rspv_center  # Approximate foremost as ostium center
                    p2_post = lspv_center  # Approximate foremost as ostium center
                    p3_post = heart_center_approx
                    v1_post = p2_post - p1_post
                    v2_post = p3_post - p1_post
                    normal_ant_post = np.cross(v1_post, v2_post)
                    a_ant_post, e_ant_post = self.create_posterior_wall_plane_actors(
                        p1_post, p2_post, p3_post, normal_ant_post, (1, 1, 0), plane_size=15)  # Yellow
                    renderer.AddActor(a_ant_post)
                    renderer.AddActor(e_ant_post)
                    state['anterior_actors'].extend([a_ant_post, e_ant_post])
            
            window.Render()
        
        def update_bounds():
            if boundary[0]:
                renderer.RemoveActor(boundary[0])
                boundary[0] = None
            if state['bounds']:
                boundary[0] = self.create_boundary_actor(regions)
                if boundary[0]:
                    renderer.AddActor(boundary[0])
            window.Render()
        
        def on_key(obj, event):
            key = interactor.GetKeySym()
            
            if key == 's':
                # Save checkpoint
                base_path = os.path.splitext(self.vtk_file)[0]
                if self.save_checkpoint(base_path):
                    state['checkpoint'] = True
                    text.SetInput("VIEW: All | Checkpoint saved! Press 'q' to quit or continue reviewing")
                    text.GetTextProperty().SetColor(0, 1, 0)
                    window.Render()
            elif key == 'q':
                state['save'] = False
                window.Finalize()
                interactor.TerminateApp()
            elif key == 'b':
                state['bounds'] = not state['bounds']
                update_bounds()
            elif key == 'p':
                toggle_posterior_planes()
            elif key == 'o':
                toggle_anterior_planes()
            elif key == 'a':
                update_view(None)
            elif key == 'n' or key == 'Right':
                valid = sorted(set(regions))
                state['view_idx'] = (state['view_idx'] + 1) % len(valid)
                update_view(valid[state['view_idx']])
            elif key == 'Left':
                valid = sorted(set(regions))
                state['view_idx'] = (state['view_idx'] - 1) % len(valid)
                update_view(valid[state['view_idx']])
            elif key.isdigit():
                r = int(key)
                if r in set(regions):
                    update_view(r)
        
        interactor.AddObserver('KeyPressEvent', on_key)
        
        update_view(None)
        window.SetWindowName("LA Segmenter - Review (s=save, q=quit)")
        interactor.Initialize()
        window.Render()
        interactor.Start()
        
        return state['save']
    
    def save_results(self, regions, prefix):
        print("\n" + "="*60)
        print("  SAVING")
        print("="*60 + "\n")
        
        np.save(f"{prefix}_regions.npy", regions)
        print(f"✓ {prefix}_regions.npy")
        
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
        
        w = vtk.vtkPolyDataWriter()
        w.SetFileName(f"{prefix}_regions.vtk")
        w.SetInputData(out)
        w.Write()
        print(f"✓ {prefix}_regions.vtk")
        
        colors = np.array([self.extended_color_map.get(int(r), (200,200,200)) for r in regions], dtype=np.uint8)
        ca = vtk.vtkUnsignedCharArray()
        ca.SetNumberOfComponents(3)
        ca.SetName("Colors")
        for c in colors:
            ca.InsertNextTuple3(*c)
        
        ply = vtk.vtkPolyData()
        ply.DeepCopy(self.mesh)
        ply.GetPointData().SetScalars(ca)
        
        pw = vtk.vtkPLYWriter()
        pw.SetFileName(f"{prefix}_regions.ply")
        pw.SetInputData(ply)
        pw.Write()
        print(f"✓ {prefix}_regions.ply")
        
        with open(f"{prefix}_landmarks.txt", 'w') as f:
            f.write("Region,Type,VertexID,X,Y,Z,Radius\n")
            for k, v in sorted(self.markers.items()):
                parts = k.rsplit('_', 1)
                reg, typ = parts[0], parts[1]
                c = v['coords']
                vid = v.get('point_id', -1) or -1
                rad = v.get('radius', 0) or 0
                f.write(f"{reg},{typ},{vid},{c[0]:.4f},{c[1]:.4f},{c[2]:.4f},{rad:.2f}\n")
        print(f"✓ {prefix}_landmarks.txt")
    
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
        else:
            # Normal flow: load mesh, select landmarks
            self.load_mesh()
            self.center_mesh()
            self.build_graph()
            self.select_landmarks_interactive()
        
        # Define regions (after checkpoint load or landmark selection)
        regions = self.define_regions()
        
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        for i, n in enumerate(self.extended_region_names):
            c = np.sum(regions == i)
            if c > 0:
                print(f"  {i:2d}. {n:20s}: {c:6d}")
        
        save = self.review_segmentation(regions)
        
        if save:
            self.save_results(regions, self.vtk_file.replace('.vtk', ''))
            print("\n✓ DONE!")
        else:
            print("\n✗ Not saved")
        
        return regions if save else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', nargs='?')
    args = parser.parse_args()
    
    if args.file is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            f = filedialog.askopenfilename(filetypes=[("VTK", "*.vtk"), ("Checkpoint", "*.wrk")])
            if not f:
                return
            args.file = f
        except:
            print("Usage: python LAsegmenter.py [file.vtk|file.wrk]")
            return
    
    LASegmenter(args.file).run()


if __name__ == '__main__':
    main()