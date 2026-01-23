#!/usr/bin/env python3
"""
Left Atrium Segmenter - Version 19
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
            ('MA', 'point1', 'MITRAL ANNULUS - Point 1 on rim'),
            ('MA', 'point2', 'MITRAL ANNULUS - Point 2 on rim'),
            ('MA', 'point3', 'MITRAL ANNULUS - Point 3 on rim'),
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
    
    def select_landmarks_interactive(self):
        print("\n" + "="*60)
        print("  LANDMARK SELECTION")
        print("="*60)
        print("\nVEINS: Click tip, then adjust plane:")
        print("  W/S: Tilt forward/back | A/D: Tilt left/right")
        print("  UP/DOWN: Move plane | +/-: Radius | R: Reset")
        print("  SPACE: Confirm")
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
        
        def compute_tilted_normal():
            base = state['base_normal'].copy()
            
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
        
        def update_plane():
            if state['plane_pos'] is None:
                return
            
            region = self.landmark_sequence[state['idx']][0]
            
            if state['ring']:
                renderer.RemoveActor(state['ring'])
            if state['disk']:
                renderer.RemoveActor(state['disk'])
            
            state['plane_normal'] = compute_tilted_normal()
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
                text.SetInput(f"[{state['idx']+1}/{len(self.landmark_sequence)}] WASD=tilt, UP/DOWN=move, SPACE=confirm")
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
                if key == 'w':
                    state['tilt_fb'] += 5
                    update_plane()
                    return
                elif key == 's':
                    state['tilt_fb'] -= 5
                    update_plane()
                    return
                elif key == 'a':
                    state['tilt_lr'] -= 5
                    update_plane()
                    return
                elif key == 'd':
                    state['tilt_lr'] += 5
                    update_plane()
                    return
                elif key == 'Up':
                    state['offset'] += 2
                    update_plane()
                    return
                elif key == 'Down':
                    state['offset'] = max(0, state['offset'] - 2)
                    update_plane()
                    return
                elif key in ['plus', 'equal']:
                    state['radius'] += 1
                    update_plane()
                    return
                elif key == 'minus':
                    state['radius'] = max(3, state['radius'] - 1)
                    update_plane()
                    return
                elif key == 'r':
                    state['offset'] = 0
                    state['tilt_fb'] = 0
                    state['tilt_lr'] = 0
                    state['radius'] = self.default_ostium_radius
                    update_plane()
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
                    if self.landmark_sequence[state['idx']][0] == 'MA':
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
    
    def point_to_line_distance_signed(self, points, line_pt1, line_pt2, ref_normal):
        """
        Calculate signed distance from points to a line in 3D.
        The sign is determined by which side of the line the point falls on,
        using ref_normal to define the plane containing the line.
        """
        line_dir = line_pt2 - line_pt1
        line_dir = line_dir / np.linalg.norm(line_dir)
        
        # Normal to the "side" plane (perpendicular to line, in the ref_normal plane)
        side_normal = np.cross(line_dir, ref_normal)
        if np.linalg.norm(side_normal) > 1e-6:
            side_normal = side_normal / np.linalg.norm(side_normal)
        else:
            # Fallback
            side_normal = np.cross(line_dir, [0, 0, 1])
            if np.linalg.norm(side_normal) < 1e-6:
                side_normal = np.cross(line_dir, [0, 1, 0])
            side_normal = side_normal / np.linalg.norm(side_normal)
        
        # Signed distance is dot product with side_normal
        return np.dot(points - line_pt1, side_normal)
    
    def fit_ellipse_from_3_points(self, p1, p2, p3):
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)
        center = (p1 + p2 + p3) / 3.0
        
        pts_2d = []
        for p in [p1, p2, p3]:
            v = p - center
            v -= np.dot(v, normal) * normal
            pts_2d.append(v)
        
        dists = [np.linalg.norm(p) for p in pts_2d]
        semi_major = max(dists)
        semi_minor = min(dists) if min(dists) > 0.5 * max(dists) else 0.7 * max(dists)
        
        major_axis = pts_2d[np.argmax(dists)]
        if np.linalg.norm(major_axis) > 1e-6:
            major_axis /= np.linalg.norm(major_axis)
        minor_axis = np.cross(normal, major_axis)
        if np.linalg.norm(minor_axis) > 1e-6:
            minor_axis /= np.linalg.norm(minor_axis)
        
        return center, semi_major, semi_minor, major_axis, minor_axis, normal
    
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
        ma_center, ma_smaj, ma_smin, ma_maj, ma_min, ma_norm = self.fit_ellipse_from_3_points(ma_p1, ma_p2, ma_p3)
        
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
            
            # Find candidates on distal side within radius
            candidates = np.zeros(len(self.points), dtype=bool)
            for i, pt in enumerate(self.points):
                if np.sign(np.dot(pt - center, normal)) == distal_side:
                    v = pt - center
                    radial = np.linalg.norm(v - np.dot(v, normal) * normal)
                    if radial < radius * 1.5:
                        candidates[i] = True
            
            # Keep only connected component containing tip
            connected = self.find_connected_component(tip_id, candidates)
            
            for vid in connected:
                regions[vid] = rid
            
            print(f"  {pv}: {len(connected)}")
        
        # =====================================================
        # 2. Define MA region - SMALLER, bounded by the 3 picked points
        # =====================================================
        print("\n2. MA region (bounded by picked points)...")
        
        # Compute MA plane normal
        v1, v2 = ma_p2 - ma_p1, ma_p3 - ma_p1
        ma_plane_norm = np.cross(v1, v2)
        ma_plane_norm = ma_plane_norm / np.linalg.norm(ma_plane_norm)
        
        # Find max distance from center to any of the 3 marking points (in-plane)
        ma_pts = [ma_p1, ma_p2, ma_p3]
        max_dist_from_center = 0
        for p in ma_pts:
            v = p - ma_center
            v_in_plane = v - np.dot(v, ma_plane_norm) * ma_plane_norm
            dist = np.linalg.norm(v_in_plane)
            if dist > max_dist_from_center:
                max_dist_from_center = dist
        
        # MA region: within the radius defined by the 3 points
        margin = 4.0  # Small margin from MA plane
        for i, pt in enumerate(self.points):
            if regions[i] != 0:
                continue
            v = pt - ma_center
            # Check distance from MA plane
            if abs(np.dot(v, ma_plane_norm)) > margin:
                continue
            # Check if within radius defined by the 3 marking points
            v_in_plane = v - np.dot(v, ma_plane_norm) * ma_plane_norm
            dist_from_center = np.linalg.norm(v_in_plane)
            if dist_from_center <= max_dist_from_center:
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
            
            # Get direction toward MA for "anterior" definition
            to_ma = ma_center - ostia_centers[pv]
            to_ma = to_ma / np.linalg.norm(to_ma)
            
            # Find most anterior border vertex
            max_anterior = -np.inf
            most_anterior_pt = ostia_centers[pv]
            for v in border_verts:
                anterior_score = np.dot(self.points[v] - ostia_centers[pv], to_ma)
                if anterior_score > max_anterior:
                    max_anterior = anterior_score
                    most_anterior_pt = self.points[v].copy()
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
        
        # Anterior-Posterior axis (from PV center to MA)
        ap_axis = ma_center - pv_center
        ap_axis = ap_axis - np.dot(ap_axis, si_axis) * si_axis  # Make orthogonal to SI
        ap_axis = ap_axis / np.linalg.norm(ap_axis)
        
        # Left-Right axis (cross product)
        lr_axis = np.cross(si_axis, ap_axis)
        lr_axis = lr_axis / np.linalg.norm(lr_axis)
        
        # Make sure LR axis points from right to left
        if np.dot(lr_axis, lspv_ost_center - rspv_ost_center) < 0:
            lr_axis = -lr_axis
        
        # ===================
        # BOUNDARY PLANES
        # ===================
        
        # POSTERIOR WALL: Rectangle between 4 PV ostia centers
        # Top border: line between RSPV and LSPV ostia centers
        post_top_plane_pt = sup_center
        post_top_plane_normal = si_axis
        
        # Bottom border: line between RIPV and LIPV ostia centers
        post_bottom_plane_pt = inf_center
        post_bottom_plane_normal = si_axis
        
        # Right border: line between RSPV and RIPV ostia centers
        right_pv_center = (rspv_ost_center + ripv_ost_center) / 2.0
        post_right_plane_normal = lr_axis
        post_right_plane_pt = right_pv_center
        
        # Left border: line between LSPV and LIPV ostia centers
        left_pv_center = (lspv_ost_center + lipv_ost_center) / 2.0
        post_left_plane_normal = lr_axis
        post_left_plane_pt = left_pv_center
        
        # ROOF anterior border: line between most anterior points of RSPV and LSPV ostia
        roof_ant_center = (rspv_anterior + lspv_anterior) / 2.0
        roof_ant_plane_pt = roof_ant_center
        roof_ant_plane_normal = ap_axis  # Points anterior, < 0 means posterior (roof side)
        
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
        
        # Calculate signed distances for all points
        dist_post_top = self.signed_distance_to_plane(self.points, post_top_plane_pt, post_top_plane_normal)
        dist_post_bottom = self.signed_distance_to_plane(self.points, post_bottom_plane_pt, post_bottom_plane_normal)
        dist_post_right = self.signed_distance_to_plane(self.points, post_right_plane_pt, post_right_plane_normal)
        dist_post_left = self.signed_distance_to_plane(self.points, post_left_plane_pt, post_left_plane_normal)
        dist_roof_ant = self.signed_distance_to_plane(self.points, roof_ant_plane_pt, roof_ant_plane_normal)
        dist_ant_right = self.signed_distance_to_plane(self.points, ant_right_plane_pt, ant_right_plane_normal)
        dist_ant_left = self.signed_distance_to_plane(self.points, ant_left_plane_pt, ant_left_plane_normal)
        
        unassigned = regions == 0
        
        # POSTERIOR WALL: Rectangle between 4 PV ostia centers
        post_mask = (unassigned &
                     (dist_post_top < 0) &      # Below sup PVs
                     (dist_post_bottom > 0) &   # Above inf PVs
                     (dist_post_right > 0) &    # Left of right PVs
                     (dist_post_left < 0))      # Right of left PVs
        regions[post_mask] = 7
        unassigned = regions == 0
        print(f"  Posterior: {np.sum(regions == 7)}")
        
        # ROOF: Between sup PVs and roof anterior line, between RSPV-LSPV width
        roof_mask = (unassigned &
                     (dist_post_top > 0) &      # Above sup PVs line
                     (dist_roof_ant < 0) &      # Posterior of roof anterior line
                     (dist_ant_right > 0) &     # Left of RSPV
                     (dist_ant_left < 0))       # Right of LSPV
        regions[roof_mask] = 8
        unassigned = regions == 0
        print(f"  Roof: {np.sum(regions == 8)}")
        
        # INFERIOR: Below posterior (below inf PVs line), same width as posterior
        inferior_mask = (unassigned &
                         (dist_post_bottom < 0) &   # Below inf PVs
                         (dist_post_right > 0) &    # Left of right PVs (same as posterior)
                         (dist_post_left < 0))      # Right of left PVs (same as posterior)
        regions[inferior_mask] = 9
        unassigned = regions == 0
        print(f"  Inferior: {np.sum(regions == 9)}")
        
        # ANTERIOR: Anterior of roof line, between RSPV and LAA (not LSPV)
        # LAA marks the left boundary of anterior, lateral is to the left of LAA
        laa_center = self.markers['LAA_ostium']['coords']
        dist_laa_boundary = self.signed_distance_to_plane(self.points, laa_center, lr_axis)
        
        anterior_mask = (unassigned &
                         (dist_roof_ant > 0) &      # Anterior of roof anterior line
                         (dist_ant_right > 0) &     # Left of RSPV
                         (dist_laa_boundary < 0))   # Right of LAA (not LSPV)
        regions[anterior_mask] = 12
        unassigned = regions == 0
        print(f"  Anterior: {np.sum(regions == 12)}")
        
        # SEPTAL: Right side (right of RSPV/RIPV line)
        septal_mask = (unassigned &
                       (dist_post_right < 0))       # Right of right PVs
        regions[septal_mask] = 11
        unassigned = regions == 0
        print(f"  Septal: {np.sum(regions == 11)}")
        
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
    
    def review_segmentation(self, regions):
        """Simple review without editing"""
        print("\n" + "="*60)
        print("  REVIEW SEGMENTATION")
        print("="*60)
        print("\n'a'=all, 0-9=region, 'n'/'p'=next/prev, 'b'=boundaries")
        print("'s'=save, 'q'=quit\n")
        
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
        
        renderer = vtk.vtkRenderer()
        renderer.AddActor(mesh_actor)
        renderer.SetBackground(0.1, 0.1, 0.2)
        
        window = vtk.vtkRenderWindow()
        window.AddRenderer(renderer)
        window.SetSize(1200, 900)
        
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(window)
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        
        state = {'view': 'all', 'view_idx': 0, 'bounds': True, 'save': False}
        
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
        
        def on_key(obj, event):
            key = interactor.GetKeySym()
            
            if key == 's':
                state['save'] = True
                window.Finalize()
                interactor.TerminateApp()
            elif key == 'q':
                state['save'] = False
                window.Finalize()
                interactor.TerminateApp()
            elif key == 'b':
                state['bounds'] = not state['bounds']
                update_bounds()
            elif key == 'a':
                update_view(None)
            elif key == 'n':
                valid = sorted(set(regions))
                state['view_idx'] = (state['view_idx'] + 1) % len(valid)
                update_view(valid[state['view_idx']])
            elif key == 'p':
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
        self.load_mesh()
        self.center_mesh()
        self.build_graph()
        self.select_landmarks_interactive()
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
    parser.add_argument('vtk_file', nargs='?')
    args = parser.parse_args()
    
    if args.vtk_file is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            tk.Tk().withdraw()
            f = filedialog.askopenfilename(filetypes=[("VTK", "*.vtk")])
            if not f:
                return
            args.vtk_file = f
        except:
            print("Usage: python la_segmenter_v17.py file.vtk")
            return
    
    LASegmenter(args.vtk_file).run()


if __name__ == '__main__':
    main()