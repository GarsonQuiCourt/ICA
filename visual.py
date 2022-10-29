import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import multiprocessing as mp
import itertools

def xyz(fi, teta, R=1):
    fi = np.array(fi)
    theta = np.array(teta)
    if type(R) is int:
        R = np.array(R).reshape(1)
    if len(fi.shape) == 1:
        r = np.stack([np.cos(teta) * np.sin(fi), np.cos(teta) * np.cos(fi), np.sin(teta)], axis=-1).reshape(1, -1, 3)
    elif len(fi.shape) == 2:
        r = np.stack([np.cos(teta) * np.sin(fi), np.cos(teta) * np.cos(fi), np.sin(teta)], axis=-1)
    else:
        r = np.concatenate([np.cos(teta) * np.sin(fi), np.cos(teta) * np.cos(fi), np.sin(teta)], axis=-1)
    return r * R.reshape(R.shape[0], -1, 1)

def random_v(num):
    fi = np.random.uniform(0, 2*np.pi, size=num)
    teta = np.random.uniform(-np.pi, np.pi, size=num)
    return xyz(fi, teta)

def periodic(num, ticks, trend=0, amplitude=1, period=1):
    trend = np.array(trend).reshape(1, -1, 1)
    amplitude = np.array(amplitude).reshape(1, -1, 1)
    period = np.array(period).reshape(1, -1, 1)

    fi = np.random.uniform(0, np.pi * 2, size=(1, num, 1)) + period * np.pi * 2 * np.linspace(0, 1, ticks).reshape((ticks, 1, 1))
    return trend * np.linspace(0, 1, ticks).reshape((ticks, 1, 1)) + amplitude * np.sin(fi)

class Space:
    class Dipole:
        def __init__(self, _dipoles, r, Q):
            self._dipoles = _dipoles
            self.r = r.reshape(-1, 1, 3)
            self.Q = Q.reshape(-1, 1, 3)

        def visual(dipole, tick, name, coloraxis, showlegend, Q_max):
            return go.Cone(
                x=dipole.r[tick, :, 0],
                y=dipole.r[tick, :, 1],
                z=dipole.r[tick, :, 2],
                u=dipole.Q[tick, :, 0],
                v=dipole.Q[tick, :, 1],
                w=dipole.Q[tick, :, 2],
                name=name,
                showlegend=showlegend,
                coloraxis=coloraxis,
                sizemode='absolute',
                sizeref=np.linalg.norm(dipole.Q[tick]) / Q_max * 0.2
            )
    
    class Dipoles:
        def __init__(self, space):
            self.space = space
            self._dipoles = {}
            self.Q_max = {}
        
        def __iter__(self):
            return iter([dipole for dipoles_group in self._dipoles.values() for dipole in dipoles_group])
        
        def __getitem__(self, key):
            return self._dipoles[key]
        
        def __len__(self):
            return len([dipole for dipoles_group in self._dipoles.values() for dipole in dipoles_group])

        def add(self, r, Q, name='Dipoles'):
            r = np.stack([r for _ in range(self.space.ticks)]) if type(r) is list or len(r.shape) != 3 or r.shape[0] != self.space.ticks else r
            r = r.reshape(self.space.ticks, -1, 3)

            Q = np.stack([Q for _ in range(self.space.ticks)]) if type(Q) is list or len(Q.shape) != 3 or Q.shape[0] != self.space.ticks else Q
            Q = Q.reshape(self.space.ticks, -1, 3)
            self.Q_max[name] = max(self.Q_max.get(name, 0), np.linalg.norm(Q, axis=-1).max())

            with mp.Pool() as pool:
                new_dipoles = pool.starmap(Space.Dipole.__call__, zip([self for _ in range(r.shape[1])], np.swapaxes(r, 0, 1), np.swapaxes(Q, 0, 1)))
            
            if name in self._dipoles:
                self._dipoles[name].extend(new_dipoles)
            else:
                self._dipoles[name] = new_dipoles
        
        def add_random(self, num=1, name='Dipoles', 
                       races=True, 
                       twinkling=True,
                       period_max=5,
                       R_lim=(0.8, 0.9),
                       R_amplitude=0.05,
                       gamma=100,
                       Qdr=0):
            if races:
                fi = periodic(num, self.space.ticks + 1, 
                              np.random.uniform(0, 2 * np.pi, num), 
                              np.random.uniform(0, 2 * np.pi / gamma, num), 
                              np.random.uniform(0, period_max, num)) + np.random.uniform(0, 2 * np.pi, (1, num, 1))
                
                teta = periodic(num, self.space.ticks + 1, 
                                np.random.uniform(0, 2 * np.pi, num), 
                                np.random.uniform(0, 2 * np.pi / gamma, num),
                                np.random.uniform(0, period_max, num)) + np.random.uniform(0, 2 * np.pi, (1, num, 1))
                
                R = np.random.uniform(*R_lim, (1, num, 1))
                R = R + periodic(num, self.space.ticks + 1, 
                                 0, 
                                 np.random.uniform(0, R_amplitude, num), 
                                 np.random.uniform(0, period_max, num))
                r = xyz(fi, teta, R)
                if twinkling:
                    dQ = periodic(num, self.space.ticks, 0, 1, 
                                             np.random.uniform(0, period_max, num)).reshape(self.space.ticks, num, 1)
                    Q = (r[1:] - r[:-1]) / np.linalg.norm(r[1:] - r[:-1], axis=-1).reshape(self.space.ticks, num, 1) * (Qdr + dQ)
                else: 
                    Q = (r[1:] - r[:-1]) / np.linalg.norm(r[1:] - r[:-1], axis=-1).reshape(self.space.ticks, num, 1)
                
                r = r[:-1]
            elif twinkling:
                r = random_v(num) * np.random.uniform(*R_lim, (1, num, 1))
                Q = random_v(num) * periodic(num, self.space.ticks, 0, 1, 
                                             np.random.uniform(0, period_max, num))
            else:
                r = random_v(num) * np.random.uniform(*R_lim, (1, num, 1))
                Q = random_v(num) * np.random.uniform(0, 1, (1, num, 1))
            
            self.add(r, Q, name)

        def visual(self, tick, coloraxis={}, showlegend={}, Q_max_general=True):
            if len(coloraxis) != len(self._dipoles):
                coloraxis = {name : 'coloraxis1' for name in self._dipoles.keys() if not name in coloraxis}

            if len(showlegend) != len(self._dipoles):
                showlegend = {name : False for name in self._dipoles.keys() if not name in showlegend}
            
            data = []
            for name, dipoles in self._dipoles.items():
                for dipole in dipoles:
                    data.append(Space.Dipole.visual(dipole, tick, name, coloraxis[name], showlegend[name], max(self.Q_max.values()) if Q_max_general else self.Q_max[name]))
            # with mp.Pool() as pool:
            #     data = pool.starmap(Space.Dipole.visual,
            #         [(dipole, tick, name, coloraxis[name], showlegend[name], max(self.Q_max.values()) if Q_max_general else self.Q_max[name]) 
            #             for name in self._dipoles.keys() for dipole in self._dipoles[name]])

            return data
    
    
    class Points:
        def __init__(self, _points, r, B, projection_function=None, size_min=0, size_max=6):
            self._points = _points

            self.r = r
            self.B = B
            
            self.projection_function = projection_function if not projection_function is None else Space.Points.radial_projection

            self.B_projection = None
            self.B_max = 0

            self.size_min = size_min
            self.size_max = size_max
        
        def update_B_projection(self):
            self.B_projection = self.projection_function(self.B, self.r)
            self.B_max = self.B_projection.max()
        
        def extend(self, r, B):
            self.r = np.concatenate([self.r, r], axis=1)
            self.B = np.concatenate([self.B, B], axis=1)
        
        def radial_projection(B, r):
            return np.sum(B * r, axis=-1)
        
        def scale(B, B_max, B_min=0, size_max=0, size_min=6):
            return ((B - B_min) / (B_max - B_min + 1e-25)) * (size_max - size_min) + size_min
        
        def visual(points, tick, name, coloraxis={}, showlegend={}, B_max=True):
            if B_max is None: B_max = points.B_max
            return go.Scatter3d(
                        x=points.r[tick, :, 0],
                        y=points.r[tick, :, 1],
                        z=points.r[tick, :, 2],
                        name=name,
                        showlegend=showlegend,
                        mode='markers',
                        marker=dict(
                            size=Space.Points.scale(np.abs([points.B_projection[tick]]), B_max, 0, points.size_max, points.size_min).reshape(-1),
                            color=np.sign(points.B_projection[tick]).reshape(-1), 
                            colorscale='bluered', 
                            opacity=0.8
                        )
                    )
        
        def calculate_B(points_r, dipole_r, dipole_Q):
            dr = points_r - dipole_r
            return np.cross(dipole_Q, dr) / np.linalg.norm(dr, axis=2).reshape(-1, points_r.shape[1], 1) ** 3
            

    class PointsGroups:
        def __init__(self, space):
            self.space = space
            self._points = {}
            self.B_max = 0
        
        def __iter__(self):
            return iter(self._points.values())
        
        def __len__(self):
            return len(self._points)

        def __getitem__(self, key):
            return self._points[key]
        
        def add(self, r, name='Points', size_min=1, size_max=6, projection_function=None, B=None):
            r = np.stack([r for _ in range(self.space.ticks)]) if type(r) is list or len(r.shape) != 3 or r.shape[0] != self.space.ticks else r
            r = r.reshape(self.space.ticks, -1, 3)

            B = np.zeros((r.shape[1], 3)) if B is None else B
            B = np.stack([B for _ in range(self.space.ticks)]) if type(B) is list or len(B.shape) != 3 or B.shape[0] != self.space.ticks else B
            B = B.reshape(self.space.ticks, -1, 3)

            if name in self._points:
                self._points[name].extend(r, B)
            else:
                self._points[name] = Space.Points(self, r, B, projection_function, size_min=size_min, size_max=size_max)
        
        def add_sphere(self, num_pts=1000, R=1, size_min=1, size_max=6, name='Sphere'):
            indices = np.arange(1, num_pts - 1)

            phi = np.arccos(1 - 2 * np.hstack([0, (indices + 6) / (num_pts + 11), 1]))
            theta = np.pi * (1 + 5 ** 0.5) * np.hstack([0, indices, 0])

            sphere = np.vstack([R * np.cos(theta) * np.sin(phi), R * np.sin(theta) * np.sin(phi), R * np.cos(phi)]).T.reshape(1, -1, 3)

            self.add(sphere, name = name, size_min = size_min, size_max = size_max)
        
        def update_B(self):
            for points in self._points.values():
                with mp.Pool() as pool:
                    B_addict = pool.starmap(Space.Points.calculate_B,
                                        zip([points.r for _ in range(len(self.space.dipoles))],
                                            [dipole.r for dipole in self.space.dipoles],
                                            [dipole.Q for dipole in self.space.dipoles]))
                points.B = np.sum(B_addict, axis=0)
                points.update_B_projection()
                self.B_max = max(self.B_max, points.B_max)
            
        
        def visual(self, tick, coloraxis={}, showlegend={}, B_max_general=True):
            if len(coloraxis) != len(self._points):
                coloraxis = {name : 'coloraxis1' for name in self._points.keys() if not name in coloraxis}

            if len(showlegend) != len(self._points):
                showlegend = {name : True for name in self._points.keys() if not name in showlegend}

            data = []
            for name in self._points.keys():
                data.append(Space.Points.visual(self._points[name], tick, name, coloraxis[name], showlegend[name], self.B_max if B_max_general else None))
            
            # with mp.Pool() as pool:
            #     data = pool.starmap(Space.Points.visual,
            #         [(self._points[name], tick, name, coloraxis[name], showlegend[name], self.B_max if B_max_general else None) 
            #             for name in self._points.keys()])
            return data
    
    class Trackers(PointsGroups):
        def track_vis(points, tick, name, coloraxis={}, showlegend={}, B_max=True, size=2):
            if B_max is None: B_max = points.B_max
            return go.Scatter3d(
                        x=points.r[tick, :, 0],
                        y=points.r[tick, :, 1],
                        z=points.r[tick, :, 2],
                        name=name,
                        showlegend=showlegend,
                        mode='markers',
                        text=[f'i: {i}  B: {B:.3f}' for i, B in enumerate(points.B_projection[tick])],
                        marker=dict(
                            size=size,
                            color=np.sign(points.B_projection[tick]).reshape(-1), 
                            colorscale='bluered', 
                            symbol='x',
                            opacity=0.8,
                        )
                    )
        
        def visual(self, tick, coloraxis={}, showlegend={}, B_max_general=True):
            if len(coloraxis) != len(self._points):
                coloraxis = {name : 'coloraxis1' for name in self._points.keys() if not name in coloraxis}

            if len(showlegend) != len(self._points):
                showlegend = {name : True for name in self._points.keys() if not name in showlegend}

            data = []
            for name in self._points.keys():
                data.append(Space.Trackers.track_vis(self._points[name], tick, name, coloraxis[name], showlegend[name], self.B_max if B_max_general else None))
            return data
        
        def plot_B(self, nt=10):
            n = sum(points.B_projection.shape[1] + 1 for points in self._points.values())
            m, k = 1, 0
            fig = plt.figure(figsize=(10, 10), dpi=100)
            fig.subplots_adjust(hspace=0)
            dt = int(self.space.ticks / nt)

            for name, points in self._points.items():
                k += 1
                ax = plt.subplot(n, m, k)
                ax.text(0.5, 0.5, f'{name}, B_max={points.B_max}', ha='center', va='center')
                ax.set_yticks([])
                ax.set_xticks([])

                for i, B in enumerate(points.B_projection.T):
                    k += 1
                    ax = plt.subplot(n, m, k)

                    ax.plot(B)
                    ax.set_yticks([])
                    ax.set_ylabel(i, fontsize='xx-small')
                    ax.set_ylim(-points.B_max * 1.15, points.B_max * 1.15)
                    ax.set_xticks(range(dt, self.space.ticks, dt))
                    if i != points.B_projection.shape[1] - 1:
                        ax.set_xticklabels([])
                    ax.grid()
                    ax.set_xlim(0, self.space.ticks-1)

                
    
    def __init__(self, ticks=60):
        self.ticks = ticks

        self.dipoles = Space.Dipoles(self)
        
        self.points = Space.PointsGroups(self)
        self.trackers = Space.Trackers(self)
        self.B_max = 0
    
    def visual(space, tick):
        data = []
        if space.dipoles_F and len(space.dipoles) > 0:
            data.extend(space.dipoles.visual(tick, space.dipoles_coloraxis, space.dipoles_showlegend, space.Q_max_global))
        if space.points_F and len(space.points) > 0:
            data.extend(space.points.visual(tick, space.points_coloraxis, space.points_showlegend, space.B_max_global))
        if space.trackers_F and len(space.points) > 0:
            data.extend(space.trackers.visual(tick, space.points_coloraxis, space.points_showlegend, space.B_max_global))
        return data
    
    def animate(self,
                dipoles_F=True,
                dipoles_coloraxis={}, 
                dipoles_showlegend={}, 
                Q_max_global=True, 
                points_F=True,
                points_coloraxis={}, 
                points_showlegend={}, 
                B_max_global=True,
                trackers_F=True):
        self.dipoles_F = dipoles_F
        self.dipoles_coloraxis = dipoles_coloraxis if not dipoles_coloraxis is {} else {name: 'coloraxis1' for name in self.dipoles._dipoles.keys()} 
        self.dipoles_showlegend = dipoles_showlegend if not dipoles_showlegend is {} else {name: False for name in self.dipoles._dipoles.keys()} 
        self.Q_max_global = Q_max_global

        self.points_F = points_F
        self.points_coloraxis = points_coloraxis if not points_coloraxis is {} else {name: 'coloraxis1' for name in self.points._points.keys()} 
        self.points_showlegend = points_showlegend if not points_showlegend is {} else {name: True for name in self.points._points.keys()} 
        self.B_max_global = B_max_global

        self.trackers_F = trackers_F

        fig = go.Figure()
        frames = []
        with mp.Pool() as pool:
            data = pool.map(self.visual, range(self.ticks))

        for obj in data[0]:
            fig.add_trace(obj)

        for tick in range(self.ticks):
            frames.append(go.Frame(data=data[tick], name=str(tick)))

        fig.frames=frames

        def frame_args(duration):
            return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration, "easing": "linear"},
                }

        sliders = [
                    {
                        "pad": {"b": 10, "t": 60},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": [
                            {
                                "args": [[f.name], frame_args(0)],
                                "label": str(k),
                                "method": "animate",
                            }
                            for k, f in enumerate(fig.frames)
                        ],
                    }
                ]

        fig.update_layout(
                title="",
                width = 800,
                height = 800,
                legend_orientation="v",
                legend_x=0,
                scene=dict(
                            xaxis_range = (-1.2, 1.2),
                            yaxis_range = (-1.2, 1.2),
                            zaxis_range = (-1.2, 1.2),
                            aspectratio_x = 1,
                            aspectratio_y = 1,
                            aspectratio_z = 1
                            ),
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "&#9654;",
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;",
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
        )
        fig.update_coloraxes(cmin=0, cmax=max(self.dipoles.Q_max.values()))

        return fig
