"""Static support geometry, distinct from obstacles and neural observations."""
from dataclasses import dataclass
import re

import numpy as np


@dataclass(frozen=True)
class TerrainPatch:
    name: str
    center: tuple[float, float]
    half_extents: tuple[float, float]
    height: float
    kind: str = 'box'
    yaw: float = 0.
    base_height: float = 0.
    friction: float = .8

    def validate(self):
        if not isinstance(self.name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', self.name):
            raise ValueError('Terrain name must be a unique simple identifier')
        if np.shape(self.center) != (2,) or np.shape(self.half_extents) != (2,):
            raise ValueError('Terrain center and half extents must be 2D')
        if self.kind not in ('box', 'ramp'):
            raise ValueError('Terrain kind must be box or ramp')
        values = [*self.center, *self.half_extents, self.height, self.yaw, self.base_height, self.friction]
        if (not np.isfinite(values).all() or min(*self.half_extents, self.height, self.friction) <= 0
                or self.base_height < 0):
            raise ValueError('Invalid finite terrain dimensions, height or friction')

    def surface_height(self, x, y):
        """Top surface at a world point, or None outside the patch footprint."""
        cs, sn = np.cos(self.yaw), np.sin(self.yaw)
        delta = np.array([x, y])-self.center
        local = np.array([[cs, sn], [-sn, cs]]) @ delta
        hx, hy = self.half_extents
        if abs(local[0]) > hx+1e-10 or abs(local[1]) > hy+1e-10:
            return None
        rise = self.height if self.kind == 'box' else self.height*(local[0]+hx)/(2*hx)
        return float(self.base_height+rise)


def add_terrain(world, assets, patches, foot_defaults):
    """Compile boxes and convex wedge meshes with actual collidable top faces."""
    import xml.etree.ElementTree as ET
    for patch in patches:
        patch.validate()
        hx, hy = patch.half_extents
        attributes = dict(name='terrain_'+patch.name,
            quat=f'{np.cos(patch.yaw/2)} 0 0 {np.sin(patch.yaw/2)}',
            friction=f'{patch.friction} .02 .01', priority='2', condim='6',
            rgba=('.30 .65 .38 1' if patch.kind == 'ramp' else
                  '.25 .65 .85 1' if patch.friction < .6 else '.63 .51 .34 1'))
        # Higher contact priority is needed for local friction; retain the foot
        # solver softness so priority does not also change that parameter.
        for parameter in ('solimp', 'solref'):
            if parameter in foot_defaults:
                attributes[parameter] = foot_defaults[parameter]
        if patch.kind == 'box':
            # Bottom extends below z=0 to avoid a coplanar contact seam.
            bottom, top = -.02, patch.base_height+patch.height
            attributes.update(type='box', size=f'{hx} {hy} {(top-bottom)/2}',
                pos=f'{patch.center[0]} {patch.center[1]} {(top+bottom)/2}')
        else:
            bottom = -.02
            # Small positive low edge avoids a degenerate zero-thickness wedge.
            vertices = [[x, y, z] for x in (-hx, hx) for y in (-hy, hy)
                        for z in (bottom, patch.base_height+(patch.height if x == hx else 0.))]
            mesh = 'terrain_mesh_'+patch.name
            ET.SubElement(assets, 'mesh', name=mesh,
                          vertex=' '.join(str(v) for point in vertices for v in point))
            attributes.update(type='mesh', mesh=mesh,
                              pos=f'{patch.center[0]} {patch.center[1]} 0')
        ET.SubElement(world, 'geom', **attributes)
