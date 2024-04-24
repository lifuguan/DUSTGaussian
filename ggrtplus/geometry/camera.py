import torch
import torch.nn as nn

from ggrtplus.base.functools import lru_cache
from ggrtplus.hack_torch.custom_grid import custom_meshgrid


########################################################################################################################

@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    """
    Create meshgrid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    xs : torch.Tensor [B,1,W]
        Meshgrid in dimension x
    ys : torch.Tensor [B,H,1]
        Meshgrid in dimension y
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = custom_meshgrid([ys, xs]) # torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])

@lru_cache(maxsize=None)
def image_grid(B, H, W, dtype, device, normalized=False):
    """
    Create an image grid with a specific resolution

    Parameters
    ----------
    B : int
        Batch size
    H : int
        Height size
    W : int
        Width size
    dtype : torch.dtype
        Meshgrid type
    device : torch.device
        Meshgrid device
    normalized : bool
        True if grid is normalized between -1 and 1

    Returns
    -------
    grid : torch.Tensor [B,3,H,W]
        Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid

########################################################################################################################

def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K

class Camera(nn.Module):
    """
    Differentiable camera class implementing reconstruction and projection
    functions for a pinhole model.
    """
    def __init__(self, K, Twc=None):
        """
        Initializes the Camera class

        Parameters
        ----------
        K : torch.Tensor [3,3]
            Camera intrinsics
        Twc : Pose
            World -> Camera pose transformation
        """
        super().__init__()
        self.K = K
        self.Twc = torch.eye(4, dtype=torch.float).repeat([len(K),1,1]) if Twc is None else Twc

    def __len__(self):
        """Batch size of the camera intrinsics"""
        return len(self.K)

    def to(self, *args, **kwargs):
        """Moves object to a specific device"""
        self.K = self.K.to(*args, **kwargs)
        self.Twc = self.Twc.to(*args, **kwargs)
        return self

########################################################################################################################

    @property
    def fx(self):
        """Focal length in x"""
        return self.K[:, 0, 0]

    @property
    def fy(self):
        """Focal length in y"""
        return self.K[:, 1, 1]

    @property
    def cx(self):
        """Principal point in x"""
        return self.K[:, 0, 2]

    @property
    def cy(self):
        """Principal point in y"""
        return self.K[:, 1, 2]

    @property
    @lru_cache()
    def Tcw(self):
        """Camera -> World pose transformation (inverse of Twc)"""
        return self.Twc.inverse()

    @property
    @lru_cache()
    def Kinv(self):
        """Inverse intrinsics (for lifting)"""
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv

########################################################################################################################

    def scaled(self, x_scale, y_scale=None):
        """
        Returns a scaled version of the camera (changing intrinsics)

        Parameters
        ----------
        x_scale : float
            Resize scale in x
        y_scale : float
            Resize scale in y. If None, use the same as x_scale

        Returns
        -------
        camera : Camera
            Scaled version of the current cmaera
        """
        # If single value is provided, use for both dimensions
        if y_scale is None:
            y_scale = x_scale
        # If no scaling is necessary, return same camera
        if x_scale == 1. and y_scale == 1.:
            return self
        # Scale intrinsics and return new camera with same Pose
        K = scale_intrinsics(self.K.clone(), x_scale, y_scale)
        return Camera(K, Twc=self.Twc)

    def reconstruct(self, depth, frame='w'):
        """
        Reconstructs pixel-wise 3D points from a depth map.

        Parameters
        ----------
        depth : torch.Tensor [B,1,H,W]
            Depth map for the camera
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.tensor [B,3,H,W]
            Pixel-wise 3D points
        """
        B, C, H, W = depth.shape
        # print(f'depth shape: {depth.shape}')
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False)  # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1)  # [B,3,HW]
        # print(f'flat_grid shape: {flat_grid.shape}')

        # Estimate the outward rays in the camera frame
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # print(f'xnorm shape: {xnorm.shape}')
        # Scale rays to metric depth
        Xc = xnorm * depth
        # print(f'Xc shape: {Xc.shape}')

        # If in camera frame of reference
        if frame == 'c':
            return Xc
        # If in world frame of reference
        elif frame == 'w':
            # print(f'Twc shape: {self.Twc.shape}')
            Rcw, tcw = self.Tcw[..., :3, :3], self.Tcw[..., :3, 3].unsqueeze(-1)
            Xc = Xc.reshape(B, 3, -1)
            # print(f'Xc shape: {Xc.shape}')
            # print(f'Rcw shape: {Rcw.shape}, tcw shape: {tcw.shape}')
            Xw = (Rcw @ Xc + tcw).reshape(B, 3, H, W)
            return Xw
        # If none of the above
        else:
            raise ValueError(f'Unknown reference frame {frame}')

    def project(self, X, frame='w', normalize=True):
        """
        Projects 3D points onto the image plane

        Parameters
        ----------
        X : torch.Tensor [B,3,H,W]
            3D points to be projected
        frame : 'w'
            Reference frame: 'c' for camera and 'w' for world

        Returns
        -------
        points : torch.Tensor [B,H,W,2]
            2D projected points that are within the image boundaries
        """
        B, C, H, W = X.shape
        assert C == 3

        # print(f'X shape: {X.shape}')

        # Project 3D points onto the camera image plane
        if frame == 'c':
            Xc = self.K.bmm(X.view(B, 3, -1))
        elif frame == 'w':
            # print(f'self.K shape: {self.K.shape}')
            # print(f'self.Twc shape: {self.Twc.shape}')
            Rwc, twc = self.Twc[..., :3, :3], self.Twc[..., :3, 3].unsqueeze(-1)
            Xw = X.reshape(B, C, -1)
            Xc = Rwc @ Xw + twc
            K = self.K if len(self.K.shape) == 3 else self.K.unsqueeze(0)
            Xc = K @ Xc
        else:
            raise ValueError('Unknown reference frame {}'.format(frame))

        # Normalize points
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        if normalize:
            Xnorm = 2 * (X / Z) / (W - 1) - 1.  #(-1, 1)
            Ynorm = 2 * (Y / Z) / (H - 1) - 1.
        else:
            Xnorm = X / Z
            Ynorm = Y / Z

        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
    # def d_2_depth(self,ref_coords,pose,world_points):

