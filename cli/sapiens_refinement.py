"""
Keypoint refinement functions from Sapiens
Includes Dark Pose and UDP refinement for sub-pixel accuracy
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy.ndimage import gaussian_filter


def gaussian_blur(heatmaps: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply Gaussian blur to heatmaps.
    
    Args:
        heatmaps: Heatmaps in shape (K, H, W)
        kernel_size: Gaussian kernel size (must be odd)
    
    Returns:
        Blurred heatmaps
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    sigma = (kernel_size - 1) / 4.0
    blurred = np.zeros_like(heatmaps)
    
    for k in range(heatmaps.shape[0]):
        blurred[k] = cv2.GaussianBlur(
            heatmaps[k], (kernel_size, kernel_size), sigma
        )
    
    return blurred


def refine_keypoints_dark(
    keypoints: np.ndarray, 
    heatmaps: np.ndarray,
    blur_kernel_size: int = 11
) -> np.ndarray:
    """
    Refine keypoint predictions using Dark Pose method.
    Distribution-aware coordinate decoding for sub-pixel accuracy.
    
    Based on: https://arxiv.org/abs/1910.06278
    
    Args:
        keypoints: Keypoint coordinates in shape (N, K, D) where D=2 or 3
        heatmaps: Heatmaps in shape (K, H, W)
        blur_kernel_size: Gaussian blur kernel size
    
    Returns:
        Refined keypoint coordinates
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    
    # Modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.maximum(heatmaps, 1e-10, heatmaps)
    np.log(heatmaps, heatmaps)
    
    # Refine each keypoint
    for n in range(N):
        for k in range(K):
            x, y = keypoints[n, k, :2].astype(int)
            
            # Check bounds
            if 1 < x < W - 2 and 1 < y < H - 2:
                # First derivatives
                dx = 0.5 * (heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1])
                dy = 0.5 * (heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x])
                
                # Second derivatives
                dxx = 0.25 * (
                    heatmaps[k, y, x + 2] - 2 * heatmaps[k, y, x] +
                    heatmaps[k, y, x - 2]
                )
                dxy = 0.25 * (
                    heatmaps[k, y + 1, x + 1] - heatmaps[k, y - 1, x + 1] -
                    heatmaps[k, y + 1, x - 1] + heatmaps[k, y - 1, x - 1]
                )
                dyy = 0.25 * (
                    heatmaps[k, y + 2, x] - 2 * heatmaps[k, y, x] +
                    heatmaps[k, y - 2, x]
                )
                
                # Compute offset using Taylor expansion
                derivative = np.array([[dx], [dy]])
                hessian = np.array([[dxx, dxy], [dxy, dyy]])
                
                # Check if Hessian is invertible
                det = dxx * dyy - dxy**2
                if abs(det) > 1e-10:
                    hessian_inv = np.linalg.inv(hessian)
                    offset = -hessian_inv @ derivative
                    offset = np.squeeze(offset)
                    
                    # Apply sub-pixel offset
                    keypoints[n, k, :2] += offset
    
    return keypoints


def refine_keypoints_dark_udp(
    keypoints: np.ndarray,
    heatmaps: np.ndarray,
    blur_kernel_size: int = 11
) -> np.ndarray:
    """
    Refine keypoints using UDP (Unbiased Data Processing) method.
    More accurate for off-center keypoints.
    
    Based on: https://arxiv.org/abs/1911.07524
    
    Args:
        keypoints: Keypoint coordinates in shape (N, K, D)
        heatmaps: Heatmaps in shape (K, H, W)
        blur_kernel_size: Gaussian blur kernel size
    
    Returns:
        Refined keypoint coordinates
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    
    # Modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)
    
    # Pad heatmaps for easier indexing
    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge'
    ).flatten()
    
    for n in range(N):
        # Calculate indices for all keypoints at once
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        
        # Get neighboring values
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]
        
        # Compute derivatives
        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)
        
        # Compute Hessian
        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        
        # Invert Hessian and apply correction
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        offset = np.einsum('imn,ink->imk', hessian, derivative).squeeze()
        
        keypoints[n] -= offset
    
    return keypoints


def refine_keypoints_simple(
    keypoints: np.ndarray,
    heatmaps: np.ndarray
) -> np.ndarray:
    """
    Simple refinement by moving 0.25 pixel towards second maximum.
    Fast but less accurate than Dark/UDP methods.
    
    Args:
        keypoints: Keypoint coordinates in shape (N, K, D)
        heatmaps: Heatmaps in shape (K, H, W)
    
    Returns:
        Refined keypoint coordinates
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]
    
    for n in range(N):
        for k in range(K):
            x, y = keypoints[n, k, :2].astype(int)
            
            # Compute gradients
            dx = 0.0
            dy = 0.0
            
            if 1 < x < W - 1 and 0 < y < H:
                dx = heatmaps[k, y, x + 1] - heatmaps[k, y, x - 1]
            
            if 1 < y < H - 1 and 0 < x < W:
                dy = heatmaps[k, y + 1, x] - heatmaps[k, y - 1, x]
            
            # Move 0.25 pixel in gradient direction
            keypoints[n, k, :2] += np.sign([dx, dy]) * 0.25
    
    return keypoints


def get_heatmap_maximum(
    heatmaps: np.ndarray,
    use_softmax: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get keypoint locations from heatmaps.
    
    Args:
        heatmaps: Heatmaps in shape (K, H, W)
        use_softmax: Whether to use softmax for confidence scores
    
    Returns:
        keypoints: Keypoint coordinates in shape (K, 2)
        scores: Confidence scores in shape (K,)
    """
    K, H, W = heatmaps.shape
    
    # Reshape for argmax
    heatmaps_reshaped = heatmaps.reshape(K, -1)
    
    # Find maximum locations
    max_indices = np.argmax(heatmaps_reshaped, axis=1)
    
    # Convert to x, y coordinates
    keypoints = np.zeros((K, 2))
    keypoints[:, 0] = max_indices % W  # x
    keypoints[:, 1] = max_indices // W  # y
    
    # Get confidence scores
    if use_softmax:
        # Apply softmax for normalized scores
        exp_heatmaps = np.exp(heatmaps_reshaped - np.max(heatmaps_reshaped, axis=1, keepdims=True))
        scores = np.max(exp_heatmaps, axis=1) / np.sum(exp_heatmaps, axis=1)
    else:
        # Just use raw maximum values
        scores = np.max(heatmaps_reshaped, axis=1)
    
    return keypoints, scores


def decode_heatmap_with_refinement(
    heatmaps: np.ndarray,
    input_size: Tuple[int, int],
    heatmap_size: Optional[Tuple[int, int]] = None,
    refinement: str = 'dark_udp',
    blur_kernel_size: int = 11,
    confidence_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Complete heatmap decoding with refinement.
    
    Args:
        heatmaps: Heatmaps in shape (K, H, W) or (N, K, H, W)
        input_size: Original image size (width, height)
        heatmap_size: Heatmap size (width, height), if None use heatmap shape
        refinement: Refinement method ('dark', 'dark_udp', 'simple', 'none')
        blur_kernel_size: Gaussian blur kernel size for refinement
        confidence_threshold: Minimum confidence for valid keypoints
    
    Returns:
        keypoints: Refined keypoints in original image space (N, K, 2)
        scores: Confidence scores (N, K)
    """
    # Handle batch dimension
    if heatmaps.ndim == 3:
        heatmaps = heatmaps[np.newaxis, ...]  # Add batch dimension
    
    N, K, H, W = heatmaps.shape
    
    if heatmap_size is None:
        heatmap_size = (W, H)
    
    # Get initial keypoint locations for each batch
    all_keypoints = []
    all_scores = []
    
    for n in range(N):
        keypoints, scores = get_heatmap_maximum(heatmaps[n])
        all_keypoints.append(keypoints)
        all_scores.append(scores)
    
    keypoints = np.stack(all_keypoints)  # (N, K, 2)
    scores = np.stack(all_scores)  # (N, K)
    
    # Apply refinement
    for n in range(N):
        if refinement == 'dark':
            keypoints[n:n+1] = refine_keypoints_dark(
                keypoints[n:n+1], heatmaps[n], blur_kernel_size
            )
        elif refinement == 'dark_udp':
            keypoints[n:n+1] = refine_keypoints_dark_udp(
                keypoints[n:n+1], heatmaps[n], blur_kernel_size
            )
        elif refinement == 'simple':
            keypoints[n:n+1] = refine_keypoints_simple(
                keypoints[n:n+1], heatmaps[n]
            )
    
    # Scale keypoints to original image size
    scale_x = input_size[0] / heatmap_size[0]
    scale_y = input_size[1] / heatmap_size[1]
    keypoints[..., 0] *= scale_x
    keypoints[..., 1] *= scale_y
    
    # Filter by confidence
    valid_mask = scores > confidence_threshold
    for n in range(N):
        for k in range(K):
            if not valid_mask[n, k]:
                keypoints[n, k] = [0, 0]  # Invalid keypoint
    
    return keypoints, scores


class SapiensKeypointDecoder:
    """
    Complete keypoint decoder with Sapiens refinement methods.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        refinement: str = 'dark_udp',
        blur_kernel_size: int = 11,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            input_size: Original image size (width, height)
            heatmap_size: Heatmap output size (width, height)
            refinement: Refinement method ('dark', 'dark_udp', 'simple', 'none')
            blur_kernel_size: Gaussian blur kernel size
            confidence_threshold: Minimum confidence threshold
        """
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.refinement = refinement
        self.blur_kernel_size = blur_kernel_size
        self.confidence_threshold = confidence_threshold
        
        # Pre-compute scale factors
        self.scale_x = input_size[0] / heatmap_size[0]
        self.scale_y = input_size[1] / heatmap_size[1]
    
    def decode(
        self,
        heatmaps: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode heatmaps to keypoints with refinement.
        
        Args:
            heatmaps: Heatmaps (K, H, W) or (N, K, H, W)
            bbox: Optional bounding box (x1, y1, x2, y2) for coordinate transformation
        
        Returns:
            keypoints: Refined keypoints in image space
            scores: Confidence scores
        """
        keypoints, scores = decode_heatmap_with_refinement(
            heatmaps,
            self.input_size if bbox is None else (bbox[2] - bbox[0], bbox[3] - bbox[1]),
            self.heatmap_size,
            self.refinement,
            self.blur_kernel_size,
            self.confidence_threshold
        )
        
        # Transform to original image space if bbox provided
        if bbox is not None:
            keypoints[..., 0] += bbox[0]
            keypoints[..., 1] += bbox[1]
        
        return keypoints, scores