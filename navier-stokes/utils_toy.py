import torch
import numpy as np
import logging
import os
from PIL import Image


from utils import maybe_lag, maybe_subsample, flatten_time, get_avg_pixel_norm
from torch.utils.data import TensorDataset, DataLoader


def generate_square_dataset():
    # Configuration
    N = 200             # Number of sequences
    T = 10              # Number of frames
    square_size = 16    # Square size (H and W)
    H = 128             # Height
    W = 128             # Width
    speed = 8           # Constant horizontal speed (pixels per timestep)
    dtype = torch.float32

    # Ensure the square does not move out of bounds
    max_start_col = W - square_size - speed * (T - 1)
    max_start_row = H - square_size
    assert max_start_col >= 0 and max_start_row >= 0, "Square will exceed bounds with current parameters."

    # Initialize the dataset
    data = torch.zeros((N, T, H, W), dtype=dtype)

    # Generate N trajectories
    for n in range(N):
        # Random top-left position within allowed range
        top = np.random.randint(0, max_start_row + 1)
        bottom = top + square_size

        start_col = np.random.randint(0, max_start_col + 1)

        for t in range(T):
            left = start_col + t * speed
            right = left + square_size
            data[n, t, top:bottom, left:right] = 1.0

    # Save to disk
    torch.save(data, "square.pt")
    print(f"Saved dataset with shape {data.shape} to square.pt")


def generate_diamond_dataset():
    # Configurations
    N = 50
    T = 10
    H = 128
    W = 128
    diag = 16                      # Diagonal length of diamond
    r = diag // 2                  # Manhattan radius
    speed = 8                      # Constant speed in pixels/frame
    dtype = torch.float32

    # Constraints to ensure diamond stays fully in bounds
    max_center_row = H - 1 - r
    min_center_row = r
    max_start_col = W - 1 - r - speed * (T - 1)
    min_start_col = r

    assert max_center_row >= min_center_row
    assert max_start_col >= min_start_col

    # Create dataset
    data = torch.zeros((N, T, H, W), dtype=dtype)

    # For each trajectory
    for n in range(N):
        # Random center row and start column
        center_row = np.random.randint(min_center_row, max_center_row + 1)
        center_col_start = np.random.randint(min_start_col, max_start_col + 1)

        for t in range(T):
            center_col = center_col_start + t * speed

            for i in range(center_row - r, center_row + r + 1):
                if i < 0 or i >= H:
                    continue
                d_i = abs(i - center_row)
                j_span = r - d_i
                left = center_col - j_span
                right = center_col + j_span + 1
                if left < 0 or right > W:
                    continue
                data[n, t, i, left:right] = 1.0

    # Save the tensor
    torch.save(data, 'diamond.pt')
    print(f"Saved dataset with shape {data.shape} to diamond.pt")


def sample_diamond_dataset():
    # Parameters
    dataset_path = 'square.pt'        # Path to .pt file (must be 4D: N, T, H, W)
    output_dir = 'sampled_frames'      # Folder to save images
    num_frames = 10                    # Number of consecutive frames to save

    # Step 1: Create folder if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Load dataset
    data = torch.load(dataset_path)    # Expect shape (N, T, H, W)
    assert data.ndim == 4, "Expected shape (N, T, H, W)"
    N, T, H, W = data.shape
    assert T >= num_frames, "Not enough frames in trajectory"

    # Step 3: Randomly sample one trajectory and 10 consecutive frames
    n = np.random.randint(0, N)
    start_t = np.random.randint(0, T - num_frames + 1)

    trajectory = data[n, start_t:start_t + num_frames]  # Shape: (10, H, W)

    # Step 4: Save each frame as PNG
    for i, frame in enumerate(trajectory):
        # Convert tensor to PIL Image (scale to 0-255)
        frame_np = (frame.numpy() * 255).astype(np.uint8)
        image = Image.fromarray(frame_np)
        image.save(os.path.join(output_dir, f"frame_{i:02d}.png"))

    print(f"Saved {num_frames} frames from trajectory {n}, starting at t={start_t}, to '{output_dir}'")



def generate_square_test_dataset():
    # Configuration
    N = 25             # Number of sequences
    T = 10              # Number of frames
    square_size = 16    # Square size (H and W)
    H = 128             # Height
    W = 128             # Width
    speed = 8           # Constant horizontal speed (pixels per timestep)
    dtype = torch.float32

    # Ensure the square does not move out of bounds
    max_start_col = W - square_size - speed * (T - 1)
    max_start_row = H - square_size
    assert max_start_col >= 0 and max_start_row >= 0, "Square will exceed bounds with current parameters."

    # Initialize the dataset
    data = torch.zeros((N, T, H, W), dtype=dtype)

    # Generate N trajectories
    for n in range(N):
        # Random top-left position within allowed range
        top = np.random.randint(0, max_start_row + 1)
        bottom = top + square_size

        start_col = np.random.randint(0, max_start_col + 1)

        for t in range(T):
            left = start_col + t * speed
            right = left + square_size
            data[n, t, top:bottom, left:right] = 1.0

    # Save to disk
    torch.save(data, "square_test.pt")
    print(f"Saved dataset with shape {data.shape} to square_test.pt")


def generate_diamond_test_dataset():
    # Configurations
    N = 25
    T = 10
    H = 128
    W = 128
    diag = 16                      # Diagonal length of diamond
    r = diag // 2                  # Manhattan radius
    speed = 8                      # Constant speed in pixels/frame
    dtype = torch.float32

    # Constraints to ensure diamond stays fully in bounds
    max_center_row = H - 1 - r
    min_center_row = r
    max_start_col = W - 1 - r - speed * (T - 1)
    min_start_col = r

    assert max_center_row >= min_center_row
    assert max_start_col >= min_start_col

    # Create dataset
    data = torch.zeros((N, T, H, W), dtype=dtype)

    # For each trajectory
    for n in range(N):
        # Random center row and start column
        center_row = np.random.randint(min_center_row, max_center_row + 1)
        center_col_start = np.random.randint(min_start_col, max_start_col + 1)

        for t in range(T):
            center_col = center_col_start + t * speed

            for i in range(center_row - r, center_row + r + 1):
                if i < 0 or i >= H:
                    continue
                d_i = abs(i - center_row)
                j_span = r - d_i
                left = center_col - j_span
                right = center_col + j_span + 1
                if left < 0 or right > W:
                    continue
                data[n, t, i, left:right] = 1.0

    # Save the tensor
    torch.save(data, 'diamond_test.pt')
    print(f"Saved dataset with shape {data.shape} to diamond_test.pt")


def get_forecasting_dataloader_square(config):
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    

    avg_pixel_norm = 1.0
    data = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    logging.info("\n\n********* DATA *********\n\n")

    lo, hi = maybe_lag(data, config.time_lag)

    lo, hi = flatten_time(lo, hi, config.hi_size)

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    dataset = TensorDataset(lo, hi)

    # split into train and val according to config.val_ratio
    N = len(dataset)
    val_size = int(N * config.val_ratio)
    train_size = N - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, avg_pixel_norm, new_avg_pixel_norm


def get_forecasting_dataloader_square_sampling(config):
    data_raw = torch.load(config.data_fname)
    data_raw = data_raw.float()

    # better to subsample after flattening time dim to actually affect the num of datapoints rather than num trajectores
    #data_raw = maybe_subsample(data_raw, config.subsampling_ratio)    

    avg_pixel_norm = 1.0
    data = data_raw/avg_pixel_norm
    new_avg_pixel_norm = 1.0

    logging.info("\n\n********* DATA *********\n\n")

    lo, hi = maybe_lag(data, config.time_lag)
    config.max_T = lo.shape[1] # length of the longest trajectory

    lo, hi = flatten_time(lo, hi, config.hi_size)

    lo = maybe_subsample(lo, config.subsampling_ratio)
    hi = maybe_subsample(hi, config.subsampling_ratio)

    logging.info(f"shape of the dataset (lo): {lo.shape}")
    logging.info(f"shape of the dataset (hi): {hi.shape}")

    dataset = TensorDataset(lo, hi)

    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    
    return test_loader


import os
from typing import Tuple, Literal, List
import numpy as np
import torch


ShapeName = Literal["square", "triangle", "circle", "diamond", "plus", "ring"]


class MovingShapesDataset:
    """
    Generate a dataset of moving shapes with shape (N, T, H, W).
    - H = W = 128
    - T = 8
    - speed = 8 pixels per step to the right
    - Each sample uses one shape from: square, triangle, circle, diamond, plus
    - Each shape fits within a 16x16 bounding box (or smaller)
    """

    def __init__(
        self,
        N: int = 200,
        T: int = 8,
        H: int = 128,
        W: int = 128,
        speed: int = 8,
        max_box: int = 16,
        dtype: torch.dtype = torch.float32,
        seed: int = None,
    ):
        self.N = N
        self.T = T
        self.H = H
        self.W = W
        self.speed = speed
        self.max_box = max_box  # maximum side of the bounding box for a shape
        self.dtype = dtype

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Precompute half-extent used for placement constraints.
        # All shapes we draw will stay within a box of size <= max_box x max_box
        self.half = max_box // 2  # 8 when max_box=16

        # Check movement feasibility (entire trajectory in-bounds)
        self._check_feasibility()

        # Available shapes
        self.shapes: List[ShapeName] = ["square", "triangle", "circle", "diamond", "plus"]

    # -----------------------------
    # Public API
    # -----------------------------
    def generate(self, save_path: str = "shapes.pt") -> torch.Tensor:
        """
        Generate the dataset and save it to `save_path`.
        Returns the tensor of shape (N, T, H, W).
        """
        data = torch.zeros((self.N, self.T, self.H, self.W), dtype=self.dtype)

        for n in range(self.N):
            shape = np.random.choice(self.shapes)

            # Choose a valid vertical center row and horizontal starting center col
            cy = np.random.randint(self.min_center_row, self.max_center_row + 1)
            cx0 = np.random.randint(self.min_start_cx, self.max_start_cx + 1)

            for t in range(self.T):
                cx = cx0 + t * self.speed
                # self._draw_shape(data[n, t], shape, cy, cx)
                self._draw_ring(data[n, t], cy, cx)
                # self._draw_square(data[n, t], cy, cx)

        torch.save(data, save_path)
        print(f"Saved dataset with shape {tuple(data.shape)} to {save_path}")
        return data

    def save_gif_from_file(self, file_path: str, n: int, gif_path: str, fps: int = 2):
        """
        Load dataset from `file_path`, select trajectory `n`, and save it as a GIF to `gif_path`.
        fps controls frames per second.
        """
        # Load dataset
        data = torch.load(file_path)
        assert data.ndim == 4, f"Expected dataset shape (N, T, H, W), got {data.shape}"
        assert 0 <= n < data.shape[0], f"n must be between 0 and {data.shape[0]-1}"

        frames = []
        for t in range(data.shape[1]):
            frame_np = (data[n, t].numpy() * 255).astype(np.uint8)
            img = Image.fromarray(frame_np, mode="L")  # grayscale
            frames.append(img)

        duration = int(1000 / fps)  # ms per frame
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved trajectory {n} from '{file_path}' as GIF to '{gif_path}'")


    def save_multi_gif_from_file(self, file_path: str, n: int, gif_path: str, length: int, fps: int = 2, boundary: int = 2):
        """
        Load dataset from `file_path` and visualize length**2 trajectories starting at index n
        arranged in a length x length grid, with clear boundaries between them.
        fps controls frames per second.
        boundary sets the number of pixels for the boundary lines.
        """
        # Load dataset
        data = torch.load(file_path)
        assert data.ndim == 4, f"Expected dataset shape (N, T, H, W), got {data.shape}"
        N, T, H, W = data.shape
        total_needed = length ** 2
        assert 0 <= n < N, f"n must be between 0 and {N-1}"
        assert n + total_needed <= N, f"Requested range [{n}, {n+total_needed-1}] exceeds dataset size {N}"

        frames = []
        for t in range(T):
            # Create a blank canvas for the grid
            big_H = length * H + (length + 1) * boundary
            big_W = length * W + (length + 1) * boundary
            combined = Image.new("L", (big_W, big_H), color=255)  # white background

            # Place each trajectory frame in the grid
            for grid_row in range(length):
                for grid_col in range(length):
                    idx = n + grid_row * length + grid_col
                    frame_np = (data[idx, t].numpy() * 255).astype(np.uint8)
                    img = Image.fromarray(frame_np, mode="L")

                    # Top-left corner of this cell
                    y0 = boundary + grid_row * (H + boundary)
                    x0 = boundary + grid_col * (W + boundary)
                    combined.paste(img, (x0, y0))

            frames.append(combined)

        duration = int(1000 / fps)  # ms per frame
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved {total_needed} trajectories [{n} to {n+total_needed-1}] as {length}x{length} grid GIF to '{gif_path}'")


    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _check_feasibility(self):
        """
        Compute placement bounds so that a shape of max_box fits in-bounds
        for all frames along the rightward motion.
        """
        # Vertical: the center must be at least `half` away from top and bottom.
        self.min_center_row = self.half
        self.max_center_row = (self.H - 1) - self.half
        assert self.max_center_row >= self.min_center_row, "Height too small for shape placement."

        # Horizontal: ensure the center + half never exceeds W-1 over the whole trajectory.
        # cx(t) = cx0 + t*speed. For all t, cx(t) + half <= W-1  and  cx(t) - half >= 0
        # The tightest constraint is at the last timestep t = T-1 for the right edge.
        self.min_start_cx = self.half
        self.max_start_cx = (self.W - 1) - self.half - self.speed * (self.T - 1)
        assert self.max_start_cx >= self.min_start_cx, "Width, speed, or T too large for in-bounds motion."

    def _draw_shape(self, canvas: torch.Tensor, shape: ShapeName, cy: int, cx: int):
        """
        Draw a single shape on `canvas` (H x W) centered at (cy, cx).
        Every shape is constrained to be within a 16x16 (or smaller) box.
        """
        if shape == "square":
            self._draw_square(canvas, cy, cx)
        elif shape == "triangle":
            self._draw_triangle_upright(canvas, cy, cx)
        elif shape == "circle":
            self._draw_circle(canvas, cy, cx)
        elif shape == "diamond":
            self._draw_diamond(canvas, cy, cx)
        elif shape == "plus":
            self._draw_plus(canvas, cy, cx)
        elif shape == "ring":
            self._draw_ring(canvas, cy, cx)
        else:
            raise ValueError(f"Unknown shape: {shape}")

    # -----------------------------
    # Shape rasterizers (binary fill)
    # -----------------------------
    def _draw_ring(self, canvas: torch.Tensor, cy: int, cx: int, outer: int = 7, thickness: int = 3):
        """
        Draw a hollow circle (ring) centered at (cy, cx).
        - outer: outer radius in pixels (<= 7 to fit in 16x16 box)
        - thickness: ring thickness in pixels (>= 1 and <= outer)
        """
        assert 1 <= thickness <= outer <= (self.max_box // 2 - 1), \
            f"Choose outer <= {self.max_box // 2 - 1} and 1 <= thickness <= outer"

        inner = max(0, outer - thickness)

        # Bounding box for the 16x16 region
        top, bottom, left, right = self._box_bounds(cy, cx, self.max_box)

        # Make masks in that box
        yy, xx = np.ogrid[top:bottom, left:right]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        mask_outer = dist2 <= outer ** 2
        mask_inner = dist2 < inner ** 2
        ring_mask = np.logical_and(mask_outer, np.logical_not(mask_inner))

        canvas[top:bottom, left:right][torch.from_numpy(ring_mask)] = 1.0

    def _box_bounds(self, cy: int, cx: int, box: int = None) -> Tuple[int, int, int, int]:
        """
        Return the bounding rows and cols for a centered box.
        The box size defaults to self.max_box.
        """
        if box is None:
            box = self.max_box
        half = box // 2
        top = cy - half
        left = cx - half
        bottom = top + box
        right = left + box
        return top, bottom, left, right

    def _draw_square(self, canvas: torch.Tensor, cy: int, cx: int):
        # Exact 16x16 square
        top, bottom, left, right = self._box_bounds(cy, cx, self.max_box)
        canvas[top:bottom, left:right] = 1.0

    def _draw_triangle_upright(self, canvas: torch.Tensor, cy: int, cx: int):
        # Upright isosceles triangle, height <= 16, base <= 16
        height = self.max_box
        base = self.max_box
        top = cy - (height // 2)
        for r in range(height):
            row = top + r
            # Linearly increasing half-width from 0 up to base//2
            half_w = (r * (base // 2)) // (height - 1)  # integer half-width
            left = cx - half_w
            right = cx + half_w
            canvas[row, left:right + 1] = 1.0

    def _draw_circle(self, canvas: torch.Tensor, cy: int, cx: int):
        # Circle with radius <= 8 that fits in 16x16. Use radius=7 for a safe tight fit in discrete grid.
        R = self.max_box // 2 - 1  # 7 when max_box=16
        top, bottom, left, right = self._box_bounds(cy, cx, self.max_box)
        yy, xx = np.ogrid[top:bottom, left:right]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= R ** 2
        canvas[top:bottom, left:right][torch.from_numpy(mask)] = 1.0

    def _draw_diamond(self, canvas: torch.Tensor, cy: int, cx: int):
        # Diamond (rotated square) using L1 distance.
        # Use Manhattan radius r=7 so the footprint fits within 16x16 on a discrete grid.
        r = self.max_box // 2 - 1  # 7
        top, bottom, left, right = self._box_bounds(cy, cx, self.max_box)
        yy, xx = np.ogrid[top:bottom, left:right]
        mask = (np.abs(yy - cy) + np.abs(xx - cx)) <= r
        canvas[top:bottom, left:right][torch.from_numpy(mask)] = 1.0

    def _draw_plus(self, canvas: torch.Tensor, cy: int, cx: int):
        # Plus sign inside 16x16: arm length 8, bar thickness 5 (odd thickness for symmetry)
        arm = self.max_box // 2  # 8
        thickness = 5
        half_th = thickness // 2
        # Horizontal bar
        canvas[cy - half_th: cy + half_th + 1, cx - arm: cx + arm] = 1.0
        # Vertical bar
        canvas[cy - arm: cy + arm, cx - half_th: cx + half_th + 1] = 1.0


def MovingShapesDataset_main():
    # Example: generate and save to 'multi_shapes.pt'
    gen = MovingShapesDataset(N=50, T=8, H=128, W=128, speed=8, max_box=16, seed=42)
    _ = gen.generate(save_path="rings.pt")

def MovingShapesDataset_save_gif_main():
    gen = MovingShapesDataset()  # parameters here do not affect GIF saving
    gen.save_multi_gif_from_file("./rings.pt", n=8, gif_path="trajectorys_rings.gif", length=6, fps=4, boundary=2)

if __name__ == "__main__":
    # generate_square_dataset()
    # generate_diamond_dataset()
    # sample_diamond_dataset()
    # generate_square_test_dataset()
    # MovingShapesDataset_main()
    # MovingShapesDataset_main()
    # MovingShapesDataset_save_gif_main()
    # MovingShapesDataset_main()
    MovingShapesDataset_save_gif_main()