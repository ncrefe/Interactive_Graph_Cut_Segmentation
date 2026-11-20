import maxflow
from sklearn.mixture import GaussianMixture
import cv2
import numpy as np
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import Toplevel, filedialog, messagebox


# you can use GUI-related libraries if needed


class GraphCutUnary:
    def __init__(self, img, scrib):
        # Load images
        self.img = img

        # Convert BGR to RGB for visualization and modeling
        self.img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Load scribble annotation image and convert to RGB
        self.scrib = scrib
        self.scrib_rgb = cv2.cvtColor(self.scrib, cv2.COLOR_BGR2RGB)

        # Create boolean masks from scribble colors:
        # white-ish scribble [255,255,207] denotes foreground scribble
        # red scribble [219,0,0] denotes background scribble
        self.foreground_mask = np.all(self.scrib_rgb == [255, 255, 207], axis=2)
        self.background_mask = np.all(self.scrib_rgb == [219, 0, 0], axis=2)

        self.h, self.w, _ = self.img.shape

    def normalize_unary(self, unary):
        mn = unary.min()
        mx = unary.max()
        return (unary - mn) / (mx - mn + 1e-6) * 50.0

    def compute_unary_gmm(self, n_components=5):
        """
        Compute unary costs using Gaussian Mixture Models (GMM).
        For each class (foreground/background) we fit a GMM to scribble pixels
        and compute the negative log-likelihood for every image pixel.
        Unary definition (per pixel p):
            unary_bg(p) = -log P_bg(color_p)
            unary_fg(p) = -log P_fg(color_p)
        The returned unary array has shape (N_pixels, 2) with columns [bg_cost, fg_cost].
        """
        # Flatten image pixels (RGB)
        img_flat = self.img_rgb.reshape(-1, 3)

        # Foreground GMM training from scribbles
        fg_pixels = self.img_rgb[self.foreground_mask]

        # adapt number of components to scribble size (avoid overfitting)
        fg_n_comp = min(n_components, max(1, len(fg_pixels) // 5))

        fg_gmm = GaussianMixture(
            n_components=fg_n_comp,
            covariance_type='full',
            reg_covar=1e-3,  # 5
            max_iter=1000
        )
        fg_gmm.fit(fg_pixels)

        # Background GMM training from scribbles
        bg_pixels = self.img_rgb[self.background_mask]

        bg_n_comp = min(n_components, max(1, len(bg_pixels) // 10))

        bg_gmm = GaussianMixture(
            n_components=bg_n_comp,
            covariance_type='full',
            reg_covar=1e-3,  # 4
            max_iter=1000
        )
        bg_gmm.fit(bg_pixels)

        # Compute negative log-likelihood for each pixel under both GMMs

        fg_ll = -fg_gmm.score_samples(img_flat)
        bg_ll = -bg_gmm.score_samples(img_flat)

        # Sanity: replace NaN/Inf and clip extreme values for stability
        fg_ll = np.nan_to_num(fg_ll, nan=100.0, posinf=100.0, neginf=0.0)
        bg_ll = np.nan_to_num(bg_ll, nan=100.0, posinf=100.0, neginf=0.0)

        # IT LIMITS THE MAXIMUM ERROR ITS OPTIONAL
        fg_ll = np.clip(fg_ll, 0, 200)
        bg_ll = np.clip(bg_ll, 0, 100)

        unary = np.stack([bg_ll, fg_ll], axis=1)
        return self.normalize_unary(unary)

    def compute_unary_histogram(self, bins=32):
        """
        Compute unary costs using color histograms in HSV space.
        We compute 3D histograms for foreground and background scribbles:
            P_fg(h,s,v) and P_bg(h,s,v)
        Then unary costs are negative log-probabilities:
            unary_bg(p) = -log P_bg(h_p,s_p,v_p)
            unary_fg(p) = -log P_fg(h_p,s_p,v_p)
        Returns: unary array shape (N_pixels, 2) with [bg_cost, fg_cost].
        """
        # Convert BGR->HSV and index the scribble pixels
        img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        fg_hsv = img_hsv[self.foreground_mask]
        bg_hsv = img_hsv[self.background_mask]

        # 3D histograms (H,S,V) over chosen ranges
        fg_hist, _ = np.histogramdd(fg_hsv, bins=bins, range=[(0, 180), (0, 256), (0, 256)])
        bg_hist, _ = np.histogramdd(bg_hsv, bins=bins, range=[(0, 180), (0, 256), (0, 256)])

        # Add small constant and normalize to form probabilities
        fg_hist = fg_hist + 1e-8
        bg_hist = bg_hist + 1e-8
        fg_hist /= fg_hist.sum()
        bg_hist /= bg_hist.sum()

        # Map every pixel to its HSV bin
        pixels = img_hsv.reshape(-1, 3)
        h_bins = np.linspace(0, 180, bins + 1)
        s_bins = np.linspace(0, 256, bins + 1)
        v_bins = np.linspace(0, 256, bins + 1)

        hi = np.digitize(pixels[:, 0], h_bins) - 1
        si = np.digitize(pixels[:, 1], s_bins) - 1
        vi = np.digitize(pixels[:, 2], v_bins) - 1

        hi = np.clip(hi, 0, bins - 1)
        si = np.clip(si, 0, bins - 1)
        vi = np.clip(vi, 0, bins - 1)

        fg_prob = fg_hist[hi, si, vi]
        bg_prob = bg_hist[hi, si, vi]

        # Negative log-probabilities form unary costs
        unary = np.stack([-np.log(bg_prob), -np.log(fg_prob)], axis=1)
        return self.normalize_unary(unary)

    def compute_pairwise(self, gamma=50.0, beta=None):
        """
        Compute pairwise smoothness weights for a 4-neighborhood grid.
        The standard form used here is:
            w(p,q) = gamma * exp(-beta * ||I_p - I_q||^2)
        where ||I_p - I_q||^2 is the squared color difference between neighboring pixels.
        Beta is automatically estimated as:
            beta = 1 / (2 * mean( ||I_p - I_q||^2 ) )
        if not provided.
        The function returns an array of shape (E, 3) with [p_idx, q_idx, weight].
        """
        h, w = self.h, self.w
        img = self.img_rgb.astype(np.float64)

        idxs_p = []
        idxs_q = []
        weights = []

        def idx(y, x):
            return y * w + x

        diffs = []

        # Right neighbors diffs
        for y in range(h):
            for x in range(w - 1):
                p = img[y, x]
                q = img[y, x + 1]
                diff2 = np.sum((p - q) ** 2)
                diffs.append(diff2)

        # Down neighbors diffs
        for y in range(h - 1):
            for x in range(w):
                p = img[y, x]
                q = img[y + 1, x]
                diff2 = np.sum((p - q) ** 2)
                diffs.append(diff2)

        diffs = np.array(diffs, dtype=np.float64)
        mean_diff2 = diffs.mean() if diffs.size > 0 else 0.0

        # Automatic beta estimation (see GrabCut literature)
        if beta is None:
            if mean_diff2 < 1e-10:
                beta = 0.0
            else:
                beta = 1.0 / (2.0 * mean_diff2)

        # Build right-edge weights
        for y in range(h):
            for x in range(w - 1):
                p_idx = idx(y, x)
                q_idx = idx(y, x + 1)
                diff2 = np.sum((img[y, x] - img[y, x + 1]) ** 2)
                w_ = gamma * np.exp(-beta * diff2)
                idxs_p.append(p_idx)
                idxs_q.append(q_idx)
                weights.append(w_)

        # Build down-edge weights
        for y in range(h - 1):
            for x in range(w):
                p_idx = idx(y, x)
                q_idx = idx(y + 1, x)
                diff2 = np.sum((img[y, x] - img[y + 1, x]) ** 2)
                w_ = gamma * np.exp(-beta * diff2)
                idxs_p.append(p_idx)
                idxs_q.append(q_idx)
                weights.append(w_)

        edges = np.stack(
            [np.array(idxs_p, dtype=np.int32),
             np.array(idxs_q, dtype=np.int32),
             np.array(weights, dtype=np.float64)],
            axis=1
        )
        return edges

    def segment(self, unary_cost, edges, use_scribbles=True):
        """
        Perform GraphCut segmentation using PyMaxflow.

        Arguments:
            unary_cost: (N,2) array where columns are [bg_cost, fg_cost]
            edges: (E,3) array listing [p_idx, q_idx, weight]
            use_scribbles: if True, enforce scribbled pixels via very large terminal capacities

        The graph construction:
            - node i has terminal link capacities (source_cost = unary[i,1], sink_cost = unary[i,0])
            - p-q edges have symmetric weight w
            - scribbled foreground pixels get strong capacity to source (or sink depending on convention)
        Returns:
            segmentation_mask (h,w) with values 0 or 1 (1 indicates foreground side of cut)
        """
        n_pixels = self.h * self.w
        INF = 1e10

        # Create graph with estimated sizes (PyMaxflow API)
        g = maxflow.Graph[float](n_pixels, edges.shape[0])
        nodes = g.add_nodes(n_pixels)

        # Flatten scribble masks for quick indexing
        fg_mask_flat = self.foreground_mask.flatten()
        bg_mask_flat = self.background_mask.flatten()

        # Add terminal edges (unary)
        for i in range(n_pixels):
            if use_scribbles:
                if fg_mask_flat[i]:
                    # Force this pixel to foreground
                    g.add_tedge(nodes[i], INF, 0)  # source=INF, sink=0
                elif bg_mask_flat[i]:
                    # Force this pixel to background
                    g.add_tedge(nodes[i], 0, INF)  # source=0, sink=INF
                else:
                    bg_cost = unary_cost[i, 0]
                    fg_cost = unary_cost[i, 1]
                    g.add_tedge(nodes[i], bg_cost, fg_cost)
            else:
                bg_cost = unary_cost[i, 0]
                fg_cost = unary_cost[i, 1]
                g.add_tedge(nodes[i], bg_cost, fg_cost)

        # Add pairwise edges
        for edge in edges:
            p_idx = int(edge[0])
            q_idx = int(edge[1])
            weight = float(edge[2])
            g.add_edge(nodes[p_idx], nodes[q_idx], weight, weight)

        # Compute maxflow / mincut
        flow = g.maxflow()
        # print(f"Computed maxflow value: {flow}")

        # Extract segmentation (0/1 per node)
        segmentation = np.zeros(n_pixels, dtype=np.uint8)
        for i in range(n_pixels):
            segmentation[i] = g.get_segment(nodes[i])

        segmentation_mask = segmentation.reshape(self.h, self.w)
        return segmentation_mask

    def compute_histogram_based_binary_mask(self, bins=32, gamma=200, use_scribbles=True):
        # Compute histogram-based unary (HSV histogram)
        unary_hist = self.compute_unary_histogram(bins)

        # Compute pairwise edges with chosen gamma (keep default behavior)
        edges = self.compute_pairwise(gamma)

        # Perform GraphCut segmentation enforcing scribbles
        seg = self.segment(unary_hist, edges, use_scribbles)

        # Clean small noise using morphological opening
        seg_uint8 = seg.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seg_clean = cv2.morphologyEx(seg_uint8, cv2.MORPH_OPEN, kernel)

        # Ensure segmentation polarity matches GT:
        seg_inv = 1 - seg_clean

        return seg_inv

    def compute_gmm_based_binary_mask(self, n_components=5, gamma=200, use_scribbles=True):
        """
        Compute binary mask using GMM-based unary costs.
        Steps:
            1) Compute unary using GMMs fitted to scribbles
            2) Compute pairwise edges
            3) Run GraphCut with scribbles enforced
            4) Clean small noise using morphological opening
            5) Return cleaned mask
        """
        # 1) Compute unary costs with GMM
        unary_gmm = self.compute_unary_gmm(n_components=n_components)

        # 2) Compute pairwise edges
        edges = self.compute_pairwise(gamma)

        # 3) GraphCut segmentation enforcing scribbles
        seg = self.segment(unary_gmm, edges, use_scribbles)

        # 4) Morphological opening to remove small noise
        seg_uint8 = seg.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        seg_clean = cv2.morphologyEx(seg_uint8, cv2.MORPH_OPEN, kernel)

        # 5) Optionally invert mask if needed (ensure foreground consistent)
        seg_inv = 1 - seg_clean

        return seg_inv  # return both versions; user can pick later


class InteractiveSegmentationGUI:
    def __init__(self, root, graph_cut_class):
        self.root = root
        self.root.title("Interactive Image Segmentation Tool")
        self.GraphCutUnary = graph_cut_class

        # State variables
        self.original_image = None
        self.display_image = None
        self.scribble_image = None
        self.segmentation_mask = None
        self.graph_cut_obj = None

        # Drawing state
        self.drawing = False
        self.current_color = [207, 255, 255]  # Foreground (white-ish) in BGR
        self.brush_size = 5
        self.last_x = None
        self.last_y = None

        # Undo history
        self.scribble_history = deque(maxlen=20)

        # Display scale
        self.display_scale = 1.0
        self.canvas_width = 800
        self.canvas_height = 600

        # Setup GUI
        self.setup_gui()

    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        # File operations
        ttk.Button(control_frame, text="Load Image",
                   command=self.load_image).grid(row=0, column=0, pady=5, sticky=tk.W + tk.E)
        ttk.Button(control_frame, text="Save Mask",
                   command=self.save_mask).grid(row=1, column=0, pady=5, sticky=tk.W + tk.E)
        ttk.Button(control_frame, text="Reset Scribbles",
                   command=self.reset_scribbles).grid(row=2, column=0, pady=5, sticky=tk.W + tk.E)

        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, sticky=tk.W + tk.E, pady=10)

        # Brush controls
        ttk.Label(control_frame, text="Brush Controls",
                  font=('TkDefaultFont', 9, 'bold')).grid(row=4, column=0, pady=5)

        # Foreground/Background selection
        self.brush_mode = tk.StringVar(value="foreground")
        ttk.Radiobutton(control_frame, text="Foreground (Yellow-ish)",
                        variable=self.brush_mode, value="foreground",
                        command=self.change_brush_mode).grid(row=5, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Background (Red)",
                        variable=self.brush_mode, value="background",
                        command=self.change_brush_mode).grid(row=6, column=0, sticky=tk.W)

        # Brush size
        ttk.Label(control_frame, text="Brush Size:").grid(row=7, column=0, pady=(10, 0), sticky=tk.W)
        self.brush_size_var = tk.IntVar(value=5)
        brush_slider = ttk.Scale(control_frame, from_=1, to=20,
                                 variable=self.brush_size_var,
                                 orient=tk.HORIZONTAL,
                                 command=self.update_brush_size)
        brush_slider.grid(row=8, column=0, sticky=tk.W + tk.E)
        self.brush_size_label = ttk.Label(control_frame, text="5 px")
        self.brush_size_label.grid(row=9, column=0, sticky=tk.W)

        ttk.Separator(control_frame, orient='horizontal').grid(row=10, column=0, sticky=tk.W + tk.E, pady=10)

        # Segmentation parameters
        ttk.Label(control_frame, text="Segmentation Method",
                  font=('TkDefaultFont', 9, 'bold')).grid(row=11, column=0, pady=5)

        self.method = tk.StringVar(value="gmm")
        ttk.Radiobutton(control_frame, text="GMM-based",
                        variable=self.method, value="gmm").grid(row=12, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Histogram-based",
                        variable=self.method, value="histogram").grid(row=13, column=0, sticky=tk.W)

        # GMM parameters
        ttk.Label(control_frame, text="GMM Components:").grid(row=14, column=0, pady=(10, 0), sticky=tk.W)
        self.gmm_components = tk.IntVar(value=5)
        ttk.Spinbox(control_frame, from_=1, to=10,
                    textvariable=self.gmm_components, width=10).grid(row=15, column=0, sticky=tk.W)

        # Histogram parameters
        ttk.Label(control_frame, text="Histogram Bins:").grid(row=16, column=0, pady=(10, 0), sticky=tk.W)
        self.hist_bins = tk.IntVar(value=32)
        ttk.Spinbox(control_frame, from_=8, to=64, increment=8,
                    textvariable=self.hist_bins, width=10).grid(row=17, column=0, sticky=tk.W)

        # Gamma parameter (common)
        ttk.Label(control_frame, text="Gamma (smoothness):").grid(row=18, column=0, pady=(10, 0), sticky=tk.W)
        self.gamma = tk.DoubleVar(value=200.0)
        ttk.Spinbox(control_frame, from_=10, to=500, increment=10,
                    textvariable=self.gamma, width=10).grid(row=19, column=0, sticky=tk.W)

        # Use scribbles checkbox
        self.use_scribbles = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Enforce Scribbles",
                        variable=self.use_scribbles).grid(row=20, column=0, pady=10, sticky=tk.W)

        # Segmentation button
        ttk.Button(control_frame, text="Run Segmentation (Space)",
                   command=self.run_segmentation,
                   style='Accent.TButton').grid(row=21, column=0, pady=10, sticky=tk.W + tk.E)

        # Undo button
        ttk.Button(control_frame, text="Undo (Ctrl+Z)",
                   command=self.undo_scribble).grid(row=22, column=0, pady=5, sticky=tk.W + tk.E)

        # Visualization options
        ttk.Separator(control_frame, orient='horizontal').grid(row=23, column=0, sticky=tk.W + tk.E, pady=10)
        ttk.Label(control_frame, text="Visualization",
                  font=('TkDefaultFont', 9, 'bold')).grid(row=24, column=0, pady=5)

        self.show_mode = tk.StringVar(value="overlay")
        ttk.Radiobutton(control_frame, text="Overlay Mask",
                        variable=self.show_mode, value="overlay",
                        command=self.update_display).grid(row=25, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Binary Mask",
                        variable=self.show_mode, value="mask",
                        command=self.update_display).grid(row=26, column=0, sticky=tk.W)
        ttk.Radiobutton(control_frame, text="Original + Scribbles",
                        variable=self.show_mode, value="scribbles",
                        command=self.update_display).grid(row=27, column=0, sticky=tk.W)

        # IoU display
        ttk.Separator(control_frame, orient='horizontal').grid(row=28, column=0, sticky=tk.W + tk.E, pady=10)
        ttk.Button(control_frame, text="Calculate IoU",
                   command=self.calculate_iou).grid(row=29, column=0, pady=5, sticky=tk.W + tk.E)
        self.iou_label = ttk.Label(control_frame, text="IoU: N/A",
                                   font=('TkDefaultFont', 10, 'bold'))
        self.iou_label.grid(row=30, column=0, pady=5)

        # Right panel - Canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="Image", padding="10")
        canvas_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width,
                                height=self.canvas_height, bg='gray')
        self.canvas.pack()

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        # Bind keyboard shortcuts
        self.root.bind("<space>", lambda e: self.run_segmentation())
        self.root.bind("<Control-z>", lambda e: self.undo_scribble())
        self.root.bind("<Control-s>", lambda e: self.save_mask())

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Load an image to begin",
                                      relief=tk.SUNKEN)
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=tk.W + tk.E, pady=5)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        # Load image
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        # Initialize scribble image (black background)
        h, w = self.original_image.shape[:2]
        self.scribble_image = np.zeros((h, w, 3), dtype=np.uint8)
        self.segmentation_mask = None
        self.scribble_history.clear()

        # Calculate display scale
        self.calculate_display_scale()

        # Update displayG
        self.update_display()
        self.status_label.config(text=f"Loaded: {file_path} | Size: {w}x{h}")

    def calculate_display_scale(self):
        if self.original_image is None:
            return

        h, w = self.original_image.shape[:2]
        scale_w = self.canvas_width / w
        scale_h = self.canvas_height / h
        self.display_scale = min(scale_w, scale_h, 1.0)  # Don't upscale

    def update_display(self):
        if self.original_image is None:
            return

        mode = self.show_mode.get()

        if mode == "mask" and self.segmentation_mask is not None:
            # Show binary mask
            display = (self.segmentation_mask * 255).astype(np.uint8)
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        elif mode == "overlay" and self.segmentation_mask is not None:
            # Overlay mask on original image
            display = self.original_image.copy()
            mask_colored = np.zeros_like(display)
            mask_colored[self.segmentation_mask == 1] = [0, 255, 0]  # Green for foreground
            display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

            # Add scribbles
            scribble_mask = np.any(self.scribble_image != 0, axis=2)
            display[scribble_mask] = self.scribble_image[scribble_mask]
        else:
            # Show original with scribbles
            display = self.original_image.copy()
            scribble_mask = np.any(self.scribble_image != 0, axis=2)
            display[scribble_mask] = self.scribble_image[scribble_mask]

        # Resize for display
        h, w = display.shape[:2]
        new_w = int(w * self.display_scale)
        new_h = int(h * self.display_scale)
        display_resized = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Convert to PIL and display
        display_rgb = cv2.cvtColor(display_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_rgb)
        self.display_image = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image)

    def change_brush_mode(self):
        if self.brush_mode.get() == "foreground":
            self.current_color = [207, 255, 255]  # Yellow-ish for foreground in BGR
        else:
            self.current_color = [0, 0, 219]  # Red for background in BGR

    def update_brush_size(self, value):
        self.brush_size = int(float(value))
        self.brush_size_label.config(text=f"{self.brush_size} px")

    def start_draw(self, event):
        if self.original_image is None:
            return

        # Save current state for undo
        self.scribble_history.append(self.scribble_image.copy())

        self.drawing = True
        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)
        self.last_x = x
        self.last_y = y
        self.draw_at_position(x, y)

    def draw(self, event):
        if not self.drawing or self.original_image is None:
            return

        x = int(event.x / self.display_scale)
        y = int(event.y / self.display_scale)

        # Draw line from last position to current
        if self.last_x is not None and self.last_y is not None:
            cv2.line(self.scribble_image, (self.last_x, self.last_y),
                     (x, y), self.current_color, self.brush_size)

        self.last_x = x
        self.last_y = y
        self.update_display()

    def stop_draw(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def draw_at_position(self, x, y):
        cv2.circle(self.scribble_image, (x, y), self.brush_size // 2,
                   self.current_color, -1)
        self.update_display()

    def undo_scribble(self):
        if len(self.scribble_history) > 0:
            self.scribble_image = self.scribble_history.pop()
            self.update_display()
            self.status_label.config(text="Undo applied")
        else:
            self.status_label.config(text="Nothing to undo")

    def reset_scribbles(self):
        if self.original_image is None:
            return

        if messagebox.askyesno("Confirm Reset", "Clear all scribbles?"):
            h, w = self.original_image.shape[:2]
            self.scribble_image = np.zeros((h, w, 3), dtype=np.uint8)
            self.scribble_history.clear()
            self.segmentation_mask = None
            self.update_display()
            self.status_label.config(text="Scribbles cleared")

    def run_segmentation(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return

        # Check if any scribbles exist
        if not np.any(self.scribble_image != 0):
            messagebox.showwarning("Warning", "Please draw some scribbles first")
            return

        self.status_label.config(text="Running segmentation...")
        self.root.update()

        try:
            # Create GraphCutUnary object
            self.graph_cut_obj = self.GraphCutUnary(self.original_image, self.scribble_image)

            # Get parameters
            method = self.method.get()
            gamma = self.gamma.get()
            use_scribbles = self.use_scribbles.get()

            # Run segmentation based on method
            if method == "gmm":
                n_components = self.gmm_components.get()
                self.segmentation_mask = self.graph_cut_obj.compute_gmm_based_binary_mask(
                    n_components=n_components,
                    gamma=gamma,
                    use_scribbles=use_scribbles
                )
            else:  # histogram
                bins = self.hist_bins.get()
                self.segmentation_mask = self.graph_cut_obj.compute_histogram_based_binary_mask(
                    bins=bins,
                    gamma=gamma,
                    use_scribbles=use_scribbles
                )

            self.show_mode.set("overlay")
            self.update_display()
            self.status_label.config(text=f"Segmentation complete ({method.upper()})")

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed: {str(e)}")
            self.status_label.config(text="Segmentation failed")

    def save_mask(self):
        if self.segmentation_mask is None:
            messagebox.showwarning("Warning", "No segmentation mask to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Mask",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if file_path:
            mask_to_save = (self.segmentation_mask * 255).astype(np.uint8)
            cv2.imwrite(file_path, mask_to_save)
            self.status_label.config(text=f"Mask saved: {file_path}")
            messagebox.showinfo("Success", "Mask saved successfully")

    def calculate_iou(self):
        if self.segmentation_mask is None:
            messagebox.showerror("Error", "Run segmentation first.")
            return

        # Get the ground truth (GT) file path from the user
        gt_path = filedialog.askopenfilename(
            title="Select Ground Truth Mask",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not gt_path:
            return

        # Read the given path and load GT
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = self.segmentation_mask.astype(np.uint8) * 255

        # Check that shapes align => Else the user specified a wrong GT path
        if gt.shape != pred.shape:
            messagebox.showerror("Error", f"GT shape {gt.shape} != predicted {pred.shape}")
            return

        # Binarize
        gt_bin = (gt > 127).astype(np.uint8)
        pred_bin = (pred > 127).astype(np.uint8)

        # IoU Calculation
        intersection = np.logical_and(gt_bin, pred_bin).sum()
        union = np.logical_or(gt_bin, pred_bin).sum()
        iou = intersection / union if union > 0 else 0.0

        # Difference 
        diff = np.abs(gt_bin - pred_bin) * 255

        # -----------------------------------------------------
        #  TKINTER Message Box for result
        # -----------------------------------------------------
        win = Toplevel(self.root)
        win.title("IoU Result")
        win.geometry("900x350")

        # IoU Label
        lbl = tk.Label(win, text=f"IoU = {iou:.4f}", font=("Arial", 14))
        lbl.pack(pady=10)

        # Matplotlib Figure
        fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=100)

        axs[0].imshow(gt_bin, cmap="gray")
        axs[0].set_title("Ground Truth")
        axs[0].axis("off")

        axs[1].imshow(pred_bin, cmap="gray")
        axs[1].set_title("Predicted Mask")
        axs[1].axis("off")

        axs[2].imshow(diff, cmap="hot")
        axs[2].set_title("Difference (XOR)")
        axs[2].axis("off")

        fig.tight_layout()

        # Embedd the widget
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack()

        # Close botton
        tk.Button(win, text="Close", command=win.destroy).pack(pady=10)


# Main application
if __name__ == "__main__":
    # Import your GraphCutUnary class
    # Assuming it's in the same file or imported from another module
    # from your_module import GraphCutUnary

    root = tk.Tk()

    # Pass your GraphCutUnary class to the GUI
    app = InteractiveSegmentationGUI(root, GraphCutUnary)

    # For standalone testing without GraphCutUnary:
    print("To use this GUI, run it with your GraphCutUnary class:")
    print("app = InteractiveSegmentationGUI(root, GraphCutUnary)")
    print("root.mainloop()")

    root.mainloop()
