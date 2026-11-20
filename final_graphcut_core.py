import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import cv2
import glob


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
        return unary

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
        return unary

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


def plot_segmentation_results(gc, seg_mask, gt_mask, iou, image_name, technique_name="Histogram-based Graph Cut"):
    """
    Visualize segmentation results in 4 panels:
    1) Input Image (RGB)
    2) User Annotation (Red=BG, White=FG)
    3) Graph Cut Result (IoU shown in title)
    4) Ground Truth mask (binary)

    Parameters:
        gc: GraphCutUnary instance
        seg_mask: segmentation mask (binary)
        gt_mask: ground truth mask (binary)
        iou: IoU score for the segmentation
        technique_name: string, technical method name to include in figure title
    """
    plt.figure(figsize=(14, 4))
    plt.suptitle(f"{image_name} - {technique_name}")

    # 1) Input image (RGB)
    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(gc.img_rgb)
    plt.axis('off')

    # 2) User annotation
    plt.subplot(1, 4, 2)
    plt.title("User Annotation (Red=BG, White=FG)")
    plt.imshow(gc.scrib_rgb)
    plt.axis('off')

    # 3) Graph Cut result with IoU
    plt.subplot(1, 4, 3)
    plt.title(f"Graph Cut Result IoU:{iou:.3f}")
    plt.imshow(seg_mask, cmap='gray')
    plt.axis('off')

    # 4) Ground Truth
    plt.subplot(1, 4, 4)
    plt.title("Ground Truth")
    plt.imshow(gt_mask, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def histogram_main():
    """
    Unified main function:
    - Finds images, scribbles and GT files automatically in the folders.
    - Uses histogram-based unary only.
    - Uses the compute_pairwise function as implemented above.
    - For each image it displays a figure with 4 panels:
        Top title: image filename
        1) Input Image (RGB)
        2) User Annotation (Red = BG, White = FG)
        3) Graph Cut Result (IoU shown in title)
        4) Ground Truth mask (binary)
    - Applies a small morphological opening to the segmentation before IoU computation.
    - Ensures mask polarity matches GT by choosing the orientation with higher IoU.
    """
    print("*" * 80)
    print("HISTOGRAM BASED UNARY")
    print("*" * 80)

    # Folders containing your dataset
    img_folder = 'dataset/images'
    scrib_folder = 'dataset/images-labels'
    gt_folder = 'dataset/images-gt'

    # Collect files (sorted for consistent matching)
    img_paths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')) +
                       glob.glob(os.path.join(img_folder, '*.png')))
    scrib_paths = sorted(glob.glob(os.path.join(scrib_folder, '*-anno.png')))
    gt_paths = sorted(glob.glob(os.path.join(gt_folder, '*.png')))

    n = len(img_paths)
    print(f"Found {n} images.")

    iou_scores = dict()

    for i in range(n):
        img_path = img_paths[i]
        scrib_path = scrib_paths[i] if i < len(scrib_paths) else None
        gt_path = gt_paths[i] if i < len(gt_paths) else None

        # Create GraphCutUnary instance
        img = cv2.imread(img_path)
        scrib = cv2.imread(scrib_path)
        gc = GraphCutUnary(img, scrib)

        seg = gc.compute_histogram_based_binary_mask()

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_img > 127).astype(np.uint8)

        # Compute IoU for both orientations
        inter1 = np.logical_and(seg, gt_bin).sum()
        union1 = np.logical_or(seg, gt_bin).sum()
        iou_histogram = inter1 / (union1 + 1e-8)

        image_name = img_path.split('/')[-1][:-4]

        iou_scores[image_name] = iou_histogram

        print(f"Image: {image_name}  IoU: {iou_histogram:.4f}")

        plot_segmentation_results(gc, seg, gt_bin, iou_histogram, image_name,
                                  technique_name="Histogram-based Graph Cut")

    # Print average IoU across dataset
    mean_iou = np.mean(list(iou_scores.values())) if len(iou_scores) > 0 else 0.0
    print(f"Average IoU over {n} images: {mean_iou:.3f}")

    return iou_scores


def gmm_main():
    """
    Unified main function:
    - Finds images, scribbles and GT files automatically in the folders.
    - Uses histogram-based unary only.
    - Uses the compute_pairwise function as implemented above.
    - For each image it displays a figure with 4 panels:
        Top title: image filename
        1) Input Image (RGB)
        2) User Annotation (Red = BG, White = FG)
        3) Graph Cut Result (IoU shown in title)
        4) Ground Truth mask (binary)
    - Applies a small morphological opening to the segmentation before IoU computation.
    - Ensures mask polarity matches GT by choosing the orientation with higher IoU.
    """
    print("\n" + "*" * 80)
    print("GMM BASED UNARY")
    print("*" * 80)

    # Folders containing your dataset
    img_folder = 'dataset/images'
    scrib_folder = 'dataset/images-labels'
    gt_folder = 'dataset/images-gt'

    # Collect files (sorted for consistent matching)
    img_paths = sorted(glob.glob(os.path.join(img_folder, '*.jpg')) +
                       glob.glob(os.path.join(img_folder, '*.png')))
    scrib_paths = sorted(glob.glob(os.path.join(scrib_folder, '*-anno.png')))
    gt_paths = sorted(glob.glob(os.path.join(gt_folder, '*.png')))

    n = len(img_paths)
    print(f"Found {n} images.")

    iou_scores = dict()

    for i in range(n):
        img_path = img_paths[i]
        scrib_path = scrib_paths[i] if i < len(scrib_paths) else None
        gt_path = gt_paths[i] if i < len(gt_paths) else None

        # Create GraphCutUnary instance

        img = cv2.imread(img_path)
        scrib = cv2.imread(scrib_path)

        gc = GraphCutUnary(img, scrib)

        seg = gc.compute_gmm_based_binary_mask(8, 200)

        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_img > 127).astype(np.uint8)

        # Compute IoU for both orientations
        inter1 = np.logical_and(seg, gt_bin).sum()
        union1 = np.logical_or(seg, gt_bin).sum()
        iou_histogram = inter1 / (union1 + 1e-8)

        image_name = img_path.split('/')[-1][:-4]

        iou_scores[image_name] = iou_histogram

        print(f"Image: {image_name}  IoU: {iou_histogram:.4f}")

        plot_segmentation_results(gc, seg, gt_bin, iou_histogram, image_name, technique_name="GMM-based Graph Cut")

    # Print average IoU across dataset
    mean_iou = np.mean(list(iou_scores.values())) if len(iou_scores) > 0 else 0.0
    print(f"Average IoU over {n} images: {mean_iou:.3f}")

    return iou_scores


if __name__ == "__main__":
    hist_scores = histogram_main()  # dict: {image: score}
    gmm_scores = gmm_main()  # dict: {image: score}

    print("\n================= PER IMAGE BEST METHOD =================\n")

    all_images = sorted(hist_scores.keys())

    for img in all_images:
        h = hist_scores[img]
        g = gmm_scores[img]

        if h > g:
            best = "Histogram"
            best_score = h
        else:
            best = "GMM"
            best_score = g

        print(f"{img:25s}  Histogram={h:.4f}   GMM={g:.4f}   → Best: {best} ({best_score:.4f})")

    print("\n================= AVERAGE SCORES =================\n")

    hist_avg = sum(hist_scores.values()) / len(hist_scores)
    gmm_avg = sum(gmm_scores.values()) / len(gmm_scores)

    print(f"Histogram Mean IoU: {hist_avg:.4f}")
    print(f"GMM Mean IoU       : {gmm_avg:.4f}")

    if hist_avg > gmm_avg:
        print("\nOverall Best Method: HISTOGRAM\n")
    else:
        print("\nOverall Best Method: GMM\n")
