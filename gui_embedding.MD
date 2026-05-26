## Technical Specification: Dense Embedding Local Self-Similarity GUI

### 1. Conceptual Overview
The objective is to visualize the semantic consistency of a deep vision model's feature space relative to a user-selected anchor point. Given an image, a vision backbone extracts a dense feature map. When a user hovers over a spatial coordinate (the "anchor"), the system computes the cosine similarity between the embedding at that anchor and the embeddings of all other spatial coordinates. The resulting scalar matrix is normalized and rendered as a heatmap.

---

### 2. Data Shape & Mathematical Logic

#### Step A: Feature Extraction
The input image is processed through a vision model (e.g., DINOv2, ViT, or a dense CNN backbone) to yield a dense, high-dimensional feature tensor:

$$F \in \mathbb{R}^{H' \times W' \times D}$$

Where:
* $H', W'$ are the spatial dimensions of the downsampled feature map (or patch grid).
* $D$ is the embedding dimension (e.g., $D = 384$ or $768$).

#### Step B: Anchor Selection
When the user hovers over the GUI, the screen-space pixel coordinates $(x_{pixel}, y_{pixel})$ are mapped via nearest-neighbor or bilinear scaling to the feature map coordinates $(i_{anchor}, j_{anchor})$, where $0 \le i_{anchor} < H'$ and $0 \le j_{anchor} < W'$. 

The anchor embedding vector is extracted:

$$\mathbf{v}_{anchor} = F[i_{anchor}, j_{anchor}, :] \quad \in \mathbb{R}^{D}$$

#### Step C: Vectorized Similarity Computation
To compute the similarity map $S \in \mathbb{R}^{H' \times W'}$, calculate the cosine similarity between $\mathbf{v}_{anchor}$ and every spatial vector $\mathbf{v}_{i,j} = F[i, j, :]$ in $F$:

$$S[i, j] = \frac{\mathbf{v}_{anchor} \cdot \mathbf{v}_{i,j}}{\|\mathbf{v}_{anchor}\|_2 \|\mathbf{v}_{i,j}\|_2}$$

> **Implementation Note for the Agent:** For real-time GUI performance, pre-normalize the entire feature tensor $F$ along the channel dimension $D$ at initialization: $\hat{F} = \frac{F}{\|F\|_2}$. The map can then be computed via a single highly optimized matrix-vector dot product: $S = \hat{F} \times \hat{\mathbf{v}}_{anchor}$.

#### Step D: Dynamic Range Normalization
Because cosine similarity yields values in $[-1, 1]$, raw values often lack visual contrast. Min-Max normalization must be applied dynamically to the current slice $S$ to scale the values into a $[0, 1]$ range for colormap mapping:

$$S_{norm}[i, j] = \frac{S[i, j] - \min(S)}{\max(S) - \min(S)}$$

---

### 3. GUI Layout & Rendering Pipeline

The interface consists of two synchronized rendering layers or canvas viewports running a real-time event loop (a Source Image Viewport and a Similarity Heatmap Viewport).

#### Interactive Canvas Interactions
1.  **Mouse Hover Listener:** Bind an `onMouseMove` event listener to the source image viewport.
2.  **Coordinate Translation:** Coordinate transform the cursor position to the native resolution of the feature tensor dimensions $(H', W')$.
3.  **Heatmap Refresh:** Recompute $S_{norm}$ instantly on the CPU/GPU thread based on the translated coordinate.
4.  **Colormap Transformation:** Map the scalar values of $S_{norm}$ using a perceptual sequential colormap (such as `magma`, `inferno`, or `plasma`) to generate RGB/RGBA textures.
5.  **Anchor Indicator (The Single Black Pixel):** Explicitly overwrite the RGB value of the target coordinate array at index `[i_anchor, j_anchor]` to `[0, 0, 0]` (Black) or a cyan-bordered mask to clearly denote the anchor point without bleeding into the heatmap array calculations.
6.  **Upscaling Output:** Render the low-resolution heatmap $(H' \times W')$ to the target display canvas using **nearest-neighbor interpolation** (rather than bilinear filtering) to preserve the distinct, clean block layout of the underlying feature patches.

+-------------------------------------------------------+
|  [GUI Window]                                         |
|                                                       |
|  +-----------------------+   +---------------------+  |
|  | Source Image Canvas   |   | Similarity Heatmap  |  |
|  |                       |   |                     |  |
|  |      (x, y)           |   |       (i, j)        |  |
|  |         o             |-->|         ▪ [Target]  |  |
|  |     [Hover]           |   |    (Inferno/Magma)  |  |
|  +-----------------------+   +---------------------+  |
+-------------------------------------------------------+