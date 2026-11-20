# Interactive Image Segmentation with Graph Cuts

How do you cut an object out of a photo when all you have is a few scribbles on it, one stroke on the object and one on the background? **Graph cuts** turn that question into an energy-minimisation problem that can be solved *exactly* and fast. This project builds the whole pipeline: the energy model, the min-cut solver, a probabilistic appearance model, and an interactive tool to correct the result.

## The idea: segmentation as a labelling problem

Every pixel must receive a label (foreground or background). A good labelling has two properties, and the energy function scores both:

- **Data term (unary):** each pixel should get the label that its colour suggests. If a pixel looks like the scribbled foreground, labelling it "background" is expensive.
- **Smoothness term (pairwise):** neighbouring pixels should usually share a label, *except* where the image has a strong edge. Cutting across a uniform region is expensive, cutting along an edge is cheap.

This is a **Markov Random Field (MRF)**. With two labels, the minimum-energy labelling is exactly the **minimum cut** of a graph in which pixels are nodes, connected to a source (foreground) and a sink (background), and to each other. Max-flow algorithms find it efficiently. With more than two labels, the **alpha-expansion** move repeatedly asks "should these pixels switch to label α?", which is again a binary min-cut, and converges to a strong approximate solution.

## Where do the unary costs come from?

From the colours under the user's scribbles. Two models were compared:

| Appearance model | Idea | Behaviour |
|---|---|---|
| **Histogram** | Count colours under the scribbles, no assumptions about their shape | Robust, stable across very different images |
| **Gaussian Mixture Model (GMM)** | Fit K Gaussians with EM to the scribbled colours | More expressive, higher peak quality when colours form clean clusters, but sensitive to initialisation and can fail badly |

## Core algorithm vs. interactive tool

The **core algorithm** runs with fixed scribbles and works well when foreground and background have clearly different colours. Its typical failures are leakage through low-contrast boundaries and under-segmentation when the object resembles the background.

The **interactive tool** lets a user add a few more coarse strokes exactly where the result is wrong and re-runs the cut. The scribbles are deliberately quick and rough, comparable in effort to the initial ones, so the comparison is fair. Results are judged against ground truth with **IoU**, and with **XOR error maps** that show *where* the errors are (false positives and false negatives), which an IoU number alone cannot.

## What the results show

- Interactive correction **consistently improves IoU** over the core algorithm.
- A few targeted strokes in ambiguous areas help more than many strokes in easy ones. They fix the weak unary costs that caused the error, so the smoothness term no longer dominates wrongly.
- Failures of the core method happen where foreground and background colours are similar, because the data term is weak and the pairwise term wins.
- Histogram models were the most stable overall; GMMs reached higher peaks but with less reliability.

## What you can take from this

1. A hard vision problem became a clean optimisation problem: define what a good labelling costs, then let an exact solver do the work.
2. Models that balance data fidelity against spatial smoothness are everywhere in vision, not only in segmentation.
3. Keeping a human in the loop is often more effective than perfecting a fully automatic method.

## Files

- `final_graphcut_core.py`: MRF construction, alpha-expansion, min-cut, histogram and GMM appearance models
- `interactive_tool.py`: the scribble-based GUI for interactive refinement
