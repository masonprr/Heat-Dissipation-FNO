# Heat-Dissipation-FNO-Error-Prediction: A Modern Twist on the Hello World of Physics Simulation
Error Confidence Interval Prediction on Fourier Neural Operator for Complex Heat Dissipation Over a Metal Bar: 

**Introduction**
This project explores how to identify and correct systematic prediction errors in Fourier Neural Operators (FNOs) trained on synthetic 1D heat diffusion data. While FNOs have shown strong performance in learning physics-based systems, they can struggle with generalization, especially in complex or noisy parameter spaces. I experimented with multiple post-processing methods — including polynomial regression, spline interpolation, and small neural networks — to correct model errors based on input parameters and the FNO’s own predictions. I also developed tools to analyze whether errors were systematic or random, laying the foundation for a lightweight correction framework that could extend FNO accuracy without requiring retraining. This work demonstrates how error correction can act as a low-cost extension to traditional surrogate models, with applications in scientific modeling, digital twins, and simulation pipelines.

This repo includes:
- A dataset generator with randomized bar length, simulation time, diffusivity, and up to 5 initial pulse types (Gaussian, sine, square, etc.)
- A full training pipeline for a 1D FNO to predict the final temperature profile from the initial condition
- Evaluation and visualization of model accuracy across different complexity regimes
- Post-processing modules that use regression, splines, and neural networks to correct FNO prediction errors
- Error analysis tools to assess whether prediction errors are random or systematic

The goal: build a modular framework to diagnose and fix FNO errors using inexpensive techniques — without retraining the core model.

**Why does this matter?**

- In many physical systems, retraining a large FNO can be expensive. A post-hoc correction strategy could be a cheap and generalizable fix.
- Understanding where and why an FNO fails helps build trust in these models and guides better deployment (e.g. in digital twins, real-time simulations, or scientific ML workflows).
- This also provides a simple framework for evaluating whether a trained model is just memorizing or actually learning physics.

The broader goal is to build a step-by-step pipeline that lets others:
- Train an FNO (or similar model),
- Determine whether its prediction errors are systematic,
- Apply efficient correction strategies to improve generalization and interpretability.

**Data Generation**
To test the FNO’s ability to generalize, I created a synthetic dataset based on the 1D heat equation, modeling how heat diffuses across a bar of varying length and material properties over time. In order for the FNO to be useful, I introduced a number of added parameters to create some error. Through experimentation, I limited complexity to keep the FNO bounded and so I could demonstrate any error prediction capabilities later.

Each training and test sample represents a 1D bar of length L, thermal diffusivity α, and total diffusion time T. The spatial grid was fixed at 300 points, but other parameters were randomized per sample to encourage generalization.

For initial heat distributions, I introduced up to five types of heat pulses per sample, randomly chosen from:
- Gaussian pulses: localized heat spikes
- Sine waves: fluctuating or periodic sources
- Square pulses: sudden on/off heat events

Each pulse had randomized amplitude, location, and width/frequency. These pulses better reflect diverse, realistic heat sources beyond textbook examples.

The forward simulation was solved using an explicit finite difference method. I generated:
- 400–1000 training samples
- 100–200 test samples (varied in structure)

All outputs were fixed on [0, 1] with light noise added to more closely mimic real-world measurements.

**Model Architecture + Training**
I implemented a 1D Fourier Neural Operator (FNO) to learn the mapping from initial temperature profiles to final temperature distributions after diffusion.

The architecture consists of:
- A custom Fourier Layer that transforms input into frequency space
- 4–5 fully connected layers (ReLU activations) that process features
- An output layer projecting back to the original spatial resolution

The model was trained using:

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam (lr = 0.0005)
- Epochs: 1000–2200 (depending on experiment)
- Batch Size: 10

For evaluation, I tracked both training and test loss across datasets of increasing complexity and size. When training on a complex, mixed-pulse dataset with 1000 samples, the FNO achieved:
- Training Loss: ~0.00014
- Test Loss: ~0.01–0.05 (depending on pulse complexity)

To understand where the FNO struggles, I computed absolute and relative errors between predicted and true temperature profiles on the test set.

I visualized:
- Spatial error maps (error vs. position along the bar)
- Sample-wise error distributions
- True vs. predicted temperature curves

Common failure points included:
- Boundaries of the domain (likely due to fewer neighboring points for diffusion)
- Complex multi-pulse initial conditions
- Edge cases in α, T, or L parameter ranges

While the FNO could interpolate well in regions similar to training, it struggled to generalize to unfamiliar combinations of parameters — even with a relatively low training loss. I also computed mean absolute error(MAE) across samples and used this to cluster or flag "hard" vs. "easy" predictions.

**Error Correction Techniques**
After identifying where and how the FNO fails, I explored simple, lightweight ways to post-process predictions and reduce error — rather than retraining or building a more complex model.
Regression-Based Correction

I used the FNO's predicted temperature profile + known parameters (α, T, L) as inputs to regress the prediction error. 

Techniques tested:
- Linear regression
- Polynomial regression (degree 2–6)
- Spline interpolation
- RBF (Radial Basis Function) interpolation

These were trained to predict and subtract the residual error from the FNO’s output. Gains were most noticeable on “hard” samples (multi-pulse, high α or T).

I also trained a shallow fully connected NN Correction: [Initial temperature profile, FNO prediction] → True final profile
This worked well occasionally as a general error corrector, slightly outperforming polynomial fits at times, though never outperforming spline interpolation.

Questions for Further Investigation:

How do we detect when FNO errors are systematic vs. random?
- Can this be determined visually, through parameter-feature correlation, or should it be learned with a separate model?
→   Idea: Build a diagnostic pipeline that flags patterns using simple tools like polynomial fits or PCA.

What level of error is “high enough” for post-correction to be useful?
- If the FNO is already near-perfect, correction seems trivial. But if it's too poor, corrections might just mask failure.
→ Ideally, we want moderate error (~0.01–0.05 test loss) where corrections show measurable gains.

Does the correction method scale with complexity of data?
- When I trained on simpler datasets (e.g. just Gaussian pulses), corrections had less impact because FNO was already accurate.
→ In more realistic, high-diversity datasets, corrections were meaningful.

Is regression a valid baseline, or should it be a real correction method?
- Linear/polynomial fits had R² ≈ 0.15–0.45 on some datasets — not amazing, but occasionally competitive with the FNO itself on low-data regimes.

Can we train a second NN to correct the first FNO?
- I briefly considered whether a second lightweight NN could correct errors in the first — especially if those errors are spatially structured. Perhaps in more complex FNO applications such as in weather prediction, this method could be more lightweight than retraining the entire FNO and also perform better than splines. However, for these experiments it showed little promise, both in computational overhead and in accuracy.

Why do boundary errors dominate?
- Errors often clustered at the edges.
→ Is this due to lack of Dirichlet boundary conditions? Should we simulate realistic edge cooling (e.g. convection to air)?

How realistic are my physical parameters?
- The α range of 0.001–0.02 does not map well to the diffusion coefficients of real materials. That said, it made for easier training and evaluation, especially in this set of experiments meant to illustrate the error-correction framework. This may be worth changing to better reflect real conditions.

When should we stop increasing model size and just post-correct?
- Given that large FNOs can be expensive to train, this project suggests post-processing may offer a cheaper and simpler alternative for moderate improvements. More experimentation is needed to understand the limits of such correction methods on much larger models.
