"""
Generate explanation text files for each type of latent trajectory visualization plot.
These explanations help users understand what each figure shows and how to interpret it.
"""

PLOT_EXPLANATIONS = {
    "per_epoch": """
LATENT TRAJECTORY - PER EPOCH VISUALIZATION
============================================

WHAT THIS FIGURE SHOWS:
-----------------------
This plot visualizes the latent space trajectory for a single 20-minute epoch (window) of fetal heart rate data.
Each point represents one time step (4 seconds) in the 300-step sequence, plotted in a 2D or 3D reduced space.

HOW IT'S GENERATED:
------------------
1. The SeqVaeTeb model processes the input features (scattering transform and phase harmonic features)
   through encoders to produce a latent representation z(t) of dimension 16 for each time step
2. Principal Component Analysis (PCA) or UMAP reduces the 16D latent vectors to 2D/3D for visualization
3. The trajectory is drawn by connecting consecutive time points with lines
4. Color coding shows a chosen feature (e.g., Transfer Entropy, time, or uncertainty)

VISUAL ELEMENTS:
---------------
- LINE/PATH: The trajectory of the latent representation over the 20-minute epoch
- COLOR GRADIENT: Typically shows Transfer Entropy (TE) - how much information flows from
  uterine pressure to FHR at each time point. Higher values (yellow/bright) indicate stronger coupling.
- START MARKER (circle): Beginning of the 20-minute window
- END MARKER (X): End of the 20-minute window
- AXES: PC1/PC2 (principal components) or UMAP1/UMAP2 dimensions

HOW TO INTERPRET:
-----------------
- TRAJECTORY SHAPE: Loops or returns to previous regions suggest periodic/cyclic behavior
- TRAJECTORY SMOOTHNESS: Smooth paths indicate gradual transitions; abrupt changes suggest sudden events
- COLOR PATTERNS:
  * High TE (bright colors): Strong maternal-fetal coupling at that moment
  * Low TE (dark colors): Weak coupling, more autonomous fetal behavior
- SPATIAL CLUSTERING: Trajectories from similar clinical conditions should cluster together
- START-TO-END DISTANCE: Large distances suggest significant state change during the epoch

CLINICAL RELEVANCE:
-------------------
- Normal labor: Typically shows progressive trajectories with moderate TE values
- Fetal distress: May show erratic paths, clustering in specific regions, or high TE values
- Different patterns emerge for different outcomes (normal, acidosis, HIE)

TECHNICAL NOTES:
----------------
- Time step: Each point represents 4 seconds (300 steps × 4s = 1200s = 20 minutes)
- Latent dimension: 16D compressed to 2D/3D for visualization
- Reduction method: PCA (linear) preserves global structure; UMAP (nonlinear) preserves local structure
""",

    "dynamics": """
LATENT DYNAMICS - SPEED AND ACCELERATION CURVES
================================================

WHAT THIS FIGURE SHOWS:
-----------------------
These plots show the temporal dynamics of the latent trajectory: how fast it's changing (speed)
and how the rate of change itself is changing (acceleration/curvature).

HOW IT'S GENERATED:
------------------
1. For each time step t, we have a latent vector z(t) in 16D space
2. Speed at time t: ||z(t) - z(t-1)|| = Euclidean distance between consecutive latent points
3. Acceleration at time t: ||Δz(t) - Δz(t-1)|| = change in the velocity vector
4. Both are plotted against time (in seconds) for the entire 20-minute epoch

VISUAL ELEMENTS:
---------------
- X-AXIS: Time in seconds (0 to 1200 for a 20-minute epoch)
- Y-AXIS (Speed plot): Magnitude of change in latent space (arbitrary units)
- Y-AXIS (Accel plot): Rate of change of the velocity (arbitrary units)
- LINE: Continuous curve showing dynamics over time

HOW TO INTERPRET:
-----------------

SPEED PLOT:
- HIGH SPEED: Rapid transitions in the latent space
  * Could indicate: changing physiological states, responses to contractions, clinical events
- LOW SPEED: Stable/steady states in latent space
  * Could indicate: homeostatic periods, between contractions
- SPEED PEAKS: Sudden events or state transitions
  * Look for corresponding clinical events (e.g., contraction peaks, interventions)

ACCELERATION PLOT:
- HIGH ACCELERATION: Rapid changes in the rate of change (high curvature in trajectory)
  * Could indicate: sharp turns in the trajectory, abrupt physiological responses
- NEAR-ZERO ACCELERATION: Linear/smooth transitions
  * Could indicate: gradual state evolution, smooth physiological adaptation
- OSCILLATIONS: Regular patterns may reflect intrinsic rhythms (e.g., heart rate variability cycles)

CLINICAL RELEVANCE:
-------------------
- Normal labor: Typically shows rhythmic speed variations aligned with contractions
- Fetal distress: May show sudden speed spikes or sustained high acceleration
- Pattern analysis: Different pathologies may have characteristic speed/acceleration signatures
- Temporal alignment: Compare with actual FHR/UC traces to understand what drives the dynamics

TECHNICAL NOTES:
----------------
- Speed = ||dz/dt|| ≈ ||z(t) - z(t-1)||/Δt where Δt = 4 seconds
- Acceleration = ||d²z/dt²|| ≈ ||Δv(t)||/Δt where v is velocity
- Units are in latent space distances (not directly interpretable in physical units)
- High values don't necessarily mean clinical significance - interpret in context
""",

    "uncertainty": """
UNCERTAINTY-WEIGHTED LATENT TRAJECTORY
======================================

WHAT THIS FIGURE SHOWS:
-----------------------
This visualization overlays the model's uncertainty about its latent representation onto the
trajectory path. Line thickness and transparency vary to show where the model is confident
vs. uncertain about the latent state.

HOW IT'S GENERATED:
------------------
1. The SeqVaeTeb model outputs not just a latent vector z(t) but also its uncertainty
   in the form of log-variance logvar_post(t) from the posterior distribution q(z|x,y)
2. Uncertainty at time t = exp(logvar_post(t)) summed over all latent dimensions
3. The trajectory line is drawn with:
   - LINE THICKNESS: Thicker where uncertainty is higher
   - ALPHA/TRANSPARENCY: More transparent where uncertainty is lower
4. Normalization: Uncertainty is scaled from min to max within each epoch for visualization

VISUAL ELEMENTS:
---------------
- THIN/TRANSPARENT LINES: Low uncertainty - model is confident about the latent state
- THICK/OPAQUE LINES: High uncertainty - model is uncertain about the latent representation
- START MARKER (circle): Beginning of epoch
- END MARKER (X): End of epoch
- AXES: PC1/PC2 or UMAP1/UMAP2 dimensions

HOW TO INTERPRET:
-----------------

HIGH UNCERTAINTY (thick/opaque lines):
- Model is uncertain about the exact latent state
- Could indicate:
  * Ambiguous input signals (noisy data, artifacts)
  * Rare/unusual physiological states not well-represented in training
  * Transition regions between distinct states
  * Data quality issues

LOW UNCERTAINTY (thin/transparent lines):
- Model is confident about the latent representation
- Could indicate:
  * Clean, typical signal patterns
  * Well-represented states from training data
  * Stable physiological conditions

PATTERNS TO LOOK FOR:
- Consistently high uncertainty: May suggest data quality issues or unusual case
- Uncertainty spikes: Could align with clinical events or artifacts
- Uncertainty at trajectory turns: Model uncertain during state transitions
- Start vs. end uncertainty: Temporal patterns in confidence

CLINICAL RELEVANCE:
-------------------
- Diagnostic confidence: Low uncertainty regions provide more reliable clinical insights
- Decision support: High uncertainty should trigger caution in automated predictions
- Quality control: Persistent high uncertainty may indicate technical/sensor issues
- Training data coverage: High uncertainty in certain regions suggests those states
  were underrepresented in the training dataset

TECHNICAL NOTES:
----------------
- Uncertainty comes from the variational posterior: q(z|x,y) = N(μ_post, exp(logvar_post))
- Total uncertainty = Σ exp(logvar_post_i) over all i=1..16 latent dimensions
- This represents the model's epistemic uncertainty (knowledge uncertainty)
- Does NOT capture aleatoric uncertainty (inherent signal noise)
- Normalized per epoch, so thickness is relative within each 20-minute window
""",

    "recurrence": """
RECURRENCE PLOT - LATENT SPACE SELF-SIMILARITY
==============================================

WHAT THIS FIGURE SHOWS:
-----------------------
A recurrence plot visualizes when the latent trajectory revisits similar states over time.
It shows self-similarity patterns, periodic behavior, and state transitions in the latent space.

HOW IT'S GENERATED:
------------------
1. For each time step, we have a 16D latent vector z(t)
2. Compute pairwise Euclidean distances: D(i,j) = ||z(i) - z(j)|| for all time pairs
3. Create a square matrix (300×300 for a 20-minute epoch) where:
   - Rows and columns represent time steps
   - Cell color shows the distance between latent states at those times
4. Plot as a heatmap with time on both axes

VISUAL ELEMENTS:
---------------
- AXES: Both axes represent time (0-300 time steps = 0-1200 seconds)
- COLOR SCALE:
  * Dark/black: Small distance (states are similar/recurrent)
  * Bright/white: Large distance (states are different)
- DIAGONAL: Always dark (distance from a point to itself = 0)

HOW TO INTERPRET:
-----------------

DIAGONAL LINES (parallel to main diagonal):
- Indicate periods where the trajectory evolves similarly
- Could represent: repetitive physiological patterns, regular contraction cycles

VERTICAL/HORIZONTAL LINES:
- Show that one time point is similar to many others
- Could represent: persistent states, plateaus in the trajectory

CHECKERBOARD PATTERNS:
- Indicate alternation between two distinct states
- Could represent: oscillatory behavior, cycling between conditions

HOMOGENEOUS BLOCKS:
- Large dark regions indicate sustained similarity over time
- Could represent: stable physiological states, prolonged steady conditions

WHITE BANDS:
- Indicate times when the state is very different from all others
- Could represent: unique events, transitions, anomalies

SYMMETRY:
- The plot is symmetric across the diagonal (D(i,j) = D(j,i))
- Asymmetric patterns would indicate calculation errors

CLINICAL RELEVANCE:
-------------------
- Normal labor: Often shows periodic patterns reflecting contraction cycles
- Pathological patterns:
  * Excessive recurrence (too dark): May indicate restricted state space (reduced variability)
  * No recurrence (too white): May indicate erratic, unpredictable behavior
- State persistence: Long diagonal segments suggest stable conditions
- Transition detection: Abrupt changes from dark to white regions mark state shifts

QUANTITATIVE ANALYSIS (RQA - Recurrence Quantification Analysis):
- RECURRENCE RATE: % of dark points (overall self-similarity)
- DETERMINISM: % of points forming diagonal lines (predictable evolution)
- LAMINARITY: % of points forming vertical/horizontal lines (state persistence)
- Entropy measures: Complexity of the recurrence pattern

TECHNICAL NOTES:
----------------
- Distance metric: Euclidean in 16D latent space
- Time resolution: 4 seconds per step
- Matrix size: T×T where T=300 for standard 20-minute epochs
- No thresholding applied - shows continuous distance values
- Interpretation requires domain expertise - patterns alone don't diagnose conditions
""",

    "per_signal": """
STITCHED TRAJECTORY - ALL EPOCHS OF ONE SIGNAL
==============================================

WHAT THIS FIGURE SHOWS:
-----------------------
This plot overlays multiple 20-minute epochs from the same patient/signal in a shared
latent space, allowing you to see the evolution of latent states across hours of monitoring.

HOW IT'S GENERATED:
------------------
1. Each 20-minute epoch from the same patient is encoded into the same latent space
2. All epochs are reduced to 2D/3D using a SINGLE fitted PCA/UMAP model
   (ensures all trajectories are in the same coordinate system)
3. Each epoch's trajectory is plotted as a separate line with different colors
4. Start points of each epoch are marked

VISUAL ELEMENTS:
---------------
- MULTIPLE COLORED LINES: Each represents one 20-minute epoch
- COLOR CODING: Different color per epoch (often sequential in time)
- MARKERS: Start points of each epoch
- AXES: Shared PC1/PC2 or UMAP1/UMAP2 coordinates
- LEGEND: Maps colors to epoch numbers/times

HOW TO INTERPRET:
-----------------

TRAJECTORY CLUSTERING:
- Nearby trajectories: Similar physiological states in those epochs
- Distant trajectories: Distinct states/conditions in those epochs
- Progressive drift: Gradual shift across epochs may show labor progression

TEMPORAL PATTERNS:
- Early epochs vs late epochs: Look for systematic changes over labor
- Cyclic returns: Trajectories revisiting similar regions suggest recurring patterns
- Directional flow: Consistent movement in one direction suggests progressive change

EPOCH RELATIONSHIPS:
- Overlapping paths: Epochs with similar latent dynamics
- Diverging paths: Epochs with distinct physiological signatures
- Connected sequences: Smooth transitions between consecutive epochs
- Disconnected jumps: Abrupt state changes between epochs

CLINICAL RELEVANCE:
-------------------
- Labor progression: Normal labor may show systematic evolution through latent space
- Stable conditions: Clustered trajectories indicate consistent state
- Deterioration: Progressive movement toward specific regions may indicate worsening
- Intervention effects: Abrupt changes after specific epochs may reflect clinical actions
- Outcome prediction: Trajectories ending in specific regions may correlate with outcomes

EXAMPLE PATTERNS:
- Normal labor: Progressive movement through a "path" in latent space
- Fetal distress: Sudden divergence from previous epoch patterns
- Recovery: Return toward earlier healthy trajectory regions

TECHNICAL NOTES:
----------------
- All epochs use the SAME dimensional reduction (fitted on combined data)
- Ensures spatial relationships are meaningful across epochs
- Epoch spacing: Typically 20-minute windows, may be overlapping or non-overlapping
- Missing epochs: Gaps in the sequence appear as disconnected trajectories
- Clinical metadata (contractions, interventions) should be overlaid when available
""",

    "states": """
LATENT STATE TIMELINE - DISCRETE STATE CLUSTERING
=================================================

WHAT THIS FIGURE SHOWS:
-----------------------
This visualization shows how the continuous latent trajectory can be segmented into
discrete "states" using clustering. It displays which state is active at each time
across multiple 20-minute epochs for one patient.

HOW IT'S GENERATED:
------------------
1. K-means clustering is applied to all 16D latent vectors across all epochs
2. Each time point is assigned to one of K discrete states (typically K=4-8)
3. State assignments are plotted over time as a step function
4. Each epoch may show a different sequence of states

VISUAL ELEMENTS:
---------------
- X-AXIS: Absolute time in seconds (spanning multiple epochs)
- Y-AXIS: State ID (discrete integers 0, 1, 2, ... K-1)
- STEP FUNCTION: Horizontal lines show which state is active
- COLOR/LINE STYLE: Different epochs may have different colors
- TRANSITIONS: Vertical jumps indicate state changes

HOW TO INTERPRET:
-----------------

STATE PERSISTENCE:
- Long horizontal segments: Stable states, prolonged physiological conditions
- Rapid oscillations: Frequent state switching, dynamic/unstable conditions
- Return to same state: Cyclic behavior, recurring patterns

STATE SEQUENCES:
- Common patterns: Look for repeated state sequences (e.g., 0→1→2→0)
- Unique sequences: Unusual patterns may indicate special conditions
- Temporal structure: States may correspond to distinct phases (baseline, stress, recovery)

TRANSITION PATTERNS:
- Frequent transitions: High variability, responsive system
- Rare transitions: Stable/locked states, limited dynamics
- Directional flow: Systematic progression through states (e.g., 0→1→2→3)
- Bidirectional flow: Reversible states (e.g., 1↔2)

CLINICAL RELEVANCE:
-------------------

STATE INTERPRETATION (requires clinical context):
- States may correspond to:
  * Different contraction phases (baseline, contraction, recovery)
  * Fetal compensation levels (compensated, partially compensated, decompensated)
  * Maternal-fetal coupling modes (independent, coupled, highly coupled)

DIAGNOSTIC PATTERNS:
- Normal labor: Typically shows regular state cycling
- Fetal distress: May show:
  * Stuck in certain states (reduced variability)
  * Rapid, erratic state switching (instability)
  * Progression toward "bad" states
- Intervention effects: State changes after clinical actions

OUTCOME ASSOCIATION:
- Certain states or state sequences may correlate with:
  * Good outcomes (normal delivery)
  * Adverse outcomes (acidosis, HIE)
  * Need for intervention (C-section)

TEMPORAL ANALYSIS:
- Dwell time: How long in each state (measured in seconds/minutes)
- Transition rate: Frequency of state changes per hour
- State proportion: % of time in each state
- Entropy: Complexity/predictability of state sequence

TECHNICAL NOTES:
----------------
- Clustering: K-means on 16D latent vectors (K typically 4-8)
- Sample size: May be subsampled for computational efficiency (e.g., 100K points)
- State numbering: Arbitrary labels (0,1,2,...) with no inherent order
- HMM alternative: Hidden Markov Models can capture transition probabilities
- Clinical mapping: States must be interpreted with domain knowledge and outcome data
""",

    "classes": """
CLASS COMPARISON - LATENT SPACE DISTRIBUTION
============================================

WHAT THIS FIGURE SHOWS:
-----------------------
This scatter plot visualizes how different clinical outcome classes (e.g., normal,
acidosis, HIE) are distributed in the reduced latent space. It shows whether the
model learns to separate outcomes in an unsupervised manner.

HOW IT'S GENERATED:
------------------
1. Collect latent representations (16D) from all signals across all time points
2. Reduce to 2D using:
   - PCA: Linear projection preserving global variance structure
   - LDA: Supervised projection maximizing class separation
   - UMAP/t-SNE: Nonlinear manifold learning (if enabled)
3. Plot points colored by their clinical outcome class
4. Each point represents one time step (4 seconds) from one patient

VISUAL ELEMENTS:
---------------
- POINTS: Each represents a latent vector z(t) from one time step
- COLORS: Different classes (e.g., blue=normal, orange=acidosis, red=HIE)
- AXES:
  * PCA plot: PC1, PC2 (linear combinations of latent dimensions)
  * LDA plot: LD1, LD2 (directions maximizing class separation)
- DENSITY: Point clustering shows frequently visited regions

HOW TO INTERPRET:
-----------------

CLUSTER SEPARATION:
- Well-separated clusters: Model learns distinct latent representations per class
- Overlapping clusters: Classes share similar latent dynamics (harder to distinguish)
- Partial separation: Some classes distinct, others overlap

CLUSTER SHAPES:
- Compact clusters: Consistent latent dynamics within a class
- Elongated clusters: Class shows variation along specific latent directions
- Multi-modal clusters: Class may have sub-populations or phases

DECISION BOUNDARIES:
- Clear gaps: Strong class discrimination possible
- Gradual transitions: Ambiguous boundary regions (uncertain classifications)
- Outliers: Unusual cases that don't fit class patterns

CLINICAL RELEVANCE:
-------------------

GOOD SEPARATION (distinct clusters):
- Model captures class-specific physiological patterns
- Latent space is informative for outcome prediction
- Different outcomes have different latent signatures

POOR SEPARATION (overlapping):
- Classes may be inherently similar in early stages
- May require longer observation windows to distinguish
- Could indicate need for different features or model architecture

PCA vs LDA COMPARISON:
- PCA (unsupervised): Shows natural data structure
  * If classes separate in PCA: Strong, fundamental differences
- LDA (supervised): Maximizes separation using labels
  * Always shows best possible linear separation
  * Compare to PCA to see if separation is "natural" or "forced"

TEMPORAL PATTERNS:
- Early labor points: May cluster regardless of final outcome
- Late labor points: Should separate if model captures deterioration
- Transition points: May appear in boundary regions

QUANTITATIVE METRICS:
- Silhouette score: How well points cluster by class (-1 to 1, higher is better)
- Davies-Bouldin index: Average similarity ratio of each cluster to its most similar (lower is better)
- Classification CV accuracy: If you train a classifier on this space

TECHNICAL NOTES:
----------------
- Point density: May downsample for visualization (e.g., 50K points max)
- Temporal independence: Points from same patient are NOT independent
- Class imbalance: Unequal point counts may bias visual impression
- Dimensionality: 16D→2D loses information; separation may exist in higher dimensions
- Interpretation: Separation in latent space ≠ guaranteed classification performance
""",

    "vector_field": """
VECTOR FIELD - LATENT FLOW DYNAMICS
===================================

WHAT THIS FIGURE SHOWS:
-----------------------
This visualization shows the "flow" of the latent trajectory in 2D reduced space.
Arrows indicate the direction and magnitude of movement at different regions,
revealing the system's dynamics and attractors.

HOW IT'S GENERATED:
------------------
1. Reduce all latent vectors to 2D (e.g., using PCA)
2. Create a regular grid over the 2D space
3. For each grid point:
   - Find K nearest actual trajectory points (e.g., K=20)
   - Compute average velocity: mean(z(t+1) - z(t)) among neighbors
   - Draw an arrow showing direction and magnitude
4. Background scatter shows actual trajectory points

VISUAL ELEMENTS:
---------------
- BACKGROUND DOTS: Actual latent states visited during monitoring
- ARROWS (quivers):
  * DIRECTION: Where trajectories tend to move from that region
  * LENGTH: Speed of movement (longer = faster)
- ARROW DENSITY: Grid resolution (e.g., 30×30)

HOW TO INTERPRET:
-----------------

FLOW PATTERNS:
- Converging arrows (pointing inward): "Attractor" region - trajectories drawn here
- Diverging arrows (pointing outward): "Repeller" region - trajectories avoid/leave
- Circular/spiraling: Cyclic dynamics, oscillatory behavior
- Uniform flow: Consistent directional movement

ATTRACTORS:
- Fixed-point attractor: Arrows converge to a point (stable equilibrium)
  * Could represent: Homeostatic states, stable baseline conditions
- Limit-cycle attractor: Arrows form a closed loop
  * Could represent: Periodic behaviors (e.g., contraction cycles)

CLINICAL RELEVANCE:
-------------------

NORMAL PATTERNS:
- May show cyclic flow corresponding to contraction-recovery cycles
- Possible attractor basins for healthy states
- Smooth, organized flow patterns

PATHOLOGICAL PATTERNS:
- Trapped in unhealthy attractors: Trajectories stuck in suboptimal regions
- Chaotic flow: Highly irregular, unpredictable dynamics
- One-way flow: Irreversible progression toward problematic states
- Weak flow: Lack of clear dynamics (random drift)

STATE SPACE STRUCTURE:
- Multiple basins: Distinct physiological modes
- Barriers: Regions separating different dynamic regimes
- Transition channels: Narrow passages between states

INTERVENTION PLANNING:
- Attractor identification: Target states to aim for or avoid
- Flow reversal: Where to intervene to redirect dynamics
- Escape routes: How to move from bad to good states

TECHNICAL NOTES:
----------------
- Velocity estimation: v(x,y) ≈ mean(z(t+1) - z(t)) for points near (x,y)
- Grid resolution: Typically 20×30 (balance detail vs. clarity)
- K-nearest neighbors: Typically 15-30 (balance local vs. average flow)
- 2D limitation: True dynamics are in 16D; flow may exist in hidden dimensions
- Temporal averaging: Flow shows average behavior, not instantaneous dynamics
- Stochastic effects: Actual trajectories include noise not shown in average flow

RELATED CONCEPTS:
- Phase portraits in dynamical systems theory
- Attractor landscapes in systems biology
- Vector fields in fluid dynamics (analogy)
""",

    "per_guid_absolute": """
PER-SIGNAL TRAJECTORY WITH EPOCH BOUNDARIES
===========================================

WHAT THIS FIGURE SHOWS:
-----------------------
This plot shows all epochs from a single patient in the same 2D latent space,
with visual markers indicating where each 20-minute epoch begins and ends.
This helps understand long-term progression and epoch-to-epoch transitions.

HOW IT'S GENERATED:
------------------
1. All epochs from the same patient are encoded into 16D latent space
2. Dimensionality reduction (PCA/UMAP) is applied consistently across all epochs
3. Each epoch's trajectory is drawn as a connected path
4. Epoch boundaries are marked with distinct symbols or colors
5. Temporal information (absolute time) is encoded in the visualization

VISUAL ELEMENTS:
---------------
- CONTINUOUS PATH: Connected trajectory spanning multiple hours
- EPOCH MARKERS: Visual indicators (circles, diamonds) at epoch boundaries
- COLOR GRADIENT: May encode absolute time or epoch number
- AXES: PC1/PC2 or UMAP1/UMAP2 dimensions
- ANNOTATIONS: Epoch numbers or timestamps

HOW TO INTERPRET:
-----------------

LONG-TERM PATTERNS:
- Progressive drift: Systematic movement over hours indicates labor progression
- Returns to origin: Cyclic behavior, homeostatic regulation
- Directional flow: Unidirectional movement may indicate deterioration or recovery
- Wandering: Random-walk-like behavior suggests no clear progression

EPOCH-TO-EPOCH TRANSITIONS:
- Smooth transitions: Continuous path across epochs (good temporal consistency)
- Discontinuous jumps: Abrupt changes between epochs
  * Could indicate: Clinical interventions, position changes, artifacts
  * Or: Rapid physiological state changes
- Overlapping paths: Different epochs visit similar latent regions

TEMPORAL STRUCTURE:
- Early epochs (start): Typically represent baseline/early labor
- Middle epochs: Active labor phase
- Late epochs (end): Near delivery or outcome
- Look for systematic changes from early→middle→late

SPATIAL PATTERNS:
- Confined region: Limited state space exploration (reduced variability)
- Wide exploration: Rich dynamics, diverse states
- Specific corridors: Preferred pathways through latent space

CLINICAL RELEVANCE:
-------------------

LABOR PROGRESSION:
- Normal: May show structured progression through latent space
- Arrested labor: Trajectories stuck in certain regions
- Rapid progression: Fast movement through latent space

FETAL CONDITION:
- Stable: Trajectories remain in "healthy" regions
- Deteriorating: Progressive movement toward "distress" regions
- Recovering: Movement back toward healthy regions after intervention

OUTCOME PREDICTION:
- Terminal regions: Where trajectories end may correlate with outcomes
- Pathway analysis: How trajectories reach endpoints may be prognostic
- Critical points: Specific latent regions may mark "points of no return"

INTERVENTION EFFECTS:
- Pre/post comparison: Changes in trajectory before/after interventions
- State shifts: Discontinuities aligned with clinical actions
- Recovery patterns: Return to baseline after interventions

TECHNICAL NOTES:
----------------
- Absolute time: Each epoch's actual time is preserved (not normalized)
- Temporal gaps: Missing epochs appear as disconnected segments
- Epoch overlap: Some datasets use overlapping windows (e.g., 10-min stride)
- Clock time: May correlate with circadian patterns in some cases
- Patient metadata: Should include information about interventions, position, medications
""",

    "per_guid_time_series": """
PER-SIGNAL TIME SERIES - LATENT DIMENSIONS OVER TIME
====================================================

WHAT THIS FIGURE SHOWS:
-----------------------
This multi-panel time series plot shows how individual latent dimensions and
derived features evolve over hours for a single patient, providing a temporal
view complementary to spatial trajectory plots.

HOW IT'S GENERATED:
------------------
1. For each time step across all epochs, extract:
   - Selected PCA components (e.g., PC1, PC2)
   - Speed (||Δz||) if enabled
   - Transfer Entropy (TE) if available
2. Plot each as a separate time series panel
3. X-axis is absolute time (seconds or hours)
4. Vertical lines or shading may mark epoch boundaries

VISUAL ELEMENTS:
---------------
- MULTIPLE PANELS: Stacked subplots, one per feature
- X-AXIS: Absolute time across all epochs (hours)
- Y-AXIS: Feature value (PC1, PC2, speed, TE, etc.)
- VERTICAL LINES: Epoch boundaries (20-minute intervals)
- COLOR/STYLE: May vary by epoch or feature type

HOW TO INTERPRET:
-----------------

INDIVIDUAL DIMENSIONS (PC1, PC2):
- Trends: Systematic increase/decrease over time
  * Could indicate: Labor progression, fetal condition changes
- Oscillations: Periodic patterns
  * Could indicate: Contraction cycles, fetal heart rate variability patterns
- Level shifts: Abrupt changes
  * Could indicate: Interventions, position changes, state transitions
- Baseline drift: Gradual changes over hours

SPEED (LATENT DYNAMICS):
- High values: Rapid state transitions
- Low values: Stable periods
- Rhythmic patterns: Regular cycles (e.g., contraction frequency)
- Increasing trend: Progressive instability or increasing variability
- Decreasing trend: Stabilization or reduced responsiveness

TRANSFER ENTROPY (TE):
- High values: Strong maternal-fetal coupling
- Low values: Autonomous fetal regulation
- Patterns:
  * Sustained high TE: Persistent coupling (may indicate stress)
  * Sustained low TE: Fetal autonomy (can be normal or concerning depending on context)
  * Oscillating TE: Dynamic coupling varying with contractions

CROSS-FEATURE PATTERNS:
- Correlations: When speed increases, does TE also increase?
- Lead-lag: Does PC1 change before speed changes?
- Coupled oscillations: Do features cycle together?

CLINICAL RELEVANCE:
-------------------

LABOR STAGES:
- Latent phase: May show specific PC values and low speed
- Active phase: Progressive changes in PCs, higher speed
- Transition: Rapid changes in all features
- Second stage: Distinct pattern (if data extends to delivery)

FETAL WELL-BEING:
- Healthy: Maintained variability, appropriate TE levels
- Compensated stress: Elevated TE, increased speed, still variable
- Decompensated: Loss of variability, extreme TE values, erratic speed

INTERVENTION MARKERS:
- Epidural: May see changes in TE and PCA dimensions
- Position change: Abrupt shifts in multiple features
- Augmentation (oxytocin): May affect coupling patterns
- Fetal scalp stimulation: Brief responses in speed/TE

PREDICTIVE FEATURES:
- Early warning signs: Gradual adverse trends before acute events
- Pattern recognition: Characteristic sequences preceding outcomes
- Baseline establishment: First few epochs set reference

TEMPORAL RESOLUTION:
- 4-second resolution reveals:
  * Individual contraction effects (each ~ 60-90 sec)
  * Heart rate variability patterns (faster than 4 sec may be smoothed)
- Hour-scale patterns reveal:
  * Labor progression
  * Fatigue accumulation
  * Response to interventions

TECHNICAL NOTES:
----------------
- Time alignment: All epochs plotted on absolute time axis
- Missing data: Gaps appear as disconnected line segments
- Smoothing: Optional moving average may be applied for clarity
- Normalization: Features may be z-scored or min-max normalized
- Epoch boundaries: Typically marked to help identify artifacts or discontinuities
- Complementary to spatial plots: Time series shows "what" changes, trajectories show "where"
""",

    "datasets": """
DATASET COMPARISON - MULTI-SOURCE ANALYSIS
==========================================

WHAT THIS FIGURE SHOWS:
-----------------------
These visualizations compare how different source datasets (e.g., different cohorts,
hospitals, or clinical conditions) are distributed in the learned latent space.
They help identify dataset-specific patterns, biases, or generalizable features.

HOW IT'S GENERATED:
------------------
1. Data from multiple HDF5 dataset files are processed through the model
2. Each data point is tagged with its source file (e.g., 'acidosis_cs.hdf5', 'test_no_cs.hdf5')
3. Dataset names are simplified for readability (e.g., 'acidosis_cs', 'test_no_cs')
4. Latent representations are reduced to 2D (PCA/UMAP) and colored by dataset source
5. Additional plots show dataset × class interactions

VISUAL ELEMENTS:
---------------

DATASET COMPARISON PLOT:
- SCATTER POINTS: Each represents one time step from one patient
- COLORS: Different colors for each source dataset
- LEGEND: Shows dataset names and point counts
- SUBTITLE: Displays exact counts per dataset for transparency

DATASET-CLASS MATRIX PLOT:
- GRID LAYOUT: One subplot per source dataset
- WITHIN EACH SUBPLOT: Points colored by clinical class/outcome
- ALLOWS: Direct comparison of how classes separate within each dataset

HOW TO INTERPRET:
-----------------

GENERAL INTERPRETATION (NON-TECHNICAL):

OVERLAPPING DATASETS:
- Good news: Model learns generalizable features across datasets
- Suggests: Latent representations capture fundamental physiological patterns
- Implication: Model may generalize well to new data sources

SEPARATED DATASETS:
- Indicates: Dataset-specific biases or systematic differences
- Could be due to:
  * Different patient populations
  * Different recording equipment/protocols
  * Different hospitals or clinical practices
  * Genuine population differences (e.g., different risk profiles)
- Requires: Careful interpretation when applying model to new datasets

PARTIAL OVERLAP:
- Some regions shared, some dataset-specific
- Suggests: Mix of generalizable and dataset-specific patterns
- Common in real clinical data

TECHNICAL INTERPRETATION:

DISTRIBUTION ANALYSIS:
- DENSITY: Where each dataset's points concentrate
- SPREAD: How much of the latent space each dataset covers
- OUTLIERS: Unique cases in each dataset

DATASET BIAS DETECTION:
- Clear separation may indicate:
  * Batch effects (technical variation, not biological)
  * Selection bias in dataset creation
  * Systematic recording differences
  * Population stratification

MODEL GENERALIZATION:
- If training on dataset A, how will it perform on dataset B?
- Overlapping regions: Likely good performance
- Separated regions: May need domain adaptation or transfer learning
- Can guide decisions about:
  * Which datasets to combine for training
  * Whether to use domain adaptation techniques
  * How to weight different datasets during training

DATASET-CLASS INTERACTIONS:
- Does dataset A show good class separation but dataset B doesn't?
  * Could indicate: Dataset quality differences
  * Or: Different clinical populations (e.g., one has more severe cases)
- Do all datasets show similar class patterns?
  * Good sign: Model captures robust class-discriminative features

CLINICAL RELEVANCE:
-------------------

MULTI-CENTER STUDIES:
- Essential for validating model generalization across hospitals
- Identifies whether patterns are universal or site-specific
- Guides deployment strategy (single-center vs. multi-center model)

POPULATION DIFFERENCES:
- May reveal genuine differences between:
  * High-risk vs. low-risk populations
  * Different ethnic/demographic groups
  * Elective vs. emergency cases
  * C-section vs. vaginal delivery trajectories

DATASET QUALITY ASSESSMENT:
- Anomalous patterns may indicate:
  * Data collection issues in specific datasets
  * Preprocessing errors
  * Label quality problems
  * Need for dataset-specific exclusion criteria

FAIRNESS AND BIAS:
- If datasets represent different demographics:
  * Separation could indicate model bias
  * May require fairness interventions
  * Important for equitable clinical deployment

PRACTICAL USE CASES:
--------------------

FOR RESEARCHERS:
1. Validate whether findings from one dataset replicate in another
2. Decide whether to pool datasets or analyze separately
3. Identify confounding factors related to data source
4. Guide feature engineering to improve cross-dataset generalization

FOR CLINICIANS:
1. Understand if the model was trained on data similar to their patient population
2. Assess confidence in model predictions for their specific clinical setting
3. Identify when results might not generalize to their practice

FOR MODEL DEVELOPERS:
1. Detect and mitigate dataset-specific overfitting
2. Apply domain adaptation techniques if needed
3. Create ensemble models that weight datasets appropriately
4. Design training strategies (e.g., balanced sampling across datasets)

EXAMPLES OF PATTERNS:

SCENARIO 1: PERFECT OVERLAP
- All datasets intermixed in latent space
- Interpretation: Model learned dataset-agnostic features
- Confidence: High for cross-dataset generalization

SCENARIO 2: COMPLETE SEPARATION
- Datasets occupy distinct regions
- Interpretation: Systematic differences between sources
- Action needed: Investigate causes, consider domain adaptation

SCENARIO 3: ONE DATASET AS SUBSET
- Dataset A contained within dataset B's region
- Interpretation: Dataset A may be a specialized subset
- Example: High-risk cases (A) are subset of all cases (B)

SCENARIO 4: DATASET-SPECIFIC CLUSTERS
- Each dataset has unique clusters not present in others
- Interpretation: Different datasets capture different clinical scenarios
- Benefit: Combining datasets may improve model comprehensiveness

TECHNICAL NOTES:
----------------
- Point count differences: Datasets may have vastly different sizes; check subtitle for counts
- Sampling: Very large datasets may be downsampled for visualization (does not affect model)
- Temporal effects: If datasets collected at different times, temporal drift may appear
- Label compatibility: Ensure outcome labels mean the same thing across datasets
- Class imbalance: Different datasets may have different class distributions
- Use with caution: Visual separation doesn't always mean poor generalization (need quantitative metrics)
""",
}


def save_plot_explanations(output_dir):
    """
    Save explanation text files in the appropriate subdirectories of the plots folder.

    Args:
        output_dir: Path to the latent_trajectory output directory (contains plots/ subfolder)
    """
    from pathlib import Path

    plots_dir = Path(output_dir) / "plots"
    if not plots_dir.exists():
        print(f"Plots directory not found: {plots_dir}")
        return

    explanations_saved = []

    # Map of directory names to explanation keys
    dir_to_key = {
        "per_epoch": "per_epoch",
        "dynamics": "dynamics",
        "uncertainty": "uncertainty",
        "recurrence": "recurrence",
        "per_signal": "per_signal",
        "states": "states",
        "classes": "classes",
        "vector_field": "vector_field",
        "per_guid_absolute": "per_guid_absolute",
        "per_guid_time_series": "per_guid_time_series",
        "datasets": "datasets",
    }

    for dir_name, key in dir_to_key.items():
        target_dir = plots_dir / dir_name
        if target_dir.exists() and target_dir.is_dir():
            explanation_file = target_dir / "README.txt"
            explanation_file.write_text(PLOT_EXPLANATIONS[key], encoding='utf-8')
            explanations_saved.append(str(explanation_file.relative_to(output_dir)))
            print(f"[OK] Saved: {explanation_file.relative_to(output_dir)}")

    # Also save a master explanation file in the plots directory
    master_file = plots_dir / "PLOT_EXPLANATIONS.txt"
    master_content = """
LATENT TRAJECTORY ANALYSIS - COMPREHENSIVE PLOT GUIDE
=====================================================

This directory contains various visualizations of the latent space trajectories
learned by the SeqVaeTeb model. Each subdirectory focuses on a different aspect
of the latent dynamics.

DIRECTORY STRUCTURE:
-------------------
"""

    for dir_name in sorted(dir_to_key.keys()):
        target_dir = plots_dir / dir_name
        if target_dir.exists():
            master_content += f"\n{dir_name}/\n"
            # Add first line of explanation as summary
            first_line = PLOT_EXPLANATIONS[dir_to_key[dir_name]].split('\n')[1].strip()
            master_content += f"  └─ {first_line}\n"
            master_content += f"     (See {dir_name}/README.txt for details)\n"

    master_content += """

GENERAL INTERPRETATION GUIDELINES:
---------------------------------

1. LATENT SPACE FUNDAMENTALS:
   - The model compresses complex FHR/UC signals into 16 dimensions
   - Visualization reduces this further to 2D/3D for human interpretation
   - Distances and positions encode physiological similarity

2. TIME RESOLUTION:
   - Each point/step represents 4 seconds of monitoring
   - 300 steps = 20-minute epoch = standard analysis window
   - Multiple epochs can span hours of labor

3. TRANSFER ENTROPY (TE):
   - Measures information flow from uterine pressure to fetal heart rate
   - High TE: Strong coupling (fetus highly responsive to contractions)
   - Low TE: Autonomy (fetal heart rate more self-regulated)
   - Context-dependent: Both high and low can be normal or pathological

4. CLINICAL CONTEXT IS ESSENTIAL:
   - Patterns must be interpreted with clinical metadata
   - Same latent pattern may have different meanings in different contexts
   - Always correlate with actual FHR/UC traces and patient history

5. MODEL UNCERTAINTY:
   - Model confidence varies across the latent space
   - Uncertainty plots show where interpretations should be cautious
   - High uncertainty doesn't always mean clinical concern

6. COMPARATIVE ANALYSIS:
   - Compare within-patient over time (progression)
   - Compare between patients with similar conditions (patterns)
   - Compare outcomes retrospectively (predictive features)

For detailed explanations of each plot type, see the README.txt file
in each subdirectory.
"""

    master_file.write_text(master_content, encoding='utf-8')
    explanations_saved.append(str(master_file.relative_to(output_dir)))
    print(f"[OK] Saved: {master_file.relative_to(output_dir)}")

    print(f"\nTotal explanations saved: {len(explanations_saved)}")
    return explanations_saved


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        # Default to test output directory
        output_dir = Path(__file__).parent.parent.parent / "test_outputs" / "trajectory_test"

    save_plot_explanations(output_dir)
