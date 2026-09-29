# Inverse Cloth Physics: Estimating Fabric Material / Dynamics from (Monocular, Imperfect) Video

Research note scope: state of the art through Sept 2026 on recovering fabric material parameters or learned cloth dynamics from video, with emphasis on (a) the supervision signal each method uses and (b) tolerance to monocular / non-coherent observations. Project context: static 3DGS of a clothed human (per-splat clothing labels + body proxy mesh); only WAN 2.2-generated monocular clips that are individually plausible but mutually inconsistent.

Methodology caveat: arxiv.org, openaccess.thecvf.com, eth-ait.github.io and is.mpg.de were blocked by the egress proxy in this session, so primary PDFs could not be opened. Findings below are drawn from search-result abstracts/snippets of primary pages (arXiv abstract pages, CVF, ACM DL, GitHub, project pages). Details that come only from background knowledge are explicitly marked "(unverified this session)" and are kept in Inferences/Gaps, not in Cited Findings.

---

## Q1. Differentiable cloth simulators and classical "material-from-video" inverse problems

### Takeaway
Gradient-based inverse cloth work (DiffAvatar, PhysAvatar, SAFT, Dress-1-to-3, DiffXPBD) almost always assumes geometric correspondence: a tracked/registered mesh, multi-view silhouettes, or a template aligned to the video. The only lineage that deliberately avoids per-point correspondence is the "measure the fabric from motion statistics" line: Bouman et al. ICCV 2013, Yang et al. ICCV 2017, Runia et al. CVPR 2020 (spectral decomposition), and "Unphased Wrinkles" (frequency-based loss). That lineage is the one to reuse for inconsistent generated clips.

### Cited Findings
**Classical video-based estimation (correspondence-light)**
- Bouman, Xiao, Battaglia, Freeman, "Estimating the Material Properties of Fabric from Video," ICCV 2013. It analyzes videos of fabric moving under unknown wind forces and recovers two properties, stiffness and area weight (mass per area). Contributions: a fabric video database, a prediction algorithm, and a perceptual study of how well humans estimate these properties. — [CVF ICCV 2013](https://openaccess.thecvf.com/content_iccv_2013/html/Bouman_Estimating_the_Material_2013_ICCV_paper.html); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/100042); [author PDF](https://people.csail.mit.edu/klbouman/pw/papers_and_presentations/iccv2013_bouman.pdf)
- Yang, Liang, Lin, "Learning-Based Cloth Material Recovery From Video," ICCV 2017. CNN+LSTM regress cloth material from video. Simulated data is used to learn the mapping from visual appearance and motion dynamics to material, so this is amortized, feed-forward inference trained on synthetic data. — [CVF ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Yang_Learning-Based_Cloth_Material_ICCV_2017_paper.html); [UMD GAMMA](https://gamma.umd.edu/publication/627/); [arXiv 1608.01250](https://arxiv.org/abs/1608.01250v3)
- Runia, Gavrilyuk, Snoek, Smeulders, "Cloth in the Wind: A Case Study of Physical Measurement through Simulation," CVPR 2020. It measures latent physical properties of cloth in wind using a new spectral decomposition layer that encodes the frequency distribution over the cloth surface: the layer decomposes a video volume into its temporal spectral power and the corresponding frequencies. — [CVF CVPR 2020 PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Runia_Cloth_in_the_Wind_A_Case_Study_of_Physical_Measurement_CVPR_2020_paper.pdf); [arXiv 2003.05065](https://arxiv.org/pdf/2003.05065)
- "Unphased Wrinkles: Estimating cloth elasticity parameters using a frequency-based loss" (arXiv 2212.08790, 2022). Earlier methods design experiments that separate stretch from bending. This one accepts that stretching causes wrinkles and adds a loss that measures the spectral similarity of wrinkles between simulated and target cloth. The recovered elasticity parameters stay consistent across different wrinkle patterns of the same fabric, i.e. the loss is phase-invariant and does not need wrinkle-to-wrinkle correspondence. — [arXiv 2212.08790](https://arxiv.org/abs/2212.08790); [HTML v2](https://arxiv.org/html/2212.08790v2)

**Differentiable simulators used for garment / fabric inverse problems**
- DiffXPBD: differentiable position-based simulation of compliant constraint dynamics (XPBD). — [arXiv 2301.01396](https://arxiv.org/pdf/2301.01396)
- DiffAvatar (CVPR 2024, Meta): simulation-ready garment optimization with differentiable simulation. It can optimize material parameters while the clothing is draped on a body, e.g. estimating bending stiffness for garments. — [arXiv 2311.12194](https://arxiv.org/pdf/2311.12194)
- PhysAvatar: combines inverse rendering with inverse physics to estimate a human's shape and appearance, plus the physical parameters of their clothing fabric, from **multi-view** video. It uses mesh-aligned 4D Gaussians for spatio-temporal mesh tracking, a physically based inverse renderer, and a physics simulator whose garment parameters are fit by gradient-based optimization. — [project page](https://qingqing-zhao.github.io/PhysAvatar)
- SAFT, "Shape and Appearance of Fabrics from Template via Differentiable Physical Simulations from Monocular Video," ICCV 2025 (Stotko et al., Uni Bonn). From a **single monocular RGB video** it reconstructs 3D geometry and PBR appearance, using cloth physics simulation plus differentiable rendering in three phases (texture mapping, geometry reconstruction, appearance estimation). Two new regularizers address monocular depth ambiguity. It is template-based, so the template must be aligned to the video. — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Stotko_SAFT_Shape_and_Appearance_of_Fabrics_from_Template_via_Differentiable_ICCV_2025_paper.pdf); [Uni Bonn PDF](https://cg.cs.uni-bonn.de/backend/v1/files/publications/stotko_2025b.pdf); [arXiv 2509.08828](https://arxiv.org/pdf/2509.08828)
- Physics-guided Shape-from-Template uses a neural surrogate of the cloth simulator for monocular video perception. — [arXiv 2311.12796](https://arxiv.org/pdf/2311.12796)
- Dress-1-to-3 (ACM TOG, July 2025). Starting from one in-the-wild image, it produces simulation-ready separated garments with sewing patterns. A pretrained image-to-sewing-pattern model gives a coarse pattern, a multi-view diffusion model generates multi-view images, and a differentiable garment simulator based on Codimensional IPC (CIPC) refines the pattern against them. — [ACM DL](https://dl.acm.org/doi/10.1145/3731177); [project](https://dress-1-to-3.github.io/); [arXiv 2502.03449](https://arxiv.org/abs/2502.03449)
- MulayCap (multi-layer human performance capture from monocular video) also appears in the monocular cloth-capture literature. — [arXiv 2004.05815](https://arxiv.org/pdf/2004.05815)
- Real-to-sim cloth parameter optimization with particle-based simulation from robot-manipulation data (J. Comput. Design & Eng., 2025) shows the same inverse-problem pattern in robotics. — [OUP JCDE](https://academic.oup.com/jcde/article/12/8/29/8206149)

### Inferences
- Background, unverified this session: Liang, Lin, Koltun, "Differentiable Cloth Simulation for Inverse Problems" (NeurIPS 2019, code released); Qiao et al., differentiable ArcSim-based multi-body/cloth simulation (NeurIPS 2021, "DiffArcSim"); DiffCloth (Li et al., ACM TOG 2022, code on GitHub, projective-dynamics based with differentiable dry frictional contact); NVIDIA Warp provides differentiable cloth/FEM/XPBD kernels. All of them fit parameters with losses on vertex positions or point clouds, which require correspondence.
- Background, unverified: Runia et al. train an embedding network (on the spectral-decomposition features) so that real and simulated videos of the same physical parameters land close together, then search simulator parameters (they used ArcSim) to minimize embedding distance. If that is right, it is exactly a "simulated and observed video need not be pixel-aligned" scheme and a direct template for our use case.
- For our pipeline, the SAFT/PhysAvatar/DiffAvatar style (template registered to the video) would require registering the splat/proxy mesh to each WAN clip. That is hard, because geometry drifts between clips and possibly within a clip.

### Gaps
- Could not open the Runia and Unphased Wrinkles PDFs to confirm the exact parameter sets, simulator and quantitative accuracy. The Unphased Wrinkles venue beyond arXiv is unconfirmed.
- Code availability for Runia 2020, Bouman 2013 and SAFT is unconfirmed this session.

---

## Q2. Correspondence-free / distribution-matching supervision (silhouettes, flow statistics, spectra, perceptual/adversarial/feature losses)

### Takeaway
Three families of loss tolerate non-aligned observations:
1. Spectral/frequency losses (Runia 2020, Unphased Wrinkles 2022).
2. Learned embeddings or feed-forward regressors trained on simulation (Bouman 2013, Yang 2017; MatPhys/EgoPhys-style codebooks in 2026).
3. Video-diffusion score distillation (SDS), which scores plausibility rather than alignment (DreamPhysics, Physics3D, OmniPhysGS).

For inconsistent WAN clips, the most defensible design pools evidence across clips at the level of statistics or embeddings (a posterior over a low-dimensional material vector), not pixels.

### Cited Findings
- Runia et al.'s spectral decomposition layer turns a video volume into temporal spectral power and frequencies, a representation that does not need per-pixel correspondence with the simulation. — [arXiv 2003.05065](https://arxiv.org/pdf/2003.05065)
- The Unphased Wrinkles loss compares the spectral similarity of wrinkles, and the recovered parameters are consistent across different wrinkle patterns of the same fabric. — [arXiv 2212.08790](https://arxiv.org/abs/2212.08790)
- DreamPhysics (AAAI 2025, code on GitHub) estimates physical properties of 3D Gaussian splats with video diffusion priors. It optimizes physical parameters by score distillation sampling (image- or text-conditioned) with frame interpolation and a log gradient. "Motion distillation sampling" emphasizes motion information during distillation, and a KAN-based material field with frame boosting eases optimization. — [arXiv 2406.01476](https://arxiv.org/abs/2406.01476); [GitHub](https://github.com/tyhuang0428/DreamPhysics)
- OmniPhysGS combines a memory-efficient MPM solver, "Constitutive Gaussians" (a hardmax selection over 12 expert constitutive models per local neighborhood), and video score distillation from a pretrained text-to-video diffusion model, trained end-to-end by SDS. — [project page](https://wgsxm.github.io/projects/omniphysgs/); [Pith summary](https://pith.science/paper/2501.18982)
- Physics3D learns physical properties of 3D Gaussians via video diffusion. — [OpenReview](https://openreview.net/forum?id=k3JgQXtpJq)
- PhysDreamer distills physics priors by aligning to a reference video produced by a video generation model, optimizing material fields (Young's modulus, Poisson's ratio) with MPM dynamics. — [ResearchGate](https://www.researchgate.net/publication/385087267_PhysDreamer_Physics-Based_Interaction_with_3D_Objects_via_Video_Generation); summarized in [survey arXiv 2503.21765](https://arxiv.org/pdf/2503.21765)
- MatPhys (arXiv 2605.19386, May 2026) uses DINO features to split an object into semantic parts and query a part-level material prior. A learned material codebook of shared embeddings bridges appearance and physics, so the same material gets consistent parameters across scenes. It targets a known failure of per-scene inverse optimization: inconsistent parameters for the same material across scenes or interactions. — [arXiv 2605.19386](https://arxiv.org/abs/2605.19386)

### Inferences
- Background, unverified: PhysDreamer (Zhang et al., ECCV 2024) generates its reference video with an image-to-video model conditioned on a **render of the same static 3DGS scene**, so the reference video is roughly pixel-aligned in early frames and a per-frame rendering loss is usable. This is the closest published analogue to our setup (splat → I2V → material). Its alignment trick only holds for a clip generated from a render at a known camera, and WAN clips drift over time. Our pipeline could copy it: condition each WAN clip on a render of our splat from a known camera, so frame 0 is registered by construction. Supervise early frames with pixel or flow losses and later frames only with correspondence-free losses (spectral, silhouette statistics, feature embeddings).
- MatPhys's cross-scene consistency problem is essentially our "many inconsistent clips" problem. A shared codebook or shared latent material vector per garment, optimized jointly over all clips with per-clip nuisance variables (camera, timing, body motion, wind), is a natural formulation.
- Losses robust to inconsistent clips, ordered by increasing invariance: silhouette/mask IoU after per-clip alignment; optical-flow magnitude histograms and temporal power spectra within garment masks (Runia-style); wrinkle spatial spectra (Unphased-Wrinkles-style); distances in pretrained video features (e.g., VideoMAE/DINO embeddings, as in the MatPhys and feed-forward lines); video-diffusion SDS (DreamPhysics/OmniPhysGS), which needs no observations at all but only yields "plausible," not "matching," dynamics.
- SDS-based physics methods are volumetric-MPM and object-centric, and report plausibility rather than parameter accuracy. They likely cannot tell apart fabrics with similar visual dynamics.

### Gaps
- No published cloth-on-body method was found that explicitly fuses **multiple mutually inconsistent** generated videos into one material estimate. This looks like an open problem, and a publishable contribution.
- No verified quantitative comparison was found between feature-space losses (DINO/VideoMAE) and spectral losses for cloth parameter recovery.

---

## Q3. Learned simulators with material conditioning and amortized material inference

### Takeaway
HOOD/ContourCraft accept per-vertex/per-edge material parameters as input, so the learned simulator itself is material-conditioned. Gaussian Garments exploits this to fit materials to multi-view registered video. Amortized inference (observation → material code) exists in several forms: Yang 2017 (video → material), Image2Garment 2026 (image → fabric attributes → simulator parameters via a VLM plus measured data), MatPhys 2026 and EgoPhys 2026 (video → spring-mass stiffness via a learned codebook). None of the video-based amortized methods targets garments on a moving body, but the recipe transfers.

### Cited Findings
- HOOD (CVPR 2023): a GNN with a hierarchical graph and multi-level message passing, trained with an unsupervised (physics-energy) scheme to predict clothing dynamics for arbitrary garments and body shapes. Augmenting the input material parameters during training lets it simulate different materials, with parameters defined per vertex and per edge. It handles topology and material changes at inference time. — [arXiv 2212.07242](https://arxiv.org/pdf/2212.07242); [MPI page](https://is.mpg.de/ps/projects/hierarchical-graphs-for-generalized-modelling-of-clothing-dynamics); [CVPR 2023 paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Grigorev_HOOD_Hierarchical_Graphs_for_Generalized_Modelling_of_Clothing_Dynamics_CVPR_2023_paper.pdf)
- ContourCraft adds an intersection loss and a collision-avoiding repulsion objective to a GNN-based neural cloth simulator. It recovers from intersections caused by missed collisions, self-penetrating bodies, or errors in hand-designed multi-layer outfits. — [search summary of ContourCraft; see MPI HOOD page](https://is.mpg.de/ps/projects/hierarchical-graphs-for-generalized-modelling-of-clothing-dynamics)
- Gaussian Garments (3DV 2025) reconstructs simulation-ready garments with photorealistic appearance from **multi-view** video. — [arXiv 2409.08189](https://arxiv.org/html/2409.08189v1); [project page](https://eth-ait.github.io/Gaussian-Garments/)
- Image2Garment (arXiv 2601.09658, Jan 2026) is feed-forward. A fine-tuned VLM infers material composition and fabric attributes (composition, fabric family, structure type, weight, thickness, aligned with commercial garment tags). Lightweight Random Forest regressors trained on a small material-physics measurement dataset then map these attributes to simulator parameters: bending, shear, stretch and buckling stiffness, damping, and friction. It introduces the FTAG and T2P datasets, and the outputs import into CLO3D/Marvelous Designer/Browzwear. — [arXiv 2601.09658](https://arxiv.org/abs/2601.09658); [project](https://image2garment.github.io/)
- MatPhys (May 2026) is a feed-forward framework that predicts spring-mass parameters from a **single-view** video of a deformable object under interaction and reconstructs a simulatable digital twin (geometry, appearance, physics). — [arXiv 2605.19386](https://arxiv.org/abs/2605.19386)
- EgoPhys (arXiv 2606.16202, June 2026) builds deformable digital twins from egocentric RGB-only video. It distills per-object inverse-physics solutions into a compact codebook, which predicts dense spring-stiffness fields for unseen objects without per-spring test-time optimization. It was demonstrated on a real xArm6 robot. — [arXiv 2606.16202](https://arxiv.org/abs/2606.16202); [HF papers](https://huggingface.co/papers/2606.16202)
- M-PhyGs (arXiv 2512.16885): multi-material object dynamics from video. Related work there notes that feed-forward estimators, including fine-tuned video transformers, usually assume one material per object. — [arXiv 2512.16885](https://arxiv.org/pdf/2512.16885)
- "Learning 3D-Gaussian Simulators from RGB Videos" (arXiv 2503.24009) learns a Gaussian-particle dynamics model directly from RGB video rather than fitting parameters of an analytic simulator. — [arXiv 2503.24009](https://arxiv.org/pdf/2503.24009)

### Inferences
- Background, unverified: Gaussian Garments fits a small set of ContourCraft/HOOD material parameters (e.g., Lamé stretch coefficients, bending stiffness, density) by gradient descent through the learned simulator, minimizing the distance between simulated garment meshes and the multi-view-registered garment mesh sequence. Its code is on GitHub (eth-ait). Because the GNN is differentiable w.r.t. its material inputs, the same simulator could take any differentiable loss, including correspondence-free ones.
- A practical amortized design for our project: (1) sample materials from a measured-fabric prior (Q5) and body motions; (2) simulate with HOOD/ContourCraft on our proxy body; (3) render via the splat (or a cheap mask/normal renderer); (4) train an encoder (video features → material posterior) on those synthetic renders, with domain randomization over camera and timing; (5) run it on each WAN clip and combine per-clip posteriors (product of Gaussians or a learned set-pooling). This follows Yang 2017, Bouman 2013 and MatPhys/EgoPhys, and it naturally tolerates inconsistency between clips because each clip is only one noisy measurement.
- Alternative: skip explicit parameters and fine-tune the dynamics model itself on generated videos ("Learning 3D-Gaussian Simulators from RGB Videos" style). That needs 3D supervision or consistency across views, which our clips lack, so it is higher risk.

### Gaps
- Could not verify ContourCraft's venue (believed SIGGRAPH 2024) or its exact material parameterization this session.
- Could not verify how many material parameters Gaussian Garments optimizes, or its loss. The project page was blocked.
- No amortized garment-on-body material encoder trained on synthetic HOOD rollouts was found in the literature.

---

## Q4. Physics-based Gaussian methods: cloth/thin-shell vs. volumetric, and supervision

### Takeaway
Most "physics Gaussian" work (PhysGaussian, PhysDreamer, DreamPhysics, Physics3D, OmniPhysGS) uses volumetric MPM with elastic/plastic constitutive models and is object-centric. Cloth-capable methods fall into three groups: spring-mass (Spring-Gaus, PhysTwin, MatPhys, EgoPhys), MPM with an anisotropic/codimensional model (MPMAvatar), and mesh-based garment simulators coupled to Gaussians (PhysAvatar, Gaussian Garments, PGC, MonoCloth). Methods for garments on bodies use multi-view capture. Methods that accept single-view input are either not garment-on-body or use SDS/plausibility supervision.

### Cited Findings
- PhysTwin (ICCV 2025): from sparse videos of deformable objects under interaction, it builds a real-time interactive twin that combines a spring-mass model, generative shape priors for geometry, and Gaussian splats for rendering. Objects include cloth, rope, stuffed animals and packages. Optimization is hierarchical sparse-to-dense: zero-order for non-differentiable parameters, then first-order refinement through a custom differentiable spring-mass simulator. — [arXiv 2503.17973](https://arxiv.org/pdf/2503.17973); [ML Anthology ICCV 2025](https://mlanthology.org/iccv/2025/jiang2025iccv-phystwin/); [ICRA'25 workshop](https://deformable-workshop.github.io/icra2025/spotlight/02_01_03_Jiang_PhysTwin.pdf)
- Spring-mass Gaussian methods such as Spring-Gaus, PhysTwin and NeuSpring combine 3D Gaussians with inverse physical fitting from video. — [PhysTwin summary via search](https://liner.com/review/phystwin-physicsinformed-reconstruction-and-simulation-deformable-objects-from-videos)
- MPMAvatar (NeurIPS 2025, code on GitHub) builds 3D Gaussian avatars from **multi-view** videos with physics-based animation of loose garments. It uses an MPM simulator tailored to garments, with an anisotropic constitutive model and a new body-garment collision algorithm, plus quasi-shadowing in rendering. — [arXiv 2510.01619](https://arxiv.org/abs/2510.01619); [GitHub](https://github.com/KAISTChangmin/MPMAvatar)
- PGC: "Physics-Based Gaussian Cloth from a Single Pose" (arXiv 2503.20779). — [arXiv 2503.20779](https://arxiv.org/pdf/2503.20779)
- MonoCloth: reconstructs and animates cloth-decoupled human avatars from monocular videos (arXiv 2508.04505). — [arXiv 2508.04505](https://arxiv.org/pdf/2508.04505)
- GausSim (arXiv 2412.17804): a Gaussian simulator for elastic objects. — [arXiv 2412.17804](https://arxiv.org/pdf/2412.17804)
- DreamPhysics / OmniPhysGS / Physics3D / PhysDreamer: all MPM-based and supervised with video-diffusion outputs or SDS (see Q2). — [DreamPhysics GitHub](https://github.com/tyhuang0428/DreamPhysics); [OmniPhysGS](https://wgsxm.github.io/projects/omniphysgs/)
- Other 2026 garment-Gaussian works surfaced: Gaussian Wardrobe (compositional 3DGS avatars for virtual try-on, arXiv 2603.04290), DAMA (disentangled body-anchored multi-layer avatars, arXiv 2605.21001), CausalGS (physical causality of dynamic 3DGS scenes, arXiv 2605.10586). — [Gaussian Wardrobe](https://arxiv.org/pdf/2603.04290); [DAMA](https://arxiv.org/pdf/2605.21001); [CausalGS](https://arxiv.org/pdf/2605.10586)

### Inferences
- Background, unverified: PhysGaussian (CVPR 2024) and PhysDreamer (ECCV 2024) treat Gaussians as MPM particles with continuum (volumetric) constitutive laws. Thin garments are under-resolved in volumetric MPM unless an anisotropic/codimensional formulation is used, which is what MPMAvatar adds. PhysTwin used 3 RGB-D views, so it is "sparse," not monocular RGB. Its code is public (GitHub Jianghanxiao/PhysTwin).
- For a clothed human with a body proxy mesh and clothing labels, the Gaussian-Garments route (mesh garment + learned GNN simulator + Gaussians bound to the mesh) or the MPMAvatar route (MPM garment) is more appropriate than volumetric MPM object methods. The inverse-supervision part can be borrowed from PhysDreamer (render-conditioned I2V reference) and from DreamPhysics/OmniPhysGS (SDS as a regularizer).

### Gaps
- Could not confirm whether PGC or MonoCloth estimate material parameters or only use fixed/default ones, nor their venues and code status.
- Vid2Sim, PhysMotion, PhysFlow and "Sim Anything" were not verified this session.

---

## Q5. Fabric property datasets and low-dimensional priors

### Takeaway
Measured fabric databases are small (on the order of ten fabrics in the classic datasets). Newer work maps semantic fabric attributes (composition, weight, weave) to simulator parameters using commercial measurements (Image2Garment, 2026). A low-dimensional prior (fabric type → parameter distribution) is the key regularizer for making monocular estimation well-posed.

### Cited Findings
- Wang, O'Brien, Ramamoorthi, "Data-driven elastic models for cloth: modeling and measurement," SIGGRAPH 2011 / ACM TOG 30(4). A piecewise-linear elastic model approximates nonlinear, anisotropic stretch and bending, with a measurement setup and a database of **ten** cloth materials. — [SIGGRAPH history](https://history.siggraph.org/?p=114658); [eScholarship PDF](https://escholarship.org/content/qt7g35q2x7/qt7g35q2x7.pdf); [DOI](https://api.crossref.org/works/10.1145%2F2010324.1964966)
- Miguel et al., "Data-Driven Estimation of Cloth Simulation Models," Eurographics 2012 (MIT CDFG). — [PDF](https://cdfg.mit.edu/assets/files/DataDrivenCloth-EG2012.pdf); [project](https://cfg.mit.edu/publications/data-driven-estimation-cloth-simulation-models)
- Image2Garment's FTAG and T2P datasets link fabric tags/attributes to physical simulator parameters (bending/shear/stretch/buckling stiffness, damping, friction) through Random Forest regressors trained on a small material-physics measurement set. — [arXiv 2601.09658](https://arxiv.org/abs/2601.09658); [project](https://image2garment.github.io/)
- Bouman 2013 released a fabric-in-wind video database labeled with stiffness and area weight. — [CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Bouman_Estimating_the_Material_2013_ICCV_paper.html)
- Textile IR (arXiv 2601.02792) proposes a bidirectional intermediate representation for physics-aware fashion CAD. It is relevant as a structured fabric-parameter schema. — [arXiv 2601.02792](https://arxiv.org/pdf/2601.02792)

### Inferences
- Background, unverified: Clyde, Teran, Tamstorf (SCA 2017) measured woven fabrics and fit hyperelastic models. Feng et al. (2022, "Learning-based bending stiffness parameter estimation by a drape tester," SIGGRAPH Asia/TOG) used Cusick-style drape tests. Commercial tools (CLO3D/Marvelous Designer, Browzwear) ship presets per fabric type that effectively form a categorical prior. Any of these could supply a prior over the HOOD/ContourCraft material vector, sampled per clothing class (e.g., denim vs. jersey vs. silk).
- Recommended prior structure: a categorical fabric type (from a VLM on the splat renders, Image2Garment-style), giving a Gaussian over log-parameters, refined by video evidence. This gives a well-posed Bayesian update even when clip evidence is weak or contradictory.

### Gaps
- Clyde 2017, Feng 2022 and the Cusick drape-based papers were not verified in this session.
- Whether the FTAG/T2P datasets and Image2Garment code are publicly released is unconfirmed.

---

## Q6. Practical: which parameters matter, identifiability from monocular video, robustness

### Takeaway
Evidence consistently points to a small effective parameter set: bending stiffness and area density (their ratio largely sets drape and wrinkle frequency), with stretch mostly acting as near-inextensibility, and damping and friction as secondary. Bouman's two-parameter (stiffness, area weight) framing and Image2Garment's six-group parameterization bracket this. Monocular identifiability is weak for absolute scales (mass vs. force/stiffness), so priors and ratio-based parameterizations are essential. Frequency/statistics losses are the published way to gain robustness to phase and correspondence errors.

### Cited Findings
- Bouman 2013 recovers just two properties, stiffness and area weight, from video of fabric under **unknown** wind forces, which indicates these two dominate visual appearance of dynamic fabric. — [CVF](https://openaccess.thecvf.com/content_iccv_2013/html/Bouman_Estimating_the_Material_2013_ICCV_paper.html)
- Image2Garment's full simulator-compatible set is bending, shear, stretch and buckling stiffness, damping, and friction. — [project](https://image2garment.github.io/)
- DiffAvatar highlights bending stiffness as the material parameter it estimates for garments draped on a body. — [arXiv 2311.12194](https://arxiv.org/pdf/2311.12194)
- Unphased Wrinkles notes that earlier approaches isolate stretch from bending in designed experiments, while stretching and wrinkling are coupled in real observations. Its frequency loss yields consistent elasticity estimates across different wrinkle patterns, i.e. it is robust to wrinkle placement. — [arXiv 2212.08790](https://arxiv.org/abs/2212.08790)
- Per-scene inverse optimization gives inconsistent parameters for the same material across scenes/interactions, which motivated MatPhys's shared material codebook. — [arXiv 2605.19386](https://arxiv.org/abs/2605.19386)
- Monocular reconstruction suffers depth ambiguity. SAFT needed two dedicated regularizers to keep monocular cloth reconstructions plausible. — [arXiv 2509.08828](https://arxiv.org/pdf/2509.08828)

### Inferences
- Identifiability: under gravity alone, quasi-static drape depends mostly on the bending-to-weight ratio (a bending length scale), so absolute bending and density are individually hard to identify from shape. Dynamic cues (oscillation frequency after body stops, flutter spectra) add information about the stiffness/mass ratio and damping. In generated video, the unknown effective "wind" and body motion act like Bouman's unknown forces, so estimating ratios (bending/density, damping ratio) is better conditioned than absolute values. This is a standard physics argument, not verified by a specific paper this session.
- WAN-generated video likely does not obey consistent physics across clips (dynamics may be "style-plausible" rather than physically exact). A robust aggregator (median or heavy-tailed likelihood over per-clip estimates) plus a strong fabric-type prior should be preferred over jointly fitting all clips with a single L2 loss.
- Suggested minimal parameter vector for HOOD/ContourCraft-style fitting: {log bending stiffness, log area density, log stretch (Lamé) stiffness — nearly fixed to high, damping}. Collision thickness/friction stay at defaults. Optimize ratios where possible.

### Gaps
- No quantitative sensitivity/identifiability study of HOOD/ContourCraft material parameters from monocular video was found.
- No study was found on whether video diffusion models (WAN 2.x or others) reproduce fabric-dependent dynamics faithfully enough for parameter recovery. Physical-accuracy benchmarks of video generators exist (see survey [arXiv 2503.21765](https://arxiv.org/pdf/2503.21765)), but none specific to fabric material fidelity was verified.
