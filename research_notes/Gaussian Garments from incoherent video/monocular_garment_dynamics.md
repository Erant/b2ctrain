# Reconstructing and learning clothing / non-rigid dynamics from monocular, unsynchronized, and incoherent video collections

Scope note (read first): This pass was done on 2026-09-29. Direct page fetches to arxiv.org, huggingface.co, openreview.net, alphaxiv.org, mlanthology.org and *.github.io project pages were blocked by the session's egress policy, so most findings below come from search-engine result summaries of the primary pages (arXiv/CVF/NeurIPS/ICLR/AAAI listings) plus GitHub repository metadata queried through the GitHub API. Numbers quoted are those that appeared in those summaries; they were not cross-checked against full PDFs. Code availability was verified via GitHub repository search (star counts as of 2026-09-29).

Project framing used for relevance: we have (a) a static canonical 3DGS of the clothed subject with per-splat clothing labels, (b) an SMPL-like body proxy, and (c) many Wan 2.2-generated monocular clips from different angles with different motions that are not mutually consistent. Gaussian Garments needs synchronized multi-view video. The pattern of interest is "known canonical model + per-clip tracking + one shared dynamics model".

## Q1. Monocular clothed-avatar methods with garment dynamics: which are history/dynamics-conditioned (not just pose-conditioned), and which handle loose clothing?

### Takeaway
Most Gaussian/NeRF avatar methods (Animatable Gaussians, LayGA, 3DGS-Avatar, GART, and similar) condition deformation on the current pose only. That fails for loose garments, because the same pose can correspond to many cloth states. A 2025–2026 wave of methods adds explicit temporal or autoregressive state: SMAGA (ICLR 2026, monocular), MonoCloth (AAAI 2026, monocular), SeqAvatar (ICCV 2025), DSAR (2026), Latent Dynamics (Meta, 2026), RealityAvatar, and R3-Avatar. The only avatar systems that produce *simulator-grade* dynamics, rather than learned correlations, are PhysAvatar, MPMAvatar and Gaussian Garments, and all three need multi-view video. DressRecon and ReLoo handle very loose clothing from a single monocular video, but they fit per-video deformation and learn no transferable dynamics.

### Cited Findings
**The problem with pose-only conditioning**
- "Loose clothing and other dynamic elements deform in ways pose alone cannot explain: the same pose can correspond to many different states, because their motion depends on history, inertia, and contact." — [Latent Dynamics for Full Body Avatar Animation (arXiv 2605.21478)](https://arxiv.org/pdf/2605.21478)
- Most avatar methods model non-rigid cloth deformation with a pose-conditioned MLP, which gives unrealistic cloth behaviour in novel poses. Existing methods "primarily rely on global pose conditioning or static per-frame representations, leading to oversmoothing and temporal inconsistencies in non-rigid regions". — [RealityAvatar (arXiv 2504.01559)](https://arxiv.org/html/2504.01559v1); [PICA (arXiv 2407.05324)](https://arxiv.org/html/2407.05324v1)
- Animatable Gaussians learns a parametric template for loosely dressed performers and uses a UNet to predict Gaussian properties from posed position maps. LayGA extends it to multiple layers for clothing transfer. Both are pose-driven. — [LayGA (arXiv 2405.07319)](https://arxiv.org/html/2405.07319v1); summary via [Forwardrobe (arXiv 2607.29106)](https://www.alphaxiv.org/abs/2607.29106)
- Current methods struggle with skirts: they show "needle artifacts" and unrealistic motion because of (1) limited Gaussian deformation under the predefined template articulation and (2) a mismatch between body-template assumptions and loose-garment geometry. — [Forwardrobe / RealityAvatar summaries](https://arxiv.org/html/2504.01559v1)

**History- or dynamics-conditioned learned avatars**
- **SMAGA** (Secondary Motion-Aware 3D Clothed Gaussian Avatars from Monocular Videos, ICLR 2026) takes a *monocular* video as input. It uses a "motion-aware autoregressive structural deformation framework" that organizes Gaussians into an approximate graph and recursively predicts structure-preserving updates. The paper describes the result as template-free cloth dynamics for loose garments such as skirts. No public code repo was found by GitHub search. — [OpenReview](https://openreview.net/forum?id=2A3Q2EtGTF); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/25fbef973f6ec82568de945ae058134a-Abstract-Conference.html)
- **MonoCloth** (Jin & He, NTU, AAAI 2026) works from *monocular* video. It decomposes the avatar into body, face, hands and clothing, and adds "a dedicated cloth simulation module that captures garment deformation using temporal motion cues and geometric constraints". Its part-based design supports clothing transfer. — [arXiv 2508.04505](https://arxiv.org/html/2508.04505v2); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37468)
- **Latent Dynamics for Full Body Avatar Animation** (SFU + Meta Codec Avatars Lab, arXiv May 2026) adds two things to a pose-conditioned 3DGS avatar:
  - a transformer decoder and a "dynamics residual latent";
  - a latent dynamics model that autoregressively rolls out the residual latent from pose history and the previous latent state. It keeps a second-order state (latent position and velocity) and splits each update into driving, restoring and dissipative forces, at "negligible added cost".
  - (Capture setup is not given in the summary, but Codec Avatars work is typically multi-view.) — [arXiv 2605.21478](https://arxiv.org/pdf/2605.21478); [review](https://www.themoonlight.io/fr/review/latent-dynamics-for-full-body-avatar-animation)
- **DSAR** (Nanjing Univ., arXiv 2608.10500, 2026) argues that "current states emerge from previous states through temporal evolution rather than instantaneous skeletal configurations". Its two streams are:
  - a geometric stream that propagates the previous frame's surface displacement;
  - a state stream that fuses current features with historical states taken from a memory bank.
  - It reports better generalization to motion patterns outside the training distribution. — [arXiv 2608.10500](https://arxiv.org/pdf/2608.10500)
- **SeqAvatar** (Sequential Gaussian Avatars with Hierarchical Motion Context, ICCV 2025) conditions on a skeleton-motion term (pose differences between adjacent frames) plus per-point velocity, using spatio-temporal multi-scale sampling. — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xu_Sequential_Gaussian_Avatars_with_Hierarchical_Motion_Context_ICCV_2025_paper.html)
- **RealityAvatar** (arXiv 2504.01559) adds a "motion trend module" and a latent bone encoder to model pose-dependent deformation and temporal variation of loose clothing. — [arXiv](https://arxiv.org/html/2504.01559v1)
- **R3-Avatar** (arXiv 2503.12751) uses a "record-retrieve-reconstruct" temporal codebook. Novel poses retrieve the timestamps of the most similar training poses. This is retrieval, not dynamics. — [arXiv](https://arxiv.org/pdf/2503.12751)
- **Dynamic Texture Modeling of 3D Clothed Gaussian Avatars from a Single Video** appeared at ICLR 2026. Only the title was retrieved. — [ICLR 2026](https://iclr.cc/virtual/2026/paper/10011771)
- **Real-time Deep Dynamic Characters (DDC)** (Habermann et al., SIGGRAPH 2021) is the older mesh-based precedent. It learns motion-dependent deformation of body and clothing, including skirt swing that depends on skeletal motion, using a graph convolutional network trained weakly supervised from *multi-view* imagery, without physics simulation. — [project page](https://vcai.mpi-inf.mpg.de/projects/2021-ddc/); [DeepAI](https://deepai.org/publication/real-time-deep-dynamic-characters)

**Very loose clothing from a single monocular video (per-video fit, no transferable dynamics)**
- **DressRecon** (Tan, Xiang, Tulsiani, Ramanan, Yang, CMU):
  - Combines generic articulated-body priors with a video-specific "bag-of-bones" deformation fit per video by test-time optimization. Body and clothing deformation are separate motion layers.
  - Uses image priors during optimization: body pose, surface normals and optical flow.
  - Outputs time-consistent meshes, or 3D Gaussians for rendering.
  - Venue: the repo lists 3DV 2025 Oral. Code: [jefftan969/dressrecon](https://github.com/jefftan969/dressrecon), about 144 stars. — [arXiv 2409.20563](https://arxiv.org/html/2409.20563v2)
- **ReLoo** (Guo et al., ETH/Microsoft, ECCV 2024) uses a layered neural representation: an inner body plus outer clothing. A "non-hierarchical virtual bone deformation module" lets the clothing layer move freely, and everything is optimized jointly with multi-layer differentiable volume rendering on in-the-wild monocular video. Code: [eth-ait/ReLoo](https://github.com/eth-ait/ReLoo). — [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1355_ECCV_2024_paper.php)
- **DLCA-Recon** (AAAI 2024) does dynamic loose-clothing avatar reconstruction from monocular video. Only the title was retrieved. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/28189)

**Physics-based avatars (simulator-grade dynamics; all need multi-view)**
- **PhysAvatar** (Zheng et al., Stanford/Google, ECCV 2024) combines inverse rendering with inverse physics. It uses "mesh-aligned 4D Gaussian" spatio-temporal mesh tracking from *multi-view video*, then estimates garment physical parameters by gradient-based optimization through a physics simulator. Code: [y-zheng18/PhysAvatar](https://github.com/y-zheng18/PhysAvatar). — [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05446.pdf)
- **MPMAvatar** (KAIST, NeurIPS 2025) learns 3DGS avatars from *multi-view videos* with a Material Point Method garment simulator (anisotropic constitutive model plus new body-collision handling). It reports beating prior physics-based avatars on dynamics accuracy, rendering, robustness and efficiency, and zero-shot generalization to unseen interactions. Code: [KAISTChangmin/MPMAvatar](https://github.com/KAISTChangmin/MPMAvatar). — [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/ccdfe117c6729267c1595cdf0a587da8-Abstract-Conference.html)
- **Gaussian Garments** (Rong, Grigorev, Wang, Black, Thomaszewski, Tsalicoglou, Hilliges; 3DV 2025) represents each garment as a 3D mesh plus a Gaussian texture. It registers garment geometry to *multi-view video*, and a pre-trained GNN (HOOD/ContourCraft family) is fine-tuned to replicate each garment's real behaviour. Garments can be combined into outfits, resized and retargeted. Code: [eth-ait/Gaussian-Garments](https://github.com/eth-ait/Gaussian-Garments), about 172 stars. A community adapter for ActorsHQ exists: [hlimach/ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments). — [MPI project](https://is.mpg.de/ps/en/projects/gaussian-garments); [arXiv 2409.08189](https://arxiv.org/html/2409.08189v1)
- **PGC: Physics-Based Gaussian Cloth from a Single Pose** (CVPR 2025) recovers simulation-ready garments from a multi-view capture of *one static pose*. It uses hybrid mesh-embedded Gaussians: Gaussians for near-field shading and detail, the mesh for albedo and reflectance. Novel poses come from physics simulation of the mesh. No code found by GitHub search. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_PGC_Physics-Based_Gaussian_Cloth_from_a_Single_Pose_CVPR_2025_paper.html)
- **Vid2Avatar-Pro** (Meta/ETH/Toronto, CVPR 2025) builds monocular avatars on a universal prior model learned from a large multi-view capture corpus, using shared canonical front/back Gaussian maps. It is pose-driven and not dynamics-focused. — [CVF](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Vid2Avatar-Pro_Authentic_Avatar_from_Videos_in_the_Wild_via_Universal_CVPR_2025_paper.html)

### Inferences
- **PGC is the closest published analogue to our starting point:** a static multi-view capture turned into simulation-ready cloth. It shows that a static capture plus a good simulator prior already gives plausible novel-pose simulation *without any video*. Video then only refines material parameters, which is a low-dimensional target.
- **Learned temporal-state avatars are still per-subject, per-sequence fits.** SMAGA, MonoCloth, DSAR and SeqAvatar learn correlations from a single, contiguous, coherent video. None is designed to pool *many inconsistent clips*.
- **Latent Dynamics suits pooling best among the learned options.** Its second-order latent dynamics (driving, restoring and dissipative forces) is a compact, physically structured state. If trained with per-clip nuisance codes, it could be fit across clips.
- **Our body proxy maps onto DressRecon/ReLoo's layer split.** They separate body and garment deformation layers, and our SMPL-like proxy plus per-splat clothing labels already provide that separation. Their "free-moving virtual bones" for the garment layer are a good per-clip tracking parameterization for loose cloth.

### Gaps
- **Named methods not verified in this pass:** SCARF, DELTA, "DGarment", MonoClothCap, DeepCap, Garment Avatars, CaPhy, GaussianAvatar, 3DGS-Avatar, MonoGaussianAvatar, GART, ExAvatar, MoSAR, D3GA, HumanSplat. Searches were budgeted elsewhere. From prior knowledge only, and unverified:
  - SCARF/DELTA (MPI) use hybrid mesh + NeRF clothing from monocular video, pose-conditioned.
  - DeepCap and MonoClothCap are monocular template-based performance capture; DeepCap uses multi-view weak supervision at training time.
  - Garment Avatars (Meta, 3DV 2022) and D3GA are multi-view and pose/driving-signal conditioned.
  - GART, 3DGS-Avatar, GaussianAvatar, MonoGaussianAvatar and ExAvatar are monocular and pose-conditioned, with limited loose-clothing support.
  - CaPhy learns physics-informed clothing from scans.
  - HumanSplat is a single-image feed-forward generator.
- "DGarment" could not be resolved to a specific paper. The closest match found was **D-Garment** (TMLR 2026, a physics-conditioned latent diffusion model for dynamic garment deformation trained on simulation data; see Q4).
- SMAGA's exact inputs, datasets and quantitative numbers, and whether code exists, could not be confirmed (OpenReview fetch blocked).

## Q2. Tracking a known template through monocular video: how well does analysis-by-synthesis fitting of a pre-existing mesh or splat work for cloth?

### Takeaway
The most directly relevant line is **Shape-from-Template (SfT) with a physics simulator in the loop**. φ-SfT is the original. Its CVPR 2024 successor uses a neural surrogate and runs 400–500× faster. **SAFT** (ICCV 2025) recovers cloth geometry and appearance from a single monocular video via differentiable simulation and rendering, and reports 2.64× lower 3D error thanks to new depth-ambiguity regularizers. For clothed humans, the template-fitting pipelines are REC-MV (CVPR 2023, monocular dynamic garment surfaces) and GarmentRecovery (CVPR 2024, fitting with learned shape and deformation priors). The generic foundation-model toolkit is now strong enough to give usable per-clip cues:
- 3D point tracking: TAPIP3D, SpatialTrackerV2;
- 2D tracking: CoTracker3;
- feed-forward 4D geometry: MonST3R, St4RTrack, MegaSaM.

Monocular depth ambiguity remains the key failure mode. Physics priors or body priors are what make the fits plausible.

### Cited Findings
- **Physics-guided SfT** (Stotko, Wandel, Klein, Univ. Bonn, CVPR 2024):
  - reconstructs cloth template geometry from monocular RGB video using a pre-trained neural surrogate simulator plus differentiable rendering;
  - pixel-wise comparison against the video lets gradient optimization recover shape *and* physical parameters (stretching, shearing, bending stiffness);
  - runs 400–500× faster than φ-SfT. — [CVF CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Stotko_Physics-guided_Shape-from-Template_Monocular_Video_Perception_through_Neural_Surrogate_Models_CVPR_2024_paper.html)
- **SAFT** (Stotko & Klein, ICCV 2025, pp. 27660–27670) jointly estimates 3D fabric geometry and PBR appearance from a *single monocular RGB video*, using physical cloth simulation and differentiable rendering. Two new regularization terms address monocular depth ambiguity and cut 3D reconstruction error by a factor of 2.64 versus recent methods. No code repo found via GitHub search. — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Stotko_SAFT_Shape_and_Appearance_of_Fabrics_from_Template_via_Differentiable_ICCV_2025_paper.html)
- **REC-MV** (CVPR 2023) reconstructs dynamic 3D garment surfaces with open boundaries from monocular video. It jointly optimizes explicit feature curves and an implicit garment SDF, then extracts open garment meshes by template registration in canonical space. The paper notes that earlier neural-rendering avatars could not separate garment from body. — [CVF CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Qiu_REC-MV_REconstructing_3D_Dynamic_Cloth_From_Monocular_Videos_CVPR_2023_paper.html)
- **Garment Recovery with Shape and Deformation Priors** (Li, Dumery, Guillard, Fua, EPFL, CVPR 2024) fits garment models to real images using shape and deformation priors learned from synthetic data, including large deformations. The output is usable directly for animation and simulation. — [CVF CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Garment_Recovery_with_Shape_and_Deformation_Priors_CVPR_2024_paper.html)
- The same EPFL line also covers reconstructing manipulated garments with a guided deformation prior. — [arXiv 2405.10934](https://arxiv.org/pdf/2405.10934)
- "High-Quality Animatable Dynamic Garment Reconstruction from Monocular Videos" frames garment reconstruction as pose-driven deformation, with a learnable garment deformation network guided by body priors. — [arXiv 2311.01214](https://ar5iv.org/html/2311.01214)
- **MulayCap** (2020) does multi-layer human performance capture from a *monocular* camera. It is an older baseline for layered cloth tracking. — [arXiv 2004.05815](https://arxiv.org/pdf/2004.05815)
- **3D point tracking:**
  - **TAPIP3D** (NeurIPS 2025) does long-term 3D point tracking in monocular RGB or RGB-D video. It lifts features into camera-stabilized spatio-temporal feature clouds using depth and camera motion, with 3D neighborhood-to-neighborhood attention. It reports beating CoTracker3 on datasets with GT depth. — [NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117634)
  - **SpatialTrackerV2** (ICCV 2025) is a feed-forward monocular 3D point tracker. It decomposes world-space motion into scene geometry, camera ego-motion and per-pixel object motion. It reports being 30% better than prior 3D trackers, matching dynamic-reconstruction methods at 50× the speed. — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_SpatialTrackerV2_Advancing_3D_Point_Tracking_with_Explicit_Camera_Motion_ICCV_2025_paper.html)
- **CloDS** (ICLR 2026) found that grounding cloth video to geometry is hard. To cope with large nonlinear deformations and severe self-occlusion, it introduced "dual-position opacity modulation" for mesh-based Gaussian splatting, enabling bidirectional 2D↔3D mapping. — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009027)
- PhysAvatar and Gaussian Garments both track with mesh-embedded Gaussians (a "Gaussian texture" or "mesh-aligned 4D Gaussians") and photometric fitting, but from multi-view input. — [PhysAvatar](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05446.pdf); [Gaussian Garments](https://is.mpg.de/ps/en/projects/gaussian-garments)

### Inferences
- **Our canonical splat already has the structure trackers need.** It carries per-splat clothing labels and a body proxy. Converting the garment splats into a mesh-embedded Gaussian texture, as Gaussian Garments/PGC do, gives a template that can be fit per clip with photometric, silhouette (SAM2 masks) and normal (Sapiens) losses. Losses from 2D tracks (CoTracker3) or lifted 3D tracks (SpatialTrackerV2/TAPIP3D) add dense correspondence. This is exactly the SfT setup.
- **Keep a simulator in the fitting loop.** Put a simulator (HOOD/ContourCraft, or a neural surrogate as in φ-SfT/SAFT) inside the per-clip fit, instead of fitting a free deformation and then learning physics from it. That turns the monocular depth ambiguity into a well-posed problem over a few parameters (material, initial state) and removes the need for accurate per-frame 4D garment surfaces.
- **Generated video will be harder than the benchmarks.** Wan 2.2 clips will not be photometrically consistent with the canonical splat (identity/texture drift) and may contain physically implausible cloth motion. Silhouette-, normal- and track-based losses should get more weight than RGB photometric loss.

### Gaps
- **No quantitative benchmark of monocular template tracking of *loose garments on humans* was found** (e.g., per-vertex error on 4D-Dress from a single view). SAFT's 2.64× figure is for fabric-only scenes, not clothed humans.
- **Not individually verified in this pass:** RAFT/SEA-RAFT, BootsTAP/TAPIR, SpatialTracker v1, Depth Anything V2, Video Depth Anything, DepthCrafter, Sapiens normals, DINOv2 features, DensePose-CSE. They are well known, but no source was fetched.

## Q3. Casual monocular dynamic reconstruction (Shape of Motion, MoSca, MonST3R, MegaSaM, St4RTrack, and others): what priors make them work, and could they supply per-clip 4D garment surfaces good enough to supervise a simulator?

### Takeaway
These systems work monocularly by combining three things:
1. foundation-model priors: monocular or video depth, long-range 2D tracks, camera pose;
2. a low-dimensional motion parameterization: SE(3) motion bases in Shape of Motion, a sparse motion scaffold in MoSca;
3. feed-forward pointmap networks in MonST3R and St4RTrack.

They give plausible visible-surface 4D reconstructions and dense trajectories. They have no notion of garment topology, occluded sides or physical plausibility. They are better used as *per-clip cue generators* (3D tracks, depth) for template-anchored fitting than as direct simulator supervision.

### Cited Findings
- **Shape of Motion** (2024) reconstructs generic dynamic scenes with full-sequence 3D motion from casual monocular video. Scene motion is represented by a compact set of SE(3) motion bases, and each point moves as a linear combination of them. It uses monocular depth maps and long-range 2D tracks as data priors. Code: [vye16/shape-of-motion](https://github.com/vye16/shape-of-motion), about 1.3k stars. — [arXiv 2407.13764](https://arxiv.org/html/2407.13764v2)
- **MoSca** (CVPR 2025) lifts video into a 4D "Motion Scaffold" that compactly and smoothly encodes deformation, using foundation-model priors. Gaussians are anchored on the scaffold, which disentangles geometry and appearance from deformation. — [CVPR 2025 supplemental](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Lei_MoSca_Dynamic_Gaussian_CVPR_2025_supplemental.pdf); [MoSca v2 PDF](https://www.cis.upenn.edu/~leijh/projects/mosca/pub/mosca_v2.pdf)
- **MonST3R** adapts DUSt3R to dynamic scenes and predicts per-frame 3D pointmaps in a common frame, feed-forward. — [MonST3R (2410.03825)](https://fugumt.com/fugumt/paper_check/2410.03825v1_enmode)
- **MegaSaM** jointly optimizes camera parameters and per-frame dense depth using monocular depth priors. It gives consistent depth for dynamic objects even with little camera parallax. — [arXiv 2412.04463](https://arxiv.org/html/2412.04463v1)
- **St4RTrack** (ICCV 2025) is feed-forward. From a pair of images it outputs two world-frame pointmaps, simultaneously tracking points from the first frame and reconstructing later frames. — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Feng_St4RTrack_Simultaneous_4D_Reconstruction_and_Tracking_in_the_World_ICCV_2025_paper.pdf)
- **SpatialTrackerV2** reports matching leading dynamic 3D reconstruction methods in accuracy at 50× speed. — [CVF ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Xiao_SpatialTrackerV2_Advancing_3D_Point_Tracking_with_Explicit_Camera_Motion_ICCV_2025_paper.html)
- **DressRecon** is the human-specialized counterpart. It uses per-video bag-of-bones deformation with image priors (pose, normals, optical flow), and its output meshes are time-consistent. — [arXiv 2409.20563](https://arxiv.org/html/2409.20563v2)

### Inferences
- **The motion parameterizations mirror what cloth trackers need.** Shape of Motion's SE(3) bases and MoSca's scaffold are low-rank, smooth deformation fields, conceptually the same as DressRecon/ReLoo's virtual bones. A garment-specific variant would anchor the scaffold nodes to our canonical garment splats, with body-proxy skinning as initialization.
- **These outputs can supervise a simulator's *visible-side trajectories* only weakly.** They cover only the visible side, without garment/body separation, and with scale/depth ambiguity. Using them as distribution-level or sparse-track targets inside a template-plus-simulator fit is more promising than treating them as ground-truth meshes.

### Gaps
- **Not verified in this pass:** Dynamic Gaussian Marbles, CUT3R, 4D-LRM, Stereo4D (searches did not return them). From prior knowledge only, unverified:
  - CUT3R is a recurrent, continuous-update 3D reconstruction model (CVPR 2025).
  - Stereo4D mines internet stereo VR180 video for 4D training data.
- No paper was found that evaluates Shape of Motion or MoSca specifically on loose-garment accuracy.

## Q4. Aggregating across many inconsistent observations: shared canonical plus per-clip deformation, per-clip latents, unsynchronized multi-view, learning shared dynamics or physics from video (including generated video)

### Takeaway
Four families are directly usable:
1. **Unsynchronized multi-video 4DGS.** SyncTrack4D and the AAAI 2026 "Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos" estimate per-video time offsets. But they assume the *same* motion event filmed by several cameras, which our clips (different motions) break.
2. **Physics/material estimation from generated video.** PhysDreamer and Physics3D optimize material parameters of a 3DGS object through differentiable MPM so that rendered motion matches video-diffusion outputs. This is exactly "distill physics from a video generator into an anchored canonical splat".
3. **Self-supervised neural cloth simulators.** SNUG and HOOD are trained with physics losses only. ContourCraft, D-Garment and Gaussian Garments fine-tune or condition them on real data.
4. **Video-to-geometry grounding then dynamics learning.** CloDS (ICLR 2026) and PhysAvatar/MPMAvatar.

For pooling clips, the strongest pattern is: shared canonical garment (our splat) + shared low-dimensional material/dynamics parameters + per-clip nuisance variables (camera, body motion, initial state, appearance/lighting code, time warp). All of it is optimized jointly across clips, with each clip fit by simulation plus differentiable rendering.

### Cited Findings
**Unsynchronized multi-view**
- **SyncTrack4D** (arXiv 2512.04315) works as follows:
  - computes dense per-video 4D feature tracks and cross-video track correspondences via Fused Gromov-Wasserstein optimal transport;
  - does frame-level temporal alignment by maximizing overlapping motion of matched tracks;
  - refines to sub-frame sync with a multi-video 4DGS on a motion-spline scaffold.
  - Reported: average sync error below 0.26 frames and 26.3 PSNR on Panoptic Studio.
  - Claimed as the first general 4DGS for unsynchronized video sets without predefined object or prior models. No public repo found via GitHub search. — [arXiv 2512.04315](https://arxiv.org/html/2512.04315v1); [radiancefields.com](https://radiancefields.com/papers/synctrack4d-cross-video-motion-alignment-and-video-synchronization-for-multi-video-4d-gaussian-splatting)
- **Dynamic Gaussian Scene Reconstruction from Unsynchronized Videos** (AAAI 2026, arXiv 2511.11175) uses a coarse-to-fine temporal alignment module: a frame-level offset per camera, then sub-frame refinement, inside 4DGS. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38129); [arXiv](https://arxiv.org/html/2511.11175v1)

**Physics from (generated) video, anchored to a 3DGS**
- **PhysDreamer** (ECCV 2024):
  - represents the object as 3D Gaussians and the material as a neural field, and simulates with differentiable MPM;
  - renders the object from a viewpoint and uses an *image-to-video generation model* to produce a reference video;
  - optimizes a spatially varying material field and initial velocity field by differentiable MPM plus rendering, to match that reference video. — [Springer ECCV 2024](https://link.springer.com/chapter/10.1007/978-3-031-72627-9_22); [NSF PAR PDF](https://par.nsf.gov/servlets/purl/10562143)
- **Physics3D** extends the parameters to elasticity plus viscosity with a viscoelastic MPM. It optimizes them by Score Distillation Sampling from a video diffusion model, which is distribution-level supervision rather than matching one specific video. — [project page](https://liuff19.github.io/Physics3D/); [arXiv 2406.04338](https://arxiv.org/html/2406.04338v3)
- **PhysAvatar** estimates fabric physical parameters by gradient-based optimization through a simulator, after 4D mesh tracking from multi-view video. — [ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05446.pdf)
- **MPMAvatar** uses an MPM garment simulator with an anisotropic constitutive model and collision handling, fit from multi-view video. It claims zero-shot generalization to unseen interactions, "not achievable with previous learning-based methods". — [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/ccdfe117c6729267c1595cdf0a587da8-Abstract-Conference.html)
- **Physics-guided SfT** and **SAFT** recover cloth stiffness parameters and geometry from *monocular* video with differentiable simulation (see Q2). — [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Stotko_Physics-guided_Shape-from-Template_Monocular_Video_Perception_through_Neural_Surrogate_Models_CVPR_2024_paper.html); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Stotko_SAFT_Shape_and_Appearance_of_Fabrics_from_Template_via_Differentiable_ICCV_2025_paper.html)
- **Image2Garment** (arXiv 2601.09658):
  - predicts fabric attributes (fabric family, structure, areal density, thickness) from a single image;
  - maps them to simulator parameters via a learned material-to-physics model;
  - notes that most prior physics-from-visual methods "require strong visual supervision with multi-view or video setups and most are not compatible with established cloth simulators". — [arXiv](https://arxiv.org/html/2601.09658v3)
- **DiffAvatar** (arXiv 2311.12194) optimizes simulation-ready garments with differentiable simulation. — [arXiv](https://arxiv.org/pdf/2311.12194)
- Classic precedent: Bhat et al. (SCA 2003) estimated cloth simulation parameters from real fabric video, using a fold-matching perceptual metric minimized by simulated annealing. — [ACM DL](https://dl.acm.org/doi/10.5555/846276.846282)
- Learning-based cloth material recovery from video: [Yang et al., ICCV 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Yang_Learning-Based_Cloth_Material_ICCV_2017_paper.pdf)

**Learning a cloth dynamics model from visual observations**
- **CloDS** (Cloth Dynamics Splatting, ICLR 2026) learns cloth dynamics unsupervised from *multi-view* visual observations with unknown physical properties. Its three stages are video-to-geometry grounding with mesh-based Gaussian splatting (dual-position opacity modulation), then training a dynamics model on the grounded meshes. It reports generalization to unseen configurations. Code: [whynot-zyl/CloDS](https://github.com/whynot-zyl/CloDS). — [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009027); [arXiv 2602.01844](https://arxiv.org/html/2602.01844v3)
- **SNUG** (CVPR 2022) recasts physics-based deformation as an optimization problem, so physics loss terms can train garment networks without ground-truth simulation data. It reports a two-orders-of-magnitude training speedup over supervised methods. — [CVF CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Santesteban_SNUG_Self-Supervised_Neural_Dynamic_Garments_CVPR_2022_paper.html)
- **HOOD** (CVPR 2023) is a hierarchical-graph GNN trained fully self-supervised with a physics loss. It generalizes to unseen garment types and shapes and allows run-time changes of material properties and garment size. — [arXiv 2212.07242](https://arxiv.org/html/2212.07242v3)
- **ContourCraft** (SIGGRAPH 2024) extends HOOD with learned intersection resolution for multi-garment simulation. Code: [Dolorousrtur/ContourCraft](https://github.com/Dolorousrtur/ContourCraft). Gaussian Garments fine-tunes this GNN family per garment. — [GitHub](https://github.com/Dolorousrtur/ContourCraft); [Gaussian Garments](https://is.mpg.de/ps/en/projects/gaussian-garments)
- **D-Garment** (TMLR 2026) is a latent diffusion model over a 2D garment parameter space. It generates dynamic garment deformation conditioned on body shape, motion *and physical material properties*, and is trained on physics-simulator data. — [arXiv 2504.03468](https://arxiv.org/pdf/2504.03468); [TMLR listing](https://mlanthology.org/tmlr/2026/dumoulin2026tmlr-dgarment/)
- **Neural Garment Dynamics via Manifold-Aware Transformers** (2024) is another learned garment-dynamics model. Only the title was retrieved. — [arXiv 2407.06101](https://arxiv.org/pdf/2407.06101)

**Training avatars on video-diffusion outputs (inconsistency handling)**
- **PERSONA** (arXiv 2508.09973) uses a diffusion animator (MimicMotion) to generate pose-rich videos from a single image, then optimizes a 3D avatar with pose-driven deformation. To counter identity drift in generated frames, it alternates between the real input image and generated frames and oversamples the input image. Related works SVAD and "Dream, Lift, Animate" follow the same generate-then-fit pattern. — [PERSONA project](https://mks0601.github.io/PERSONA/); [SVAD](https://arxiv.org/pdf/2505.05475); [DLA](https://arxiv.org/html/2507.15979)
- **MV-Performer** (arXiv 2510.07190) uses a video diffusion model to synthesize *synchronized multi-view* performer videos. Only the title and abstract snippet were retrieved. — [arXiv](https://arxiv.org/pdf/2510.07190)

### Inferences
**Recommended aggregation design, synthesized from the above:**
- **(a) Canonical, shared across clips.** Use our garment splats, converted to mesh plus Gaussian texture (Gaussian Garments/PGC style).
- **(b) Shared across clips: material parameters.** Treat HOOD/ContourCraft material parameters, or a small per-garment material latent, as the *only* dynamics parameters shared across clips. Gaussian Garments already fine-tunes this GNN, and HOOD supports run-time material changes.
- **(c) Per clip:**
  - camera trajectory;
  - SMPL body motion, from a monocular human mesh recovery method (e.g., the SAM-3D-Body proxy);
  - garment initial state/velocity;
  - an appearance/lighting code (NeRF-W style);
  - optionally a time-warp to absorb generator speed artifacts.
- **(d) Per-clip losses:**
  - silhouette/garment mask;
  - normal maps;
  - 2D/3D point tracks projected from the simulated mesh;
  - down-weighted RGB.
- **(e) Joint optimization.** Optimize (b) jointly over all clips, with (c) per clip. This is the PhysDreamer/PhysAvatar recipe, extended from one video to N clips with nuisance codes.
- **(f) Fallback where exact matching is unreliable.** Use distribution-level losses: SDS from Wan 2.2 à la Physics3D, or track-statistics matching.

**Why unsynchronized-sync methods do not transfer:** they solve a different problem, one event seen by N cameras. Our clips show different motions, so no time offset aligns them. The only shared quantities are the canonical garment and its material/dynamics.

**Alternative worth benchmarking:** use MV-Performer-style multi-view video diffusion, or multi-view-conditioned Wan generation, to create *synchronized* multi-view clips. That would make Gaussian Garments usable almost as-is.

**Latent Dynamics (Meta 2026) is the best learned-model alternative to a simulator.** It uses a second-order latent state with driving, restoring and dissipative decomposition. Trained across clips with per-clip codes, it could generalize with few parameters. It is still likely to need more data than a physics-parameter fit.

**Generated video may not satisfy real cloth physics.** Fitting a physics simulator (low-dimensional, physically constrained) is a safeguard. It cannot overfit to implausible generated motion the way a free-form learned deformation can.

### Gaps
- **No paper was found that pools *many independent monocular clips with different motions* of the same garment to fit one shared cloth simulator/material.** This appears to be an open niche. PhysDreamer and Physics3D use generated video but for objects, not garments on moving bodies.
- **Not verified:** Sync-NeRF (AAAI 2024), audio-based sync, and NeRF-W (appearance/transient embeddings, CVPR 2021). They are well known, but no source was fetched in this pass.
- **The exact Gaussian Garments fine-tuning losses and parameters were not confirmed** (arXiv fetch blocked): which HOOD material parameters, and whether the loss is on tracked vertices or photometric.

## Q5. Datasets with ground truth for validating a monocular pipeline: loose-garment dynamics with synchronized views

### Takeaway
**4D-Dress** (CVPR 2024) is the best fit for validating garment geometry. It has real 4D textured scans with per-vertex garment labels, garment meshes and SMPL(-X) fits, including dresses and outerwear, and it is what Gaussian Garments and related ETH work build on. **ActorsHQ** (160 cameras, 12 MP, 16 sequences) is the standard for high-res multi-view rendering and physics-avatar work, and has a Gaussian Garments adapter. **DNA-Rendering** has dedicated loose-garment sequences. **MVHumanNet/MVHumanNet++** give scale (thousands of subjects, 16 views) but mostly everyday clothing.

A practical validation protocol: pick one view from a synchronized dataset as the "monocular clip", run the pipeline, and score against the held-out views and 4D scans.

### Cited Findings
- **4D-Dress** (CVPR 2024, ETH):
  - 64 outfits, 520 motion sequences, 78k textured scans;
  - vertex-level semantic labels, garment meshes and fitted SMPL(-X);
  - garments: 4 dresses, 28 lower, 30 upper and 32 outer;
  - multi-view renderings from 24 views on a sphere;
  - benchmarks for clothing simulation and reconstruction. — [CVF CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_4D-DRESS_A_4D_Dataset_of_Real-World_Human_Clothing_With_Semantic_CVPR_2024_paper.pdf); [arXiv 2404.18630](https://arxiv.org/html/2404.18630v1)
- **ActorsHQ:** 12 MP footage from 160 cameras, 16 sequences, 8 actors. — [summary via MV-Fashion (CVPR 2026)](https://arxiv.org/html/2603.08147)
- ActorsHQ has been adapted for Gaussian Garments garment-mesh reconstruction (gs2mesh). — [hlimach/ActorsHQ-for-Gaussian-Garments](https://github.com/hlimach/ActorsHQ-for-Gaussian-Garments)
- **DNA-Rendering:** 500 subjects / 1,500 outfits from a multi-camera rig. Follow-up work used 6 loose-garment sequences with 24 training views and 6 test views. — [summary via search results incl. Sequential Gaussian Avatars / R3-Avatar](https://arxiv.org/pdf/2411.16768)
- **MVHumanNet:** multi-view (16-view training videos) captures of over 9,000 identities in everyday clothing.
- **MVHumanNet++:** 4,500 subjects, 9,000 outfits, 500 action types, 60,000 motion sequences, 645.1M frames, up to 12 MP. — [arXiv 2312.02963](https://arxiv.org/pdf/2312.02963); [MVHumanNet++ listing](https://lacuna.tiptreesystems.com/work/mvhumannet-a-large-scale-dataset-of-multi-view-daily-dressing-human-captures/wrk_eed3997bfb37c0e23f867a544dac5d03)
- **MV-Fashion** (CVPR 2026) is a new multi-view paired dataset aimed at try-on and size estimation. — [CVF CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Laczko_MV-Fashion_Towards_Enabling_Virtual_Try-On_and_Size_Estimation_with_Multi-View_CVPR_2026_paper.pdf)
- **CloDS** evaluates learned cloth dynamics from multi-view video, and its code/data are on GitHub. — [whynot-zyl/CloDS](https://github.com/whynot-zyl/CloDS)

### Inferences
- **4D-Dress for geometry, ActorsHQ/DNA-Rendering for appearance and dynamics.**
  - 4D-Dress (dress and outer categories) is the only one of these with *garment-level 4D geometric GT*. It is best for scoring per-clip tracking error and simulated-vs-real garment trajectories.
  - ActorsHQ and DNA-Rendering loose-garment sequences are best for novel-view/novel-motion rendering metrics against PhysAvatar/MPMAvatar/Gaussian Garments baselines.
- **To mimic our setting**, simulate "incoherent clips" from a synchronized dataset: take different sequences of the same outfit, one random camera each. Compare a clip-pooled fit against the Gaussian Garments multi-view fit on held-out motions.

### Gaps
- **Frame rates** of ActorsHQ, DNA-Rendering and MVHumanNet were not found in the summaries.
- **Not verified in this pass:** ZJU-MoCap, CLOTH3D, CLOTH4D, DynaCap and THuman details. From prior knowledge only, unverified:
  - ZJU-MoCap is multi-view with mostly tight clothing.
  - CLOTH3D and CLOTH4D are synthetic simulated-garment datasets.
  - DynaCap (MPI) is multi-view with some loose clothing.
  - THuman consists of static scans.
- **Which datasets PhysAvatar and MPMAvatar evaluate on** (believed to be ActorsHQ) was not confirmed in this pass.
