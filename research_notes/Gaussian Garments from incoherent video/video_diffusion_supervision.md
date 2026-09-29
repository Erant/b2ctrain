# Video Diffusion Models (Wan 2.2 and variants) as Supervision for 4D Clothed-Human / Fabric Dynamics

Research notes, current as of 2026-09-29. Scope: (1) what Wan 2.2 and its descendants can actually do for our "render splat -> animate with video model -> learn fabric motion" idea, (2) methods that lift generated or monocular video into 4D or physics, and how they cope with geometric and temporal inconsistency.

Method notes: arxiv.org, huggingface.co and humanaigc.github.io were blocked by the network proxy in this session. Paper claims below therefore come from search-engine snippets of the arXiv/CVF pages, GitHub READMEs fetched directly, and CVF/ICCV/ICLR listing pages. Claims that rest only on secondary or aggregator sites are marked **[secondary]**. Claims I could not verify at all are in the Gaps sections.

---

## Q1. Wan 2.2 and its variants: capabilities relevant to pose-driven, camera-controlled clothed-human animation; newer Wan releases

### Takeaway
Wan 2.2 (Apache 2.0, July 2025) is still the newest *open-weight* Wan base model. Wan 2.5, 2.6, 2.7 and 3.0 are closed or API-only. The official Wan GitHub org has no Wan 3.0 repo, which contradicts aggregator claims that Wan 3.0 was open-sourced. For our use the relevant open models are:
- **Wan2.2-Animate-14B** (Sept 2025): skeleton plus implicit-face driven character animation or replacement.
- **Wan-Animate-2** (Aug 2026, Apache 2.0): consumes a raw driving video with no skeleton extractor, and adds *text-driven viewpoint control* that decouples output camera from the driving video. This is the closest open tool to "same body motion, different viewpoint".
- **Wan2.2-Fun / VACE-Fun control** (pose, depth, Canny, trajectory) and **Fun-Camera** control.
- **Uni3C** (built on Wan2.1): the one open method that jointly conditions on an **SMPL-X character and a camera trajectory** in a shared 3D world. That is exactly the "specified SMPL pose sequence AND specified camera" we want.

Even so, no Wan-family model guarantees that clips rendered from different viewpoints share the same *cloth* motion. Body motion can be shared through the pose or SMPL condition. Secondary motion (fabric swing, folds) is sampled independently in every generation.

### Cited Findings
**Wan 2.2 base family**
- The Wan2.2 repo lists T2V-A14B and I2V-A14B (480P and 720P, ~80 GB VRAM on a single GPU), TI2V-5B (720P at 1280x704, 24 fps, 5 s, runs on a 24 GB RTX 4090), S2V-14B (audio-driven, length follows the audio) and Animate-14B. The license is Apache 2.0. S2V was released 2025-08-26, Animate-14B on 2025-09-19, and Animate was integrated into Diffusers on 2025-11-13. — [Wan-Video/Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- The README summary I fetched lists Animate-14B output at 30 fps; I could not cross-check this. — [Wan-Video/Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- A14B is a Mixture-of-Experts model: a high-noise expert for early (layout) denoising steps and a low-noise expert for later (detail) steps, routed by timestep. It has 27B total parameters and 14B active. **[secondary]** — [ComfyUI Wiki: Wan2.2](https://comfyui-wiki.com/en/models/wan/wan-2-2); [SiliconFlow model page](https://www.siliconflow.com/models/wan-ai-wan2-2-t2v-a14b)
- The official Wan-Video GitHub org holds exactly 5 public repos that search can see: Wan2.1, Wan2.2, Wan-Dancer (created 2026-07-13), Wan-Animate-2 (created 2026-07-13) and Wan-skills. None is Wan 2.5, 2.6 or 3.x. — GitHub repository search `org:Wan-Video` (queried directly 2026-09-29; results listed [Wan2.2](https://github.com/Wan-Video/Wan2.2), [Wan-Animate-2](https://github.com/Wan-Video/Wan-Animate-2), [Wan-Dancer](https://github.com/Wan-Video/Wan-Dancer))

**Newer Wan versions**
- Wan 2.6 (Dec 2025) is described as a closed commercial API, with up to 15 s multi-shot clips and audio. **[secondary]** — [wan27.org: Is Wan 2.6 open source](https://wan27.org/blog/wan-2-6-open-source-guide); [MindStudio](https://mindstudio.ai/blog/what-is-wan-2-6-video-open-source)
- Wan 3.0 is described as "API-only public beta since August 6, 2026, no weights". Wan 2.5, 2.6, 2.7 and 3.0 are all described as closed, even though Alibaba reportedly pre-announced Wan 3.0 as Apache-2.0 around April 2026. **[secondary]** — [Atlas Cloud blog](https://www.atlascloud.ai/blog/tips/is-wan-3.0-open-source)
  - This is contradicted by [wan27.org](https://wan27.org/blog/latest-wan-model), which claims 1.3B and 14B Apache-2.0 Wan 3.0 weights shipped in April 2026. The official GitHub org listing above supports the "closed" account.

**Wan2.2-Animate-14B (Wan-Animate)**
- It is one model for two tasks. *Animation* replicates the expressions and movements of a reference video onto a character image. *Replacement* inserts the animated character into the reference video and matches its lighting and colour tone. Body motion comes from spatially aligned **skeleton signals**, and expressions come from **implicit facial features**. — [Wan-Animate arXiv 2509.14055](https://arxiv.org/html/2509.14055v1)
- Preprocessing produces a pose video and a face video, plus a background video and mask video in replacement mode. — [Wan-Video/Wan2.2 GitHub](https://github.com/Wan-Video/Wan2.2)

**Wan-Animate-2 (Aug 2026)**
- It is an end-to-end 14B DiT that consumes the driving video directly and "eliminates intermediate motion extractors" for better motion fidelity and identity preservation. It adds **text-driven viewpoint control** that decouples the output camera perspective from the driving video. A distilled Lite variant targets real-time streaming. Default is 720P, with 480P tested on 2x A800 and the default config sized for 8x A800. Apache 2.0. Weights were released 2026-08-07. — [Wan-Video/Wan-Animate-2 GitHub](https://github.com/Wan-Video/Wan-Animate-2); [arXiv 2608.06009 listing](https://arxiv.org/abs/2608.06009)
- The paper frames prior methods in three groups: explicit-motion methods (identity drift from extraction errors), implicit-motion methods (lose fine dynamics) and in-context methods (too costly). — [arXiv 2608.06009 (snippet)](https://arxiv.org/abs/2608.06009)
- Weights are on HF as Wan-AI/Wan2.2-Animate-2-14B, and a fal "Distilled-FlashPack" variant exists. — [HF Wan-AI/Wan2.2-Animate-2-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B); [fal distilled](https://huggingface.co/fal/Wan2.2-Animate-2-Distilled-FlashPack)

**Wan-Dancer (July 2026)**
- A 14B model for music-to-dance video. It produces 720p at 30 fps, is minute-scale with a hierarchical two-stage method, and uses an optical-flow-based motion-continuity loss. It is built on Wan2.1 and DiffSynth-Studio, Apache 2.0, arXiv 2607.09581. It is not pose-controllable, so it is less relevant to us. — [Wan-Video/Wan-Dancer GitHub](https://github.com/Wan-Video/Wan-Dancer)

**Pose, depth and camera control add-ons**
- Wan2.2-Fun-A14B-Control accepts Canny, Depth, **Pose**, MLSD and trajectory control. It supports 512, 768 and 1024 resolutions and was trained on 81-frame clips at 16 fps. Wan2.2-VACE-Fun-A14B is built on T2V-A14B and supports the same controls plus subject reference. "Wan 2.2 14B Fun Camera Control" supports pan, zoom and rotation. Code lives in the VideoX-Fun repo. **[secondary: ComfyUI docs]** — [ComfyUI Wiki: Wan2.2 Fun Control](https://comfyui-wiki.com/en/tutorial/advanced/video/wan2.2/wan2-2-fun-control); [Comfy.org Fun Control workflow](https://comfy.org/es/workflows/video_wan2_2_14B_fun_control-67a816af8a73/)
- **Uni3C** (DAMO/Alibaba, built on Wan2.1) has two parts:
  - PCDController is a plug-and-play camera-control module conditioned on point clouds unprojected from monocular depth.
  - At inference, a "jointly aligned 3D world guidance" puts scene point clouds and **SMPL-X characters** in one space, unifying camera and human-motion control. The camera and human-motion modules can be trained separately.
  - — [Uni3C arXiv 2504.14899](https://arxiv.org/html/2504.14899v1)
- **ReCamMaster** re-renders a *given* video along a new camera trajectory:
  - The open version is built on Wan2.1, because the paper's internal model is proprietary. The README warns that the open model "may not achieve the same results as demonstrated in the demo".
  - Output is 81 frames at 15 fps and 1280x1280 (center-cropped). There are 10 preset trajectories: pan, tilt, zoom, translate and arc.
  - It was trained on MultiCamVideo, synthetic UE5 data with 13.6k scenes x 10 synchronized cameras = 136k videos (37 environments, 66 characters, 93 animations). MIT license, ICCV 2025.
  - — [KwaiVGI/ReCamMaster GitHub](https://github.com/KwaiVGI/ReCamMaster); [ICCV 2025 paper](https://openaccess.thecvf.com/content/ICCV2025/html/Bai_ReCamMaster_Camera-Controlled_Generative_Rendering_from_A_Single_Video_ICCV_2025_paper.html)
- **HumanVid / CamAnimate**: a dataset of 20k 1080p human videos (real plus synthetic from 2.3k avatar assets) with camera annotations, and a baseline that controls human pose and camera together (NeurIPS 2024). **RealisMotion** decomposes trajectory, orientation, action, subject and background control in 3D world space. It notes that most animation methods assume a static camera. — [HumanVid arXiv 2407.17438](https://arxiv.org/html/2407.17438v3); [RealisMotion arXiv 2508.08588](https://arxiv.org/html/2508.08588v1)

### Inferences
- **Driving with an SMPL sequence plus a chosen camera: feasible now through three routes.**
  - (a) Render our SMPL-like proxy's 2D skeleton (DWPose/OpenPose style) from each target camera and feed it to Wan2.2-Animate or Fun-Control, with the splat render from the same camera as the reference image.
  - (b) Use Uni3C, which takes SMPL-X plus a camera trajectory natively.
  - (c) Use Wan-Animate-2 with a driving video rendered from our proxy and a text viewpoint instruction.
  - Route (a) is the most controllable because both the skeleton and the reference image are rendered from the same known camera, so the camera is implicit in the conditioning. Each clip is then *body-pose-consistent* by construction. It is not *cloth-consistent*, because every clip samples its own fabric dynamics.
- **Clips are short.** Most Wan-family control models are trained on ~81 frames at 16 fps (about 5 s). Any long motion has to be chunked, which adds seams at chunk boundaries.
- **Skeleton control leaves the cloth to the model.** The skeleton-based Animate path hands garment geometry entirely to the video prior. That is what we want for "learning" fabric motion, but it also means loose garments may be reshaped (see Q4/Q5).
- **Wan-Animate-2 is a double-edged sword.** Because it removes the skeleton extractor, it could be driven by *our own rendered splat animation* (e.g., LBS-skinned) as the driving video, letting the model "add" cloth dynamics. Whether it keeps silhouettes aligned to the driver is unverified.
- **Plan on open weights only.** Given the closed Wan 2.5–3.0 line, build on Wan 2.2 A14B, Animate(-2), VACE/Fun and Wan2.1-based research code.

### Gaps
- I could not access the Wan-Animate or Wan-Animate-2 papers directly (arXiv blocked). Missing: exact frame counts, clip length, quantitative identity/clothing metrics, and how fine-grained the "text-driven viewpoint control" is (discrete prompts such as "side view", or continuous angles). Also unknown: whether it can hold motion fixed while changing only the viewpoint.
- VRAM for Animate-14B and Wan-Animate-2 on a single consumer GPU was not documented in what I could read.
- I did not verify whether a Wan2.2-native (rather than Wan2.1) release of ReCamMaster or Uni3C exists.
- No source evaluates identity or clothing preservation of Wan-Animate specifically on loose garments (skirts, coats).

---

## Q2. Multi-view-consistent / novel-view video generation that could give synchronized views directly

### Takeaway
The strongest route to beating inconsistency is to generate **one** monocular clip (Wan-Animate) and then expand it into synchronized multi-view video with a novel-view video model. Human-specialised, Wan-based options were published in 2025–2026:
- **MV-Performer** (SIGGRAPH Asia 2025)
- **Flex4DHuman** (Jun 2026, Wan2.1-1.3B, pose-free)
- **4DAnyone** (SIGGRAPH Asia 2026, scales to tens of consistent views for 4DGS)
- **Diffuman4D** (sparse-view input)

General-purpose options include ReCamMaster, TrajectoryCrafter, GEN3C, SV4D 2.0 and CAT4D, plus SynCamMaster for text-to-synchronized multi-camera generation. Because all views then come from one generated motion, the cloth dynamics are shared. The open question is how faithfully unseen-side fabric is hallucinated. None of these papers evaluates loose-garment dynamics specifically.

### Cited Findings
- **MV-Performer**:
  - Generates synchronized 360° novel-view videos from a monocular full-body capture, using a video diffusion model with depth-based warping and camera-dependent normal maps from oriented partial point clouds. Trained on MVHumanNet.
  - Stated limitations: it depends on stable depth estimation and fails when depth is poor; the VAE limits face detail; multi-step denoising is slow.
  - — [MV-Performer arXiv 2510.07190](https://arxiv.org/html/2510.07190v1); [ACM SIGGRAPH Asia 2025](https://dl.acm.org/doi/full/10.1145/3757377.3763935)
- **Flex4DHuman**:
  - Turns monocular or sparse multi-view video into synchronized dense multi-view video using only relative camera-pose and text conditioning, with *no* skeleton, depth, normal or rendered target geometry.
  - Built on Wan2.1-T2V-1.3B with a five-axis RoPE (space, time, view index, continuous SE(3) relative camera). Outputs feed directly into 4D Gaussian splat reconstruction.
  - Reported to beat prior SOTA on DNA-Rendering and ActorsHQ. Code is on GitHub.
  - — [Flex4DHuman arXiv 2606.13655](https://arxiv.org/html/2606.13655v1); [GitHub Andy-Cheng/Flex4DHuman](https://github.com/Andy-Cheng/Flex4DHuman)
- **4DAnyone**:
  - Reconstructs 4D humans from an uncalibrated casual monocular video by generating "reconstruction-grade" multi-view-consistent videos and lifting them to 4DGS.
  - Key finding: camera-controlled video diffusion models look plausible per view but **lose consistency when scaled to the tens of views that 4DGS needs**. The authors call this a bounded attention-context problem: views must be split into groups when they exceed one DiT forward pass.
  - Fixes: Reference Context Packing (fixed-length, mixed-resolution reference context) and Target Context Routing (rotating view groupings across denoising steps).
  - Code is at ant-research/4DAnyone.
  - — [4DAnyone arXiv 2608.20335](https://arxiv.org/html/2608.20335); [project page](https://4danyone.github.io/); [GitHub](https://github.com/ant-research/4DAnyone)
- **Diffuman4D** (ICCV 2025): 4D-consistent human view synthesis from *sparse-view* videos. It uses a sliding iterative denoising process over a latent grid that encodes image, camera pose and human pose per view and timestamp. — [arXiv 2507.13344](https://arxiv.org/abs/2507.13344)
- **Human4DiT** (2024): a 4D diffusion transformer that generates 360° spatio-temporally coherent human video from a reference image. It can produce monocular, multi-view, static-3D and rotating video. — [arXiv 2405.17405](https://arxiv.org/html/2405.17405v2)
- **SV4D 2.0** (Stability, ICCV 2025) turns a monocular video into multiple novel-view videos, then optimises 4D. Improvements over SV4D: no dependency on reference multi-views, blended 3D and frame attention, progressive 3D→4D training, and a 2-stage refinement with progressive frame sampling to "handle 3D inconsistency and large motion". — [SV4D 2.0 ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_SV4D_2.0_Enhancing_Spatio-Temporal_Consistency_in_Multi-View_Video_Diffusion_for_ICCV_2025_paper.html); [Stability AI blog](https://stability.ai/news/stable-video-4d-20-new-upgrades-for-high-fidelity-novel-views-and-4d-generation-from-a-single-video)
- **CAT4D** is a multi-view video diffusion model trained on mixed datasets. It synthesises novel views at any specified camera and timestamp from monocular video and then fits a 4D scene. — [CAT4D arXiv 2411.18613](https://arxiv.org/pdf/2411.18613)
- **4Real-Video-V2** uses fused view-time attention plus feed-forward reconstruction for 4D scene generation. — [arXiv 2506.18839](https://arxiv.org/html/2506.18839v1)
- **ReCamMaster** (see Q1): re-renders an input video along new camera trajectories while keeping the dynamics synchronized. It is trained on 10-camera synchronized UE5 data that includes 66 animated characters. — [GitHub](https://github.com/KwaiVGI/ReCamMaster)
- **TrajectoryCrafter** (ICCV 2025) redirects camera trajectories of monocular videos using video depth and a "double-reprojection" training strategy. **GEN3C** (NVIDIA, CVPR 2025) conditions on renders of a 3D point-cloud cache built from predicted depth. It reports SOTA sparse-view NVS, including on monocular dynamic video. — [TrajectoryCrafter ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Yu_TrajectoryCrafter_Redirecting_Camera_Trajectory_for_Monocular_Videos_via_Diffusion_Models_ICCV_2025_paper.pdf); [GEN3C NVIDIA](https://research.nvidia.com/labs/toronto-ai/GEN3C)
- **SynCamMaster** generates multiple synchronized videos of the same dynamic scene from a text prompt plus several camera parameters. — [arXiv 2412.07760](https://arxiv.org/html/2412.07760v1)
- **UniWorld-View** (Aug 2026) does large-baseline view synthesis with video diffusion models. It appeared in the human-NVS search results; I did not examine it. — [arXiv 2608.04701](https://arxiv.org/pdf/2608.04701)

### Inferences
- **Recommended pipeline:** render splat → Wan2.2-Animate (pose from SMPL proxy) produces one "hero" monocular clip → a human novel-view model (Flex4DHuman or 4DAnyone; MV-Performer if a normal/depth prior helps) produces N synchronized views → run a Gaussian-Garments-style multi-view pipeline on those views.
  - This turns "many mutually inconsistent clips" into "one clip plus a consistent multi-view expansion".
  - Diversity of motion then comes from multiple hero clips with *different* motions, each expanded independently. That is equivalent to multiple multi-view capture sequences, which is exactly what Gaussian Garments consumes.
- **Our static splat is a strong extra input for these models.** We can render true multi-view reference images at t=0 (and the proxy's depth or normals from any camera). That largely removes the "unseen side" hallucination at the first frame. MV-Performer-style depth/normal conditioning, or GEN3C/TrajectoryCrafter point-cloud renders, can use it directly.
- **Scale the view count carefully.** 4DAnyone's finding suggests naive grouping of tens of views breaks consistency. Gaussian Garments-style reconstruction may need ~10–20 views, so choose a model that addresses view-count scaling, or accept fewer views.
- **General camera models are weakest on deforming humans.** ReCamMaster and TrajectoryCrafter are trained mostly on scenes, UE5 characters or general video. Depth-warp methods can smear thin, fast-moving fabric edges because monocular video depth is weakest exactly there. Human-specific models (MV-Performer, Flex4DHuman, 4DAnyone, Diffuman4D) are the better bets.

### Gaps
- No paper found that evaluates novel-view video models specifically on loose-garment or secondary cloth motion (skirts, capes). The DNA-Rendering and ActorsHQ benchmarks contain some loose clothing, but I found no per-garment analysis.
- Weight availability and licences for MV-Performer, SV4D 2.0 and CAT4D were not verified. I believe CAT4D has no public code, but this is unverified.
- Resolution, view count and frame count of Flex4DHuman and 4DAnyone outputs were not accessible (arXiv blocked).

---

## Q3. Methods that fit 4D Gaussians / dynamic avatars to generated videos, and their strategies for inconsistency

### Takeaway
Four families exist:
- **(a) SDS / score distillation** from image, multi-view or video diffusion: Consistent4D, DreamGaussian4D, AYG, STAG4D, SC4D. Robust to inconsistency because it never fits pixels exactly, but tends to blur and oversmooth motion.
- **(b) Generate-then-reconstruct**: SV4D 2.0, CAT4D, 4DAnyone, Flex4DHuman, L4GM (feed-forward).
- **(c) Absorbing inconsistency into extra degrees of freedom**: AniGS reformulates 3D-from-inconsistent-views as **4D reconstruction with a per-frame deformation/colour module**. PERSONA uses balanced sampling and geometry-weighted losses.
- **(d) Low-dimensional physical parameter fitting** against generated video: PhysDreamer, DreamPhysics.

For our goal, accumulating fabric knowledge across clips that don't agree, (c) and (d) are the most transferable. Per-clip latent codes or residual deformations absorb per-clip disagreement, while a shared low-dimensional physical model (material parameters, a GNN simulator) captures what is common.

### Cited Findings
- **AniGS** (CVPR 2025):
  - Adapts a transformer T2V model to generate multi-view canonical-pose images plus normals.
  - To handle inconsistency among the generated views, it *"reformulate[s] the problem of 3D reconstruction from inconsistent images as a 4D reconstruction task"*. It uses a canonical 3DGS plus a per-frame deformation module that estimates shape and colour variation of each Gaussian conditioned on the frame index. The canonical model is then used for SMPL-X-driven animation.
  - — [AniGS arXiv 2412.02684](https://arxiv.org/html/2412.02684v1); [CVPR 2025 poster](https://cvpr.thecvf.com/virtual/2025/poster/34215)
- **PERSONA** (ICCV 2025, Sim & Moon):
  - Generates pose-rich videos from one image with a diffusion model, then optimises a 3DGS avatar with pose-driven (cloth) deformations.
  - Two tricks:
    - **Balanced sampling** oversamples the real input image to counter identity drift in generated frames.
    - **Geometry-weighted optimization** prioritises geometry constraints over image loss so rendering stays good in diverse poses.
  - It names the core problem as diffusion outputs suffering "pose-dependent identity entanglement".
  - — [PERSONA arXiv 2508.09973](https://arxiv.org/html/2508.09973v1); [ICCV 2025 open access](https://openaccess.thecvf.com/content/ICCV2025/html/Sim_PERSONA_Personalized_Whole-Body_3D_Avatar_with_Pose-Driven_Deformations_from_a_ICCV_2025_paper.html)
- **Other single-image avatar pipelines that lean on video-diffusion multi-views**:
  - **Dream, Lift, Animate (DLA)**: video-diffusion multi-views → 3DGS → pose-aware UV-space Gaussians. — [arXiv 2507.15979](https://arxiv.org/html/2507.15979v2)
  - **SVAD**: single image to 3D avatar through synthetic data from video diffusion plus augmentation. — [arXiv 2505.05475](https://arxiv.org/pdf/2505.05475)
  - **Forwardrobe** (Jul 2026): garment-aware Gaussian avatars from a single image. Its "Garment Dynamic Module" combines LBS coarse motion with pose-dependent geometry and appearance residuals, with separate garment representations. — [arXiv 2607.29106](https://arxiv.org/html/2607.29106)
  - **Generator-Refiner-Examiner** (May 2026) is a data-augmentation framework for avatar learning from monocular video. It is listed only; I did not examine it. — [arXiv 2605.23555](https://arxiv.org/pdf/2605.23555)
- **SDS / video-to-4D family**:
  - Consistent4D optimises a dynamic NeRF by SDS from 2D diffusion. — [4D survey arXiv 2503.14501](https://arxiv.org/pdf/2503.14501)
  - STAG4D anchors multi-view diffusion outputs on input frames. It uses first-frame temporal anchoring in self-attention and a training-free temporal attention module, because per-frame multi-view generation is temporally inconsistent. — [STAG4D ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5288_ECCV_2024_paper.php)
  - SC4D learns sparse control Gaussians for shape and motion, which then drive dense splats. — [survey / search summary](https://arxiv.org/pdf/2503.14501)
  - L4GM is a feed-forward 4D Gaussian reconstruction model rather than per-scene optimisation. — [L4GM arXiv 2406.10324](https://arxiv.org/pdf/2406.10324)
  - An aligned critique: "Not All Frame Features Are Equal" decouples dynamic and static features for video-to-4D. — [arXiv 2502.08377](https://arxiv.org/pdf/2502.08377)
- **SV4D 2.0** explicitly adds a 2-stage 4D refinement and progressive frame sampling to handle residual 3D inconsistency in its own generated views. — [SV4D 2.0 ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Yao_SV4D_2.0_Enhancing_Spatio-Temporal_Consistency_in_Multi-View_Video_Diffusion_for_ICCV_2025_paper.html)
- **4Real** (NeurIPS 2024) uses video diffusion models to generate photorealistic 4D scenes. It appears in the same line of generate-then-reconstruct work with deformation absorbing inconsistency. Details not verified. — [4Real arXiv 2406.07472](https://arxiv.org/pdf/2406.07472)
- **PhysDreamer**:
  - Represents objects as 3D Gaussians with a neural material field and simulates them with differentiable MPM.
  - It generates a video with a video-generation model, runs the simulation, renders it, compares frame by frame with the generated video, and backpropagates into material parameters.
  - This distils dynamics priors from a video model into physical properties.
  - — [PhysDreamer arXiv 2404.13026](https://arxiv.org/html/2404.13026v2); [IAIFI summary](https://research.iaifi.org/posts/physdreamer-physics-based-interaction-with-3d-objects-via-video-generation)
- **DreamPhysics** (AAAI 2025) learns a material field with video diffusion priors (score distillation, not a fixed generated clip), driving an MPM simulator. It adds "motion distillation sampling" to emphasise motion information and a KAN-based material field with frame boosting. It notes that earlier methods produced small or discontinuous motions. — [DreamPhysics arXiv 2406.01476](https://arxiv.org/html/2406.01476v3); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32389)
- **Cloth-specific physics-from-vision**:
  - **Gaussian Garments** fine-tunes a pre-trained GNN cloth simulator (parameters, material vectors, rest geometry) to match observed multi-view motion. — [Gaussian Garments project page](https://eth-ait.github.io/Gaussian-Garments/)
  - **CloDS** (ICLR 2026) learns cloth dynamics *unsupervised from multi-view video* under unknown physical conditions. It first grounds video to mesh geometry with a "Spatial Mapping Gaussian Splatting" module (dual-position opacity modulation), then trains a dynamics model on the grounded meshes. — [CloDS arXiv 2602.01844](https://arxiv.org/pdf/2602.01844); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10009027)
  - **PGC** (physics-based Gaussian cloth from a single pose). — [arXiv 2503.20779](https://arxiv.org/pdf/2503.20779)
  - **Dress-1-to-3** (single image to simulation-ready outfit, diffusion prior plus differentiable physics). — [arXiv 2502.03449](https://arxiv.org/pdf/2502.03449)
  - **Image2Garment** (VLM infers fabric attributes, mapped to physical parameters). — [arXiv 2601.09658](https://arxiv.org/html/2601.09658v3)
- **Learned garment dynamics priors that could be the "shared model" fitted across clips**:
  - **D-Garment** is a latent diffusion model of garment deformation parameterised by bending, stretching and density. — [arXiv 2504.03468](https://arxiv.org/html/2504.03468)
  - **DiT-Garment** (Sept 2026) is a diffusion transformer on UV position maps conditioned on body motion and physical parameters. It was trained on synthetic simulations and generalises to captured and artist garments. — [arXiv 2609.18510](https://arxiv.org/html/2609.18510)
  - **DSAR** (Aug 2026): dual-stream autoregressive modelling of temporal cloth dynamics for animatable avatars. — [arXiv 2608.10500](https://arxiv.org/pdf/2608.10500)

### Inferences
- **Proposed recipe for incoherent Wan clips**, combining AniGS, PERSONA and PhysDreamer ideas:
  1. **Keep one canonical splat fixed.** It is the ground truth appearance; PERSONA-style, treat it as the high-weight anchor.
  2. **Absorb per-clip disagreement.** Give each clip a per-clip latent code and a small residual deformation/colour field (AniGS-style per-frame deformation, generalised to per-clip plus per-frame). This absorbs hallucinated appearance changes, garment morphing and camera or scale mismatch.
  3. **Share the physics.** Share a *low-dimensional* physical model across all clips: a GNN cloth simulator with per-garment material vectors (Gaussian Garments), or a D-Garment/DiT-Garment prior with physical parameters. Every clip must be explained by the same material parameters driven by the known SMPL motion, plus a heavily regularised per-clip residual.
  4. **Use robust, confidence-weighted losses.** Weight by per-clip reprojection agreement with the body proxy, silhouette/segmentation IoU against the known clothing labels, and optical-flow consistency. Down-weight or reject clips where the garment topology changes, for example a skirt becoming trousers.
  5. **Supervise on low-frequency cues.** Prefer silhouettes, mask boundaries, optical flow and 2D tracks over per-pixel RGB. Fabric-motion knowledge lives mostly in silhouette swing and fold motion, and RGB is where texture swimming happens.
- **Monocular clips constrain the cloth physics even though they conflict geometrically.** Because the body motion is known (the SMPL condition), each clip gives a 2D observation of cloth response to a known forcing. That is a well-posed low-dimensional inverse problem, much like PhysDreamer, which fits material from a single generated view.
- **SDS from Wan is possible but costly (14B) and blurs high-frequency cloth.** Direct reconstruction losses on sampled clips, with robust weighting, are more practical. An SDS-like "video prior score" could still serve as a regulariser for physically unobserved sides.

### Gaps
- I did not find any published method that fits *garment physics parameters* specifically to *video-diffusion-generated* clothed-human video. PhysDreamer and DreamPhysics target objects (plants, elastic items) with MPM. This appears to be an open niche, unverified beyond my searches.
- Not individually verified (not fetched): Animate124, 4DGen, DreamGaussian4D, AYG, Disco4D, AvatarGO, MotionDreamer and Dream-in-4D. I only know them from background knowledge, so no claims are made about them.
- I found no papers that use "per-clip latent codes" explicitly for generated-video inconsistency in human avatars. AniGS's per-frame deformation is the closest analogue found.

---

## Q4. Physical plausibility of cloth in generated video: benchmarks and failure modes

### Takeaway
General physics benchmarks consistently show that video models, Wan2.1 included, are poor at physics:
- Best VideoPhy-2 hard-split score is about 22%.
- The original Physics-IQ best was 29.5/100.
- Newer world models (Cosmos3) reach about 44–60.
- Physical accuracy is "unrelated to visual realism".

None of the major benchmarks has a dedicated cloth/garment category (Physics-IQ covers solid mechanics, fluids, optics, thermodynamics and magnetism). Generated fabric motion should therefore be treated as *visually plausible, not physically accurate*, which argues for pairing it with a physical simulator prior rather than trusting it as ground truth.

### Cited Findings
- **Physics-IQ**: 396 real scenes across solid mechanics, fluid dynamics, optics, thermodynamics and magnetism, each filmed from 3 fixed viewpoints.
  - Best original model (VideoPoet multiframe) scored 29.5% of 100.
  - Physical understanding is "severely limited, and unrelated to visual realism" across Sora, Runway, Pika, Lumiere, SVD and VideoPoet.
  - — [Physics-IQ arXiv 2501.09038](https://arxiv.org/pdf/2501.09038); [WACV 2026](https://wacv.thecvf.com/virtual/2026/poster/440)
- **Physics-IQ, newer results**: Cosmos3-Super reports 43.8 (I2V direct) and 59.7 (V2V direct). With WMReward and best-of-N it reaches 48.9 and 63.4. — [Cosmos 3 arXiv 2606.02800](https://arxiv.org/pdf/2606.02800)
- **Physics-IQ Verified** is a 2026 revision of the benchmark; not examined. — [arXiv 2606.18943](https://arxiv.org/pdf/2606.18943)
- **VideoPhy-2** (ICLR 2026): 3,940 action-centric prompts with human Likert evaluation.
  - The best model, **Wan2.1-14B**, achieves only 32.6% (full) and 21.9% (hard) joint semantic-plus-physical adherence.
  - Models struggle especially with conservation of mass and momentum.
  - One secondary source mentions a 47.7% figure, which conflicts; the paper's headline is ~22% on hard.
  - — [VideoPhy-2 arXiv 2503.06800](https://arxiv.org/pdf/2503.06800); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10010425)
- **PhyGenBench** has 160 prompts with an automated PhyGenEval, and is criticised as too small. Other benchmarks: PhyWorldBench (2025) and PhyGround (2026). — [VideoPhy-2 related work](https://arxiv.org/pdf/2503.06800); [PhyWorldBench](https://arxiv.org/html/2507.13428v2); [PhyGround arXiv 2605.10806](https://arxiv.org/pdf/2605.10806)
- **Human-animation failure modes named in the literature**:
  - Explicit-motion (skeleton) methods suffer "extraction errors and identity drift". Implicit-motion methods "lose fine-grained dynamics". — [Wan-Animate-2 arXiv 2608.06009](https://arxiv.org/abs/2608.06009)
  - Diffusion-generated avatar training videos show "identity shifts" and "pose-dependent identity entanglement". — [PERSONA](https://arxiv.org/html/2508.09973v1)
  - Generated views are inconsistent at extreme poses. — [AniGS](https://arxiv.org/html/2412.02684v1)
  - Video try-on must filter back-facing frames and apply 3D mask smoothing to get temporal consistency. — [CatV2TON](https://arxiv.org/html/2501.11325v1)
- **Skeleton priors alone miss loose-clothing deformation**: methods relying solely on skeletal or parametric body priors often fail to capture dresses and coats realistically. — [RealityAvatar arXiv 2504.01559](https://arxiv.org/html/2504.01559v1)

### Inferences
- **Expected Wan failure modes on fabric:**
  - Texture "swimming", where prints slide over the garment surface.
  - Garment morphing: hemline length, garment type or silhouette changing across frames or across clips.
  - Frame-to-frame fold flicker, and cloth sticking to the body when a skeleton drives it.
  - Physically implausible momentum, e.g. a skirt that doesn't lag or overshoot.

  These are consistent with the cited identity-drift and conservation-law findings, but **no source quantifies them for garments specifically**.
- **Consequence for our pipeline:** use generated clips mainly for *qualitative* motion statistics (amplitude, lag, fold patterns) that a physically constrained model (GNN simulator or D-Garment prior) is fitted to. Don't use them as ground-truth trajectories. A simulator-in-the-loop also filters out physically impossible hallucinations.
- **Don't generalise from Wan2.1's benchmark ranking.** Wan2.1-14B topping VideoPhy-2 suggests Wan is among the best open models for physics. Absolute scores remain low, and no Wan2.2 or Animate result on a physics benchmark was found.

### Gaps
- I found no benchmark with a cloth or fabric subcategory, and no quantitative study of garment-dynamics accuracy in human-animation models (Wan-Animate, UniAnimate, Animate Anyone 2).
- I found no WorldModelBench results in this session.
- No metrics were found for "texture swimming" or "garment topology change".

---

## Q5. Human-motion and garment-specific animation / video try-on models

### Takeaway
Human image animation has moved onto large video DiTs:
- **UniAnimate-DiT**: Wan2.1 plus LoRA plus a 3D-conv pose encoder.
- **Animate Anyone 2**: environment affordance.
- **Wan-Animate / Wan-Animate-2**: Alibaba's own.

Video try-on models (CatV2TON, Dynamic Try-On, ViViD) explicitly target garment fidelity over time, using garment-feature injection and limb-aware attention. They are designed to keep *appearance* consistent, not to produce physically correct dynamics. For us, try-on style garment conditioning could reduce garment morphing across clips.

### Cited Findings
- **UniAnimate-DiT** builds on Wan2.1 with LoRA fine-tuning and a lightweight pose encoder of stacked 3D convolutions, plus reference-pose conditioning for appearance alignment. It was trained at 480p (832x480) and generalises to 720p at inference. — [UniAnimate-DiT arXiv 2504.11289](https://arxiv.org/html/2504.11289v1); [HF model](https://huggingface.co/ZheWang123/UniAnimate-DiT)
- **Animate Anyone 2** (ICCV 2025) conditions on environment representations (the region excluding the character) with a shape-agnostic mask strategy and an object guider for interactions. — [Animate Anyone 2 arXiv 2502.06145](https://arxiv.org/html/2502.06145v1); [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Hu_Animate_Anyone_2_High-Fidelity_Character_Image_Animation_with_Environment_Affordance_ICCV_2025_paper.html)
- **CatV2TON**: a single DiT for image and video try-on via temporal concatenation of garment and person inputs. It uses overlapping clip-based inference with Adaptive Clip Normalization (AdaCN) for long-video consistency, and introduces the ViViD-S dataset (ViViD filtered of back-facing frames, with 3D mask smoothing). — [CatV2TON arXiv 2501.11325](https://arxiv.org/html/2501.11325v1)
- **Dynamic Try-On** (BMVC 2025): a DiT-based video try-on that uses the DiT backbone itself as garment encoder, with a dynamic feature-fusion module and **limb-aware dynamic attention** for temporal consistency of body parts. — [BMVC 2025](https://bmvc2025.bmva.org/proceedings/602/)
- **Wan-Animate**: skeleton plus implicit-face control, and replacement mode with relighting to match the scene. — [arXiv 2509.14055](https://arxiv.org/html/2509.14055v1)
- **MonoCloth** reconstructs and animates cloth-decoupled avatars from monocular video. **Gaussian Wardrobe** builds compositional 3DGS avatars for try-on. Both are relevant as garment-separated representations that could consume generated video. — [MonoCloth arXiv 2508.04505](https://arxiv.org/pdf/2508.04505); [Gaussian Wardrobe arXiv 2603.04290](https://arxiv.org/pdf/2603.04290)

### Inferences
- **UniAnimate-DiT is the fallback if Animate-14B is too heavy.** It is a lighter Wan2.1 route for pose-driven animation.
- **Test models empirically on our own subject.** Wan-Animate(-2) should be compared with UniAnimate-DiT on garment consistency, measuring per-frame garment-mask IoU against the splat's clothing labels projected via the proxy.
- **Try-on conditioning could stabilise garment appearance.** Try-on-style garment reference conditioning (for example a VACE subject reference built from garment-only renders of the splat) is a plausible way to reduce garment morphing across clips. This is untested.

### Gaps
- MimicMotion, StableAnimator, HumanDiT, Champ, ViViD and Fashion-VDM were not verified in this session, so no claims are made about them.
- I found no quantitative comparison of these models on loose-garment secondary motion.
