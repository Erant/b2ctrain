#pragma once
#include "gpu/util.cuh"
#include "model.h"
#include <string>
#include <vector>

namespace b2c {

// Cage deformation of the canonical splat (b2crig: animating a trained subject).
//
// A cage file carries one or more triangle-mesh LAYERS in the canonical pose (the MHR body, and one mesh per loose
// garment or hair layer that b2crig builds from the splat's seg labels), the Sapiens2 classes each layer owns, and per
// FRAME the posed position of every cage vertex (b2crig skins the cages; the body layer comes straight from the MHR
// forward, pose correctives included). Every splat is bound to the nearest triangle of the layer that owns its
// seg_label (splats whose class no layer owns, or whose seg_conf is under `min_conf`, take the nearest triangle of any
// layer). GaussianAvatars-style: a splat keeps its barycentric foot point on the triangle and its offset from it in
// the triangle's frame, scaled by the triangle's size; its rotation turns with the frame and its scales grow with it.
// The SH view direction is rotated back into the canonical frame, so the colour does not swim as the surface turns.
//
// Per frame the cage vertices may carry a learnable displacement `delta` (fit-cage), added to the posed vertices
// before the triangle frames are built. Its gradient takes the translational path only: a splat's positional gradient
// goes to its triangle's three vertices by barycentric weight; the offset's dependence on the frame is not
// differentiated (the offsets are millimetres to a few centimetres, and the cage Laplacian keeps the frame smooth).
//
// Dual binding (b2crig's binding fit): where the seg label cannot say which limb a splat belongs to (a hand resting at a
// pocket, an arm against the torso), b2crig lists the splat with a second candidate triangle B. Such a splat is posed as
// the blend (1 - w) A + w B of its two bindings (position and scales; rotation and SH frame from the nearer one), and
// fit-cage --fit-binding learns w from the video. File (b2crig writes it): char magic[8] = "B2CALT01"; int32 n;
// n x { int32 splat, int32 face_b (global), float w }.
//
// File layout (little-endian), written by b2crig (b2crig/b2ctrain.py:write_cage):
//   char magic[8] = "B2CCAGE1"
//   int32 n_layers, n_verts, n_faces, n_frames
//   per layer: int32 v_off, v_count, f_off, f_count; uint32 class_bits (bit c = Sapiens2 class c)
//   float3 verts[n_verts]           canonical, world
//   int3   faces[n_faces]           global vertex indices, each face inside its layer's vertex range
//   char   names[n_frames][64]      frame (camera / image) names
//   float3 posed[n_frames][n_verts]
//   optional, the opening gate of two-state splats (gpu/cage_open.h; b2crig/rig/open_gate.py computes it):
//   char   tag[8] = "B2COPEN1"; float radius, min_dist (how b2crig measured it, for the log);
//   half   theta[n_frames][n_verts] degrees
struct CageRig {
  struct Layer { int v_off, v_count, f_off, f_count; uint32_t class_bits; };
  int nv = 0, nf = 0, nframes = 0;
  std::vector<Layer> layers;
  std::vector<std::string> names;
  std::vector<int> face_layer_h;             // [nf]
  DevBuf<float3> verts0;                     // [nv] canonical
  DevBuf<int3> faces;                        // [nf]
  DevBuf<float3> posed;                      // [nframes][nv]
  DevBuf<float3> delta; bool has_delta = false;   // [nframes][nv], learnable (fit-cage)
  DevBuf<float3> verts_view;                 // [nv] posed + delta of the current frame
  int frame = -1;                            // the frame pose() last posed
  DevBuf<float> open_theta; bool has_open_theta = false;   // [nframes][nv] the file's B2COPEN1 section (degrees)
  float open_radius = 0.f, open_min_dist = 0.f;
  DevBuf<float4> tri_q0;                     // [nf] canonical triangle frame as a quaternion (w, x, y, z)
  DevBuf<float> tri_k0;                      // [nf] canonical triangle size
  DevBuf<float4> tri_dq; DevBuf<float> tri_k; DevBuf<float> tri_R;  // [nf] current frame: quat(R R0^T), size k, R row-major [nf][9]
  // per-splat binding
  DevBuf<int> bind_f; DevBuf<float2> bind_b; DevBuf<float3> bind_off;   // triangle, barycentrics (b1, b2), offset from the foot point in the canonical triangle frame, / k0
  int bound_n = 0;
  float max_growth = 0.f;                    // > 0: clamp the triangle size splats follow to [k0/g, k0*g] (tri_pose_kernel)
  // fade_end > fade_start > 0: a bound splat's opacity scales linearly to 0 as its triangle's (unclamped) size ratio goes
  // from fade_start to fade_end: LBS stretches an opening crease (the armpit as the arm lifts) into a membrane whose
  // splats render a fuzzy web; they fade instead (the posed .w opacity logit; its gradient passes straight through)
  float fade_start = 0.f, fade_end = 0.f;
  // Filler splats (ply cage_fill > 0; b2crig tools/armpit_fill.py): the fade's mirror. Their opacity rises linearly from 0
  // to its value as the triangle's unclamped ratio goes fill_start -> fill_end (fade_* when unset): they cover the skin an
  // opening crease reveals, which the capture never showed, and stay invisible near the canonical pose.
  float fill_start = 0.f, fill_end = 0.f;
  // cage_fill < 0: a crease splat that fades OUT over the same range. cage_gate_a/_b (ply): a pair of cage vertices, one
  // on each wall of the crease; a fill / crease splat reads their distance ratio posed / canonical instead of a triangle
  // stretch (the walls only rotate apart, the fold in between stretches; the distance says how open the crease is).
  DevBuf<float> fill; bool has_fill = false;   // [n]
  DevBuf<int2> gate; bool has_gate = false;    // [n] vertex pair, (-1, -1) = the splat's own face stretch
  void set_fill(const std::vector<float>& f, const std::vector<float>& g, cudaStream_t stream);   // g empty = no gates
  DevBuf<float> tri_ratio;                   // [nf] unclamped k / k0 of the current frame (fade only)
  int n_fallback = 0;                        // splats bound across layers (class unowned or low confidence)
  // dual binding (see above): splat index, triangle B, its barycentrics and offset, the weight of B
  int n_alt = 0;
  DevBuf<int> alt_idx, alt_f; DevBuf<float2> alt_b; DevBuf<float3> alt_off; DevBuf<float> alt_w;
  DevBuf<float3> alt_dpos;                   // [n_alt] pos_B - pos_A of the current frame (the weight's gradient direction)
  // posed splat parameters for the current frame
  DevBuf<float4> pos_view, quat_view, lscale_view, sh_frame;

  bool load(const std::string& path);        // false when missing; throws on malformed
  int frame_index(const std::string& name) const;
  // Bind the model's splats (labels: host [n][2] seg_label, seg_conf, or empty for "any layer").
  void bind(const Model& m, const std::vector<float>& labels, float min_conf, cudaStream_t stream);
  // Posed means / rotations / scales / SH frames of frame `f` into the *_view buffers.
  void pose(int f, const Model& m, cudaStream_t stream);
  // fit-cage: allocate zeroed per-frame displacements.
  void enable_delta(cudaStream_t stream);
  // Re-derive every bound splat's offset from its current canonical position (triangle and barycentrics kept):
  // after the canonical means moved (fit-cage --fit-splats), so pose() follows them.
  void refresh_offsets(const Model& m, cudaStream_t stream);
  // Dual binding: after bind(); triangle B's foot point and offset come from the splat's canonical position.
  void bind_alt(const Model& m, const std::vector<int>& splat, const std::vector<int>& face_b, const std::vector<float>& w, cudaStream_t stream);
  // Per face: the largest size ratio k / k0 it reaches over every frame of the cage (unclamped). A pose library can
  // ride along as extra frames no training view uses.
  std::vector<float> face_stretch(cudaStream_t stream);
  // Per bound splat: weight * log(stretch of its face), 0 where the face never grows (host, [n]).
  std::vector<float> splat_stretch_weights(const std::vector<float>& stretch, float weight, cudaStream_t stream);
};

bool load_alt_binding(const std::string& path, std::vector<int>& splat, std::vector<int>& face_b, std::vector<float>& w);
void save_alt_binding(const std::string& path, const std::vector<int>& splat, const std::vector<int>& face_b, const std::vector<float>& w);

}  // namespace b2c
