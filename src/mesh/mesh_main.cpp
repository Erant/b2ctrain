#include "mesh/mesh_main.h"
#include <cstring>
namespace b2c {
int mesh_main(int argc, char** argv) {
  if (argc < 2) return -1;
  const char* s = argv[1];
  if (!strcmp(s, "mesh-fuse")) return mesh_fuse_main(argc, argv);
  if (!strcmp(s, "mesh-refine")) return mesh_refine_main(argc, argv);
  if (!strcmp(s, "mesh-bake")) return mesh_bake_main(argc, argv);
  if (!strcmp(s, "mesh-unwrap")) return mesh_unwrap_main(argc, argv);
  if (!strcmp(s, "mesh-render")) return mesh_render_main(argc, argv);
  if (!strcmp(s, "mesh-backproject")) return mesh_backproject_main(argc, argv);
  return -1;
}
}  // namespace b2c
