#pragma once
// The `b2ctrain mesh-*` subcommands (docs/mesh-plan.md): fuse, refine, bake, unwrap, render, backproject.
namespace b2c {
int mesh_fuse_main(int argc, char** argv);
int mesh_refine_main(int argc, char** argv);
int mesh_bake_main(int argc, char** argv);
int mesh_unwrap_main(int argc, char** argv);
int mesh_render_main(int argc, char** argv);
int mesh_backproject_main(int argc, char** argv);
// Dispatch on argv[1]; returns -1 when it is not a mesh subcommand.
int mesh_main(int argc, char** argv);
}  // namespace b2c
