//! Build the native CPU kernel library, benchmark, and numerical tests.
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const core = b.createModule(.{ .root_source_file = b.path("native/kernels.zig"), .target = target, .optimize = optimize });
    const lib = b.addLibrary(.{ .name = "kernel_optimizer", .linkage = .dynamic, .root_module = core });
    b.installArtifact(lib);
    const cli = b.addExecutable(.{ .name = "kernel-bench-zig", .root_module = b.createModule(.{
        .root_source_file = b.path("native/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "kernels", .module = core }},
    }) });
    const compile_device = b.addSystemCommand(&.{ b.graph.zig_exe, "build-obj" });
    compile_device.addFileArg(b.path("native/device.zig"));
    compile_device.addArgs(&.{ "-target", "nvptx64-cuda", "-mcpu", "sm_75", "-OReleaseFast", "-fno-emit-bin" });
    const ptx = compile_device.addPrefixedOutputFileArg("-femit-asm=", "device.ptx");
    cli.root_module.addAnonymousImport("device_ptx", .{ .root_source_file = ptx });
    cli.root_module.link_libc = true;
    b.installArtifact(cli);
    const tests = b.addTest(.{ .root_module = core });
    b.step("test", "Run native kernel tests").dependOn(&b.addRunArtifact(tests).step);
}
