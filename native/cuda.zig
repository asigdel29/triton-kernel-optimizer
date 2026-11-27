//! CUDA driver client for the standalone benchmark. Resources live within one context.
const std = @import("std");
const builtin = @import("builtin");
pub const ptx = @embedFile("device_ptx").* ++ "\n" ++ @embedFile("entry.ptx").*;
pub const Case = enum { attention, rms_norm };
const Handle = ?*anyopaque;
const DevicePtr = u64;
const Status = c_int;

const Driver = struct {
    cuInit: *const fn (c_uint) callconv(.c) Status,
    cuDeviceGet: *const fn (*c_int, c_int) callconv(.c) Status,
    cuDeviceGetName: *const fn ([*]u8, c_int, c_int) callconv(.c) Status,
    cuCtxCreate_v2: *const fn (*Handle, c_uint, c_int) callconv(.c) Status,
    cuCtxDestroy_v2: *const fn (Handle) callconv(.c) Status,
    cuModuleLoadData: *const fn (*Handle, [*:0]const u8) callconv(.c) Status,
    cuModuleGetFunction: *const fn (*Handle, Handle, [*:0]const u8) callconv(.c) Status,
    cuMemAlloc_v2: *const fn (*DevicePtr, usize) callconv(.c) Status,
    cuMemcpyHtoD_v2: *const fn (DevicePtr, *const anyopaque, usize) callconv(.c) Status,
    cuMemcpyDtoH_v2: *const fn (*anyopaque, DevicePtr, usize) callconv(.c) Status,
    cuLaunchKernel: *const fn (Handle, u32, u32, u32, u32, u32, u32, u32, Handle, [*]const *const anyopaque, Handle) callconv(.c) Status,
    cuCtxSynchronize: *const fn () callconv(.c) Status,
};

fn check(status: Status) !void {
    if (status != 0) {
        std.log.err("CUDA driver returned status {d}", .{status});
        return error.CudaDriverError;
    }
}

/// Own the context, module and bounded-size benchmark buffers until deinit.
pub const Run = struct {
    library: std.DynLib,
    driver: Driver,
    context: Handle,
    attention: Handle,
    rms: Handle,
    buffers: [4]DevicePtr,
    name: [256]u8,

    pub fn init(inputs: [3][]f32) !Run {
        if (builtin.os.tag != .linux) return error.LinuxCudaDriverRequired;
        var run: Run = undefined;
        run.library = std.DynLib.open("libcuda.so.1") catch return error.LinuxCudaDriverRequired;
        errdefer run.library.close();
        inline for (std.meta.fields(Driver)) |field| {
            @field(run.driver, field.name) = run.library.lookup(field.type, field.name) orelse return error.MissingDriverSymbol;
        }
        try check(run.driver.cuInit(0));
        var device: c_int = undefined;
        try check(run.driver.cuDeviceGet(&device, 0));
        try check(run.driver.cuDeviceGetName(&run.name, run.name.len, device));
        try check(run.driver.cuCtxCreate_v2(&run.context, 0, device));
        errdefer _ = run.driver.cuCtxDestroy_v2(run.context);
        var module: Handle = null;
        try check(run.driver.cuModuleLoadData(&module, ptx));
        try check(run.driver.cuModuleGetFunction(&run.attention, module, "ko_attention"));
        try check(run.driver.cuModuleGetFunction(&run.rms, module, "ko_rms"));
        const bytes = std.mem.sliceAsBytes(inputs[0]).len;
        for (&run.buffers) |*ptr| try check(run.driver.cuMemAlloc_v2(ptr, bytes));
        for (inputs, run.buffers[0..3]) |source, dest| try check(run.driver.cuMemcpyHtoD_v2(dest, source.ptr, bytes));
        return run;
    }

    /// Destroying the context releases its module and every device allocation.
    pub fn deinit(run: *Run) void {
        _ = run.driver.cuCtxDestroy_v2(run.context);
        run.library.close();
    }

    /// Record a synchronous launch on the default stream in the dedicated context.
    pub fn launch(run: *Run, case: Case, n: u32, d: u32) !void {
        const epsilon: f32 = 1e-6;
        const args = [_]*const anyopaque{ &run.buffers[0], &run.buffers[1], &run.buffers[2], &run.buffers[3], if (case == .attention) &n else &d, if (case == .attention) @ptrCast(&d) else @ptrCast(&epsilon) };
        const function = if (case == .attention) run.attention else run.rms;
        try check(run.driver.cuLaunchKernel(function, n, 1, 1, 32, 1, 1, 0, null, &args, null));
        try check(run.driver.cuCtxSynchronize());
    }

    /// Recover output once per case for comparison with the native CPU reference.
    pub fn copyOutput(run: *Run, dest: []f32) !void {
        try check(run.driver.cuMemcpyDtoH_v2(dest.ptr, run.buffers[3], std.mem.sliceAsBytes(dest).len));
    }
};
