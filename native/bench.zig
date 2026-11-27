//! Time standalone FP32 CPU and CUDA kernels with deterministic inputs and monotonic timestamps.
const std = @import("std");
const builtin = @import("builtin");
const kernels = @import("kernels");

const cuda = @import("cuda.zig");
const Case = cuda.Case;

fn cpu(case: Case, inputs: [3][]f32, out: []f32, scratch: []f32, n: usize, d: usize) void {
    switch (case) {
        .attention => kernels.attention(f32, inputs[0].ptr, inputs[1].ptr, inputs[2].ptr, out.ptr, scratch, 1, n, d),
        .rms_norm => kernels.rmsNorm(f32, inputs[0].ptr, inputs[1].ptr, inputs[2].ptr, out.ptr, n, d, 1e-6),
    }
}

/// Benchmark cases reuse tensors, output, and one attention scratch row.
pub fn main(init: std.process.Init) !void {
    const alloc = init.arena.allocator();
    const args = try init.minimal.args.toSlice(alloc);
    var use_cuda = false;
    var n: usize = 129;
    var d: usize = 64;
    var repeats: usize = 10;
    var buffer: [4096]u8 = undefined;
    var file = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    const writer = &file.interface;
    var index: usize = 1;
    while (index < args.len) : (index += 2) {
        const option = args[index];
        if (std.mem.eql(u8, option, "--emit-ptx")) {
            try writer.writeAll(cuda.ptx);
            return writer.flush();
        }
        if (std.mem.eql(u8, option, "--help")) {
            try writer.writeAll("kernel-bench-zig [--device cpu|cuda] [--tokens N] [--depth D] [--repeats R]\n");
            return writer.flush();
        }
        if (index + 1 >= args.len) return error.MissingValue;
        if (std.mem.eql(u8, option, "--device")) {
            const name = args[index + 1];
            use_cuda = std.mem.eql(u8, name, "cuda");
            if (!use_cuda and !std.mem.eql(u8, name, "cpu")) return error.UnknownDevice;
            continue;
        }
        const value = try std.fmt.parseInt(usize, args[index + 1], 10);
        if (value == 0) return error.ExpectedPositiveValue;
        if (std.mem.eql(u8, option, "--tokens")) n = value else if (std.mem.eql(u8, option, "--depth")) d = value else if (std.mem.eql(u8, option, "--repeats")) repeats = value else return error.UnknownOption;
    }
    if (use_cuda and (d > 128 or n > std.math.maxInt(i32))) return error.UnsupportedGpuShape;
    const count = try std.math.mul(usize, n, d);
    var inputs: [3][]f32 = undefined;
    var random = std.Random.DefaultPrng.init(872);
    for (&inputs) |*input| {
        input.* = try alloc.alloc(f32, count);
        for (input.*) |*x| x.* = 2 * random.random().float(f32) - 1;
    }
    var gpu: ?cuda.Run = if (use_cuda) try cuda.Run.init(inputs) else null;
    defer if (gpu) |*run| run.deinit();
    const out = try alloc.alloc(f32, count);
    const scratch = try alloc.alloc(f32, d);
    const samples = try alloc.alloc(u64, repeats);
    for ([_]Case{ .attention, .rms_norm }) |case| {
        for (0..try std.math.add(usize, repeats, 5)) |trial| {
            const start = std.Io.Clock.awake.now(init.io);
            if (gpu) |*run| {
                try run.launch(case, @intCast(n), @intCast(d));
            } else {
                cpu(case, inputs, out, scratch, n, d);
            }
            std.mem.doNotOptimizeAway(out.ptr);
            const elapsed = start.durationTo(std.Io.Clock.awake.now(init.io)).toNanoseconds();
            if (trial >= 5) samples[trial - 5] = @intCast(elapsed);
        }
        if (gpu) |*run| {
            const result = try alloc.alloc(f32, count);
            try run.copyOutput(result);
            cpu(case, inputs, out, scratch, n, d);
            for (result, out) |actual, expected| {
                if (!std.math.isFinite(actual) or @abs(actual - expected) > 2e-4 * (1 + @abs(expected))) return error.CudaParityFailed;
            }
        }
        const device = if (use_cuda) "cuda" else "cpu";
        const hardware = if (gpu) |*run| std.mem.sliceTo(&run.name, 0) else @tagName(builtin.cpu.arch);
        const work = @as(u128, count) * (if (case == .attention) 2 * (@as(u128, n) + 1) else 5);
        const bytes = if (case == .attention) 16 * @as(u128, count) else 4 * (3 * @as(u128, count) + d);
        try writer.print("{{\"kernel\":\"{s}\",\"backend\":\"zig\",\"device\":\"{s}\",\"hardware\":{f},\"dtype\":\"float32\",\"tokens\":{d},\"depth\":{d},\"modeled_flops\":{d},\"minimum_bytes\":{d},\"latency_ns\":{f}}}\n", .{ @tagName(case), device, std.json.fmt(hardware, .{}), n, d, work, bytes, std.json.fmt(samples, .{}) });
    }
    try writer.flush();
}
