//! Forward CPU kernels with FP32 accumulators and a contiguous tensor ABI.
const std = @import("std");

/// BF16 storage uses the upper 16 bits of an IEEE binary32 value.
const BF16 = packed struct(u16) { bits: u16 };

fn load(comptime T: type, x: T) f32 {
    if (T == BF16) return @bitCast(@as(u32, x.bits) << 16);
    return @floatCast(x);
}

fn store(comptime T: type, x: f32) T {
    if (T != BF16) return @floatCast(x);
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return .{ .bits = @as(u16, @truncate(bits >> 16)) | 0x40 };
    return .{ .bits = @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16) };
}

/// Compute causal attention without allocating a quadratic score matrix.
/// Buffers contain groups*tokens*depth elements. Scratch holds depth FP32 values.
pub fn attention(comptime T: type, q: [*]const T, k: [*]const T, v: [*]const T, out: [*]T, scratch: []f32, groups: usize, tokens: usize, depth: usize) void {
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(depth)));
    for (0..groups) |group| {
        const base = group * tokens * depth;
        for (0..tokens) |row| {
            @memset(scratch, 0);
            var maximum = -std.math.inf(f32);
            var mass: f32 = 0;
            const query = q[base + row * depth ..][0..depth];
            for (0..row + 1) |col| {
                const offset = base + col * depth;
                var dot: f32 = 0;
                for (query, k[offset..][0..depth]) |a, b| dot += load(T, a) * load(T, b);
                const score = dot * scale;
                const next = @max(maximum, score);
                const prior = @exp(maximum - next);
                const weight = @exp(score - next);
                mass = mass * prior + weight;
                for (scratch, v[offset..][0..depth]) |*acc, value| acc.* = acc.* * prior + weight * load(T, value);
                maximum = next;
            }
            for (scratch, out[base + row * depth ..][0..depth]) |sum, *dest| dest.* = store(T, sum / mass);
        }
    }
}

/// Fuse residual addition and row normalization without allocating an activation tensor.
pub fn rmsNorm(comptime T: type, x: [*]const T, residual: [*]const T, weight: [*]const T, out: [*]T, rows: usize, depth: usize, eps: f32) void {
    for (0..rows) |row| {
        const offset = row * depth;
        var sum: f32 = 0;
        for (x[offset..][0..depth], residual[offset..][0..depth]) |a, b| {
            const total = load(T, a) + load(T, b);
            sum += total * total;
        }
        const inv = 1.0 / @sqrt(sum / @as(f32, @floatFromInt(depth)) + eps);
        for (0..depth) |i| out[offset + i] = store(T, (load(T, x[offset + i]) + load(T, residual[offset + i])) * inv * load(T, weight[i]));
    }
}

fn fits(a: usize, b: usize) bool {
    return a != 0 and b != 0 and a <= std.math.maxInt(usize) / b;
}

/// Dispatch 0=FP32, 1=FP16, 2=BF16. Caller owns aligned, non-overlapping output and scratch.
export fn ko_attention(dtype: u32, q: *const anyopaque, k: *const anyopaque, v: *const anyopaque, out: *anyopaque, scratch: [*]f32, groups: usize, tokens: usize, depth: usize) c_int {
    if (!fits(groups, tokens) or !fits(groups * tokens, depth)) return 1;
    inline for (.{ f32, f16, BF16 }, 0..) |T, id| {
        if (dtype == id) {
            attention(T, @ptrCast(@alignCast(q)), @ptrCast(@alignCast(k)), @ptrCast(@alignCast(v)), @ptrCast(@alignCast(out)), scratch[0..depth], groups, tokens, depth);
            return 0;
        }
    }
    return 1;
}

/// Normalize compatible contiguous buffers. Return one for an unsupported format or dimension.
export fn ko_rms_norm(dtype: u32, x: *const anyopaque, r: *const anyopaque, w: *const anyopaque, out: *anyopaque, rows: usize, depth: usize, eps: f32) c_int {
    if (!fits(rows, depth) or !std.math.isFinite(eps) or eps <= 0) return 1;
    inline for (.{ f32, f16, BF16 }, 0..) |T, id| {
        if (dtype == id) {
            rmsNorm(T, @ptrCast(@alignCast(x)), @ptrCast(@alignCast(r)), @ptrCast(@alignCast(w)), @ptrCast(@alignCast(out)), rows, depth, eps);
            return 0;
        }
    }
    return 1;
}

test "online softmax gives uniform causal averages for equal dot products" {
    const q = [_]f32{0} ** 6;
    const v = [_]f32{ 2, 4, 4, 8, 6, 12 };
    var out: [6]f32 = undefined;
    var scratch: [2]f32 = undefined;
    attention(f32, &q, &q, &v, &out, &scratch, 1, 3, 2);
    try std.testing.expectEqualSlices(f32, &.{ 2, 4, 3, 6, 4, 8 }, &out);
}

test "residual normalization preserves finite zeros in all storage types" {
    inline for (.{ f32, f16, BF16 }) |T| {
        const x = [_]T{store(T, 1)} ** 3;
        const r = [_]T{store(T, -1)} ** 3;
        var out: [3]T = undefined;
        rmsNorm(T, &x, &r, &x, &out, 1, 3, 0.000001);
        for (out) |y| try std.testing.expectEqual(@as(f32, 0), load(T, y));
    }
}
