//! CUDA device code. One warp owns each row of the FP32 forward kernels.
fn laneId() u32 {
    return asm volatile ("mov.u32 %[result], %laneid;"
        : [result] "=r" (-> u32),
    );
}

fn blockId() u32 {
    return asm volatile ("mov.u32 %[result], %ctaid.x;"
        : [result] "=r" (-> u32),
    );
}

fn warpSum(value: f32) f32 {
    var sum = value;
    inline for (.{ 16, 8, 4, 2, 1 }) |offset| {
        sum += asm volatile ("shfl.sync.bfly.b32 %[result], %[value], %[offset], 31, 0xffffffff;"
            : [result] "=f" (-> f32),
            : [value] "f" (sum),
              [offset] "r" (@as(u32, offset)),
        );
    }
    return sum;
}

fn invSqrt(value: f32) f32 {
    return asm ("rsqrt.approx.f32 %[r], %[v];"
        : [r] "=f" (-> f32),
        : [v] "f" (value),
    );
}

fn exponential(value: f32) f32 {
    return asm ("ex2.approx.ftz.f32 %[r], %[v];"
        : [r] "=f" (-> f32),
        : [v] "f" (value * 1.4426950408889634),
    );
}

/// Each block handles one causal query. The host checks depth <= 128.
export fn attention_f32(q: [*]const f32, k: [*]const f32, v: [*]const f32, out: [*]f32, tokens: u32, depth: u32) callconv(.nvptx_device) void {
    const row = blockId();
    const lane = laneId();
    const local = row % tokens;
    const group_base = @as(usize, row - local) * depth;
    const query_base = @as(usize, row) * depth;
    var acc = [_]f32{0} ** 4;
    var maximum: f32 = -@import("std").math.inf(f32);
    var mass: f32 = 0;
    var col: u32 = 0;
    while (col <= local) : (col += 1) {
        const key_base = group_base + @as(usize, col) * depth;
        var dot: f32 = 0;
        inline for (0..4) |part| {
            const channel = lane + part * 32;
            if (channel < depth) dot += q[query_base + channel] * k[key_base + channel];
        }
        const score = warpSum(dot) * invSqrt(@floatFromInt(depth));
        const next = @max(maximum, score);
        const prior = exponential(maximum - next);
        const weight = exponential(score - next);
        mass = mass * prior + weight;
        inline for (0..4) |part| {
            const channel = lane + part * 32;
            if (channel < depth) acc[part] = acc[part] * prior + weight * v[key_base + channel];
        }
        maximum = next;
    }
    inline for (0..4) |part| {
        const channel = lane + part * 32;
        if (channel < depth) out[query_base + channel] = acc[part] / mass;
    }
}

/// Each warp computes one second moment and normalizes its activation row.
export fn rms_f32(x: [*]const f32, residual: [*]const f32, w: [*]const f32, out: [*]f32, depth: u32, eps: f32) callconv(.nvptx_device) void {
    const offset = @as(usize, blockId()) * depth;
    const lane = laneId();
    var sum: f32 = 0;
    var channel: usize = lane;
    while (channel < depth) : (channel += 32) {
        const total = x[offset + channel] + residual[offset + channel];
        sum += total * total;
    }
    const inv = invSqrt(warpSum(sum) / @as(f32, @floatFromInt(depth)) + eps);
    channel = lane;
    while (channel < depth) : (channel += 32) {
        out[offset + channel] = (x[offset + channel] + residual[offset + channel]) * inv * w[channel];
    }
}
