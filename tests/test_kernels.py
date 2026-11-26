"""Check each inference operation against independent high-precision equations."""
import importlib.util
import unittest

import torch

from kernel_optimizer import causal_attention, residual_rms_norm


class KernelTests(unittest.TestCase):
    """Exercise odd token counts, multiple heads, masks, precision, and validation."""

    def compare(self, device, backend, dtype, n, d):
        """Check one kernel pair with an independent FP64 CPU reference."""
        torch.manual_seed(872)
        q, k, v = [torch.randn(2, 2, n, d, device=device).to(dtype) for _ in range(3)]
        weight = torch.randn(d, device=device).to(dtype)
        a, b, c = [t.cpu().double() for t in (q, k, v)]
        scores = a @ b.transpose(-1, -2) / d**0.5
        mask = torch.ones(n, n, dtype=torch.bool).triu(1)
        expected = scores.masked_fill(mask, -torch.inf).softmax(-1) @ c
        tolerance = {torch.bfloat16: .025, torch.float16: .003, torch.float32: 2e-6}[dtype]
        actual = causal_attention(q, k, v, backend=backend)
        torch.testing.assert_close(actual.cpu().double(), expected, atol=tolerance, rtol=tolerance)
        total = a + b
        expected = total / (total.square().mean(-1, keepdim=True) + 1e-6).sqrt() * weight.cpu().double()
        actual = residual_rms_norm(q, k, weight, backend=backend)
        torch.testing.assert_close(actual.cpu().double(), expected, atol=tolerance, rtol=tolerance)

    def test_cpu_references(self):
        """Compare all supported CPU precisions with the reference equations."""
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            self.compare("cpu", "torch", dtype, 37, 16)

    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"), "requires CUDA and Triton")
    def test_triton(self):
        """Exercise partial tiles and all advertised head dimensions."""
        for dtype in (torch.float16, torch.bfloat16):
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for depth in (16, 32, 64, 128):
                self.compare("cuda", "triton", dtype, 67, depth)

    def test_zero_sum(self):
        """A cancelled residual must yield finite zeros."""
        x = torch.ones(2, 17)
        torch.testing.assert_close(residual_rms_norm(x, -x, torch.ones(17)), torch.zeros_like(x))

    def test_invalid_shapes(self):
        """Reject incompatible shapes and invalid normalization constants."""
        x = torch.ones(2, 17)
        with self.assertRaises(ValueError):
            causal_attention(x, x, x)
        with self.assertRaises(ValueError):
            residual_rms_norm(x, x, torch.ones(3))
        with self.assertRaises(ValueError):
            residual_rms_norm(x, x, torch.ones(17), eps=0)
