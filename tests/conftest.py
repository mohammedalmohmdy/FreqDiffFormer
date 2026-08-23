"""Shared pytest configuration: reproducibility for the test suite.

These tests assert the manuscript-mandated configuration and behaviour. CPU-only
torch (2.2.x) is sufficient for all checks (shapes, losses, ablations, stats).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
torch.manual_seed(0)
