"""
Minimal stub for CA1_CorrelationGate.
The consolidation experiment does not use this module,
but fact_learning_paradigm.py imports it at the top level.
"""

import torch

class CA1_CorrelationGate:
    """Simple correlation-based gating for CA1 output filtering."""
    def __init__(self, d_ec, device='cpu', dtype=torch.float32):
        self.d_ec = d_ec
        self.device = device
        self.dtype = dtype

    def filter(self, retrieved_ec, query_ec):
        """Pass-through filter: return retrieved_ec weighted by correlation."""
        sim = torch.dot(
            retrieved_ec / (torch.linalg.norm(retrieved_ec) + 1e-10),
            query_ec / (torch.linalg.norm(query_ec) + 1e-10)
        )
        gate = torch.clamp(sim, 0.0, 1.0)
        return retrieved_ec * float(gate)
