"""Thin re-exports from caliperlib.ubnext (UB-X).

All allocator, tensor, and collective implementations live in the caliperlib
package (ub-x).  This module re-exports them under the names that TE's linear
and layernorm_linear modules expect.
"""

from caliperlib.ubnext import SymmAllocator, SymmTensor
from caliperlib.ubnext.ops import (
    request_allocator as ubx_request_allocator,
    get_sym_tensor as ubx_get_sym_tensor,
    allreduce as ubx_allreduce,
    free_residual as ubx_free_residual,
    restore as ubx_restore,
    mem_stats as ubx_mem_stats,
)

__all__ = [
    "SymmAllocator",
    "SymmTensor",
    "ubx_request_allocator",
    "ubx_get_sym_tensor",
    "ubx_allreduce",
    "ubx_free_residual",
    "ubx_restore",
    "ubx_mem_stats",
]
