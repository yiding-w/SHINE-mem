"""
Utility functions for loradict manipulation.

A loradict is a nested dictionary structure used to store LoRA parameters.
At the leaf level, it is either None or {"A": Tensor, "B": Tensor, "C": Tensor | None}.
At higher levels, it is a dict mapping keys (int or str) to sub-loradicts.
"""

from typing import Dict, List, Optional
import torch
from torch import Tensor


def _apply_to_loradict_tensors(loradict, fn):
    """Recursively apply fn to every tensor in the loradict."""
    if loradict is None:
        return
    if isinstance(loradict, dict):
        for key, value in loradict.items():
            if isinstance(value, Tensor):
                fn(value)
            elif isinstance(value, dict):
                _apply_to_loradict_tensors(value, fn)


def collect_loradict_tensors(loradict) -> List[Tensor]:
    """Recursively collect all tensors from the loradict into a flat list.

    Args:
        loradict: A nested loradict structure. Can be None.

    Returns:
        A list of all tensors found in the loradict.
    """
    tensors = []
    if loradict is None:
        return tensors
    if isinstance(loradict, dict):
        for key, value in loradict.items():
            if isinstance(value, Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                tensors.extend(collect_loradict_tensors(value))
    return tensors


def freeze_loradict(loradict):
    """Freeze all tensors in the loradict by setting requires_grad to False.

    Args:
        loradict: A nested loradict structure. Can be None.
    """
    _apply_to_loradict_tensors(loradict, lambda t: t.requires_grad_(False))


def unfreeze_loradict(loradict):
    """Unfreeze all tensors in the loradict by setting requires_grad to True.

    Args:
        loradict: A nested loradict structure. Can be None.
    """
    _apply_to_loradict_tensors(loradict, lambda t: t.requires_grad_(True))


def _concat_leaf_loradicts(leaves: List[dict]) -> dict:
    """Concatenate a list of leaf loradicts along the rank dimension.

    Each leaf is {"A": [Lb, in, r_i], "B": [Lb, r_i, out], "C": [Lb, out] | None}.
    Result: {"A": [Lb, in, sum(r_i)], "B": [Lb, sum(r_i), out], "C": summed or None}.

    If batch dimensions differ (e.g., one leaf has Lb=1 and another has Lb=B),
    the Lb=1 tensors are expanded to match the maximum batch size before concat.
    """
    A_list = [leaf["A"] for leaf in leaves]
    B_list = [leaf["B"] for leaf in leaves]

    # Handle mismatched batch dimensions: expand Lb=1 to max Lb
    max_Lb = max(A.shape[0] for A in A_list)
    if max_Lb > 1:
        A_list = [A.expand(max_Lb, -1, -1) if A.shape[0] == 1 else A for A in A_list]
        B_list = [B.expand(max_Lb, -1, -1) if B.shape[0] == 1 else B for B in B_list]

    A_cat = torch.cat(A_list, dim=2)  # concat along r dimension
    B_cat = torch.cat(B_list, dim=1)  # concat along r dimension

    # For bias C: since the effective output is sum of (x @ Ai @ Bi + Ci),
    # the concatenated lora's bias is the sum of all individual biases.
    C_list = [leaf["C"] for leaf in leaves]
    if all(c is None for c in C_list):
        C_cat = None
    else:
        # Treat None as zero; expand Lb=1 to max Lb if needed
        non_none = [c for c in C_list if c is not None]
        if max_Lb > 1:
            non_none = [c.expand(max_Lb, -1) if c.shape[0] == 1 else c for c in non_none]
        C_cat = sum(non_none)

    return {"A": A_cat, "B": B_cat, "C": C_cat}


def _concat_loradict_recursive(loradicts: List[dict]) -> dict:
    """Recursively concatenate a list of loradicts.

    At each level, if the values are leaf dicts (containing "A", "B" keys),
    concatenate them. Otherwise, recurse into sub-dicts.

    All loradicts must have exactly the same keys at each level; raises
    ValueError if any mismatch is detected.
    """
    # Use the first loradict as reference for keys
    reference = loradicts[0]

    # Check if this is a leaf loradict (has "A" and "B" keys)
    if "A" in reference and "B" in reference:
        return _concat_leaf_loradicts(loradicts)

    # Verify all loradicts have the same keys
    ref_keys = set(reference.keys())
    for i, ld in enumerate(loradicts[1:], start=1):
        ld_keys = set(ld.keys())
        if ld_keys != ref_keys:
            raise ValueError(
                f"concat_loradict: key mismatch at current level. "
                f"loradict[0] has keys {sorted(ref_keys)}, "
                f"but loradict[{i}] has keys {sorted(ld_keys)}."
            )

    # Recurse into sub-keys
    result = {}
    for key in reference:
        sub_values = [ld[key] for ld in loradicts]
        # Filter out None sub-values
        non_none_subs = [v for v in sub_values if v is not None]
        if len(non_none_subs) == 0:
            result[key] = None
        elif len(non_none_subs) == 1:
            result[key] = non_none_subs[0]
        else:
            result[key] = _concat_loradict_recursive(non_none_subs)
    return result


def concat_loradict(loradict_list: List[Optional[dict]]) -> Optional[dict]:
    """Concatenate a list of loradicts along the rank dimension.

    For leaf-level loradicts with shapes A: [Lb, in, r_i], B: [Lb, r_i, out],
    the result has rank r1 + r2 + ... + rN by concatenating A along dim 2
    and B along dim 1. Bias C (if present) is summed.

    None entries in the list are ignored. If all entries are None, returns None.
    If only one non-None entry exists, returns it directly.

    Args:
        loradict_list: A list of loradicts (or None values).

    Returns:
        The concatenated loradict, or None if all inputs are None.
    """
    # Filter out None entries
    non_none = [ld for ld in loradict_list if ld is not None]

    if len(non_none) == 0:
        return None
    if len(non_none) == 1:
        return non_none[0]

    return _concat_loradict_recursive(non_none)


def check_nograd_loradict(nograd_loradict, path: str = "") -> List[str]:
    """Recursively check that all tensors in nograd_loradict have no gradient.

    Checks both requires_grad=False AND grad_fn is None to catch tensors
    that are still connected to the computation graph (e.g. not properly detached).

    Args:
        nograd_loradict: A nested loradict structure. Can be None.
        path: Internal use for building the key path in error messages.

    Returns:
        A list of error strings. Empty list means all checks passed.
    """
    errors = []
    if nograd_loradict is None:
        return errors
    if isinstance(nograd_loradict, dict):
        for key, value in nograd_loradict.items():
            current_path = f"{path}.{key}" if path else str(key)
            if isinstance(value, Tensor):
                if value.requires_grad:
                    errors.append(
                        f"  [{current_path}] requires_grad=True, "
                        f"shape={list(value.shape)}, grad_fn={value.grad_fn}"
                    )
                elif value.grad_fn is not None:
                    errors.append(
                        f"  [{current_path}] requires_grad=False but grad_fn={value.grad_fn}, "
                        f"shape={list(value.shape)} — tensor not properly detached!"
                    )
            elif isinstance(value, dict):
                errors.extend(check_nograd_loradict(value, current_path))
    return errors


# ===========================================================================
# wdict utility functions
#
# A wdict is a nested dictionary structure used to store full-rank weight
# matrices (the result of A@B). At the leaf level, it is either None or
# {"W": Tensor[Lb, in, out], "C": Tensor[Lb, out] | None}.
# At higher levels, it is a dict mapping keys (int or str) to sub-wdicts.
# ===========================================================================


def _loradict_leaf_to_wdict_leaf(leaf: dict) -> dict:
    """Convert a single loradict leaf {"A": [Lb, in, r], "B": [Lb, r, out], "C": ...}
    to a wdict leaf {"W": [Lb, in, out], "C": ...} by computing A @ B."""
    A = leaf["A"]  # [Lb, in, r]
    B = leaf["B"]  # [Lb, r, out]
    C = leaf.get("C", None)
    W = torch.bmm(A, B)  # [Lb, in, out]
    return {"W": W, "C": C}


def _is_loradict_leaf(d: dict) -> bool:
    """Check if a dict is a loradict leaf (has 'A' and 'B' keys)."""
    return "A" in d and "B" in d


def _is_wdict_leaf(d: dict) -> bool:
    """Check if a dict is a wdict leaf (has 'W' key)."""
    return "W" in d


def loradict_to_wdict(loradict) -> Optional[dict]:
    """Convert a loradict to a wdict by computing A @ B at every leaf.

    Recursively traverses the loradict structure. At each leaf
    {"A": [Lb, in, r], "B": [Lb, r, out], "C": [Lb, out] | None},
    computes W = A @ B and returns {"W": [Lb, in, out], "C": ...}.

    Args:
        loradict: A nested loradict structure. Can be None.

    Returns:
        A wdict with the same nested structure, or None if input is None.
    """
    if loradict is None:
        return None
    if _is_loradict_leaf(loradict):
        return _loradict_leaf_to_wdict_leaf(loradict)
    # Recurse into sub-dicts
    result = {}
    for key, value in loradict.items():
        if value is None:
            result[key] = None
        elif isinstance(value, dict):
            result[key] = loradict_to_wdict(value)
        else:
            result[key] = value
    return result


def add_wdict(wdict_list: List[Optional[dict]]) -> Optional[dict]:
    """Element-wise add multiple wdicts together.

    At each leaf, sums the W tensors and C tensors (if present).
    None entries in the list are ignored. If all entries are None, returns None.

    Args:
        wdict_list: A list of wdicts (or None values) to add together.

    Returns:
        The summed wdict, or None if all inputs are None.
    """
    non_none = [w for w in wdict_list if w is not None]
    if len(non_none) == 0:
        return None
    if len(non_none) == 1:
        return non_none[0]
    return _add_wdict_recursive(non_none)


def _add_wdict_recursive(wdicts: List[dict]) -> dict:
    """Recursively add a list of wdicts element-wise."""
    reference = wdicts[0]

    # Check if this is a wdict leaf
    if _is_wdict_leaf(reference):
        W_sum = sum(w["W"] for w in wdicts)
        C_list = [w.get("C", None) for w in wdicts]
        if all(c is None for c in C_list):
            C_sum = None
        else:
            C_sum = sum(c for c in C_list if c is not None)
        return {"W": W_sum, "C": C_sum}

    # Recurse into sub-keys
    result = {}
    for key in reference:
        sub_values = [w[key] for w in wdicts]
        non_none_subs = [v for v in sub_values if v is not None]
        if len(non_none_subs) == 0:
            result[key] = None
        elif len(non_none_subs) == 1:
            result[key] = non_none_subs[0]
        else:
            result[key] = _add_wdict_recursive(non_none_subs)
    return result


def detach_loradict(loradict) -> Optional[dict]:
    """Recursively detach and clone all tensors in a loradict.

    Ensures the returned loradict is completely disconnected from any
    computation graph. All tensors will have requires_grad=False.

    Args:
        loradict: A nested loradict structure. Can be None.

    Returns:
        A new loradict with all tensors detached and cloned, or None if input is None.
    """
    if loradict is None:
        return None
    if isinstance(loradict, dict):
        result = {}
        for key, value in loradict.items():
            if isinstance(value, Tensor):
                result[key] = value.detach().clone()
            elif isinstance(value, dict):
                result[key] = detach_loradict(value)
            else:
                result[key] = value
        return result
    return loradict


def detach_wdict(wdict) -> Optional[dict]:
    """Recursively detach and clone all tensors in a wdict.

    Ensures the returned wdict is completely disconnected from any
    computation graph. All tensors will have requires_grad=False.

    Args:
        wdict: A nested wdict structure. Can be None.

    Returns:
        A new wdict with all tensors detached and cloned, or None if input is None.
    """
    if wdict is None:
        return None
    if _is_wdict_leaf(wdict):
        W = wdict["W"].detach().clone()
        C = wdict.get("C", None)
        if C is not None:
            C = C.detach().clone()
        return {"W": W, "C": C}
    # Recurse into sub-dicts
    result = {}
    for key, value in wdict.items():
        if value is None:
            result[key] = None
        elif isinstance(value, dict):
            result[key] = detach_wdict(value)
        else:
            result[key] = value
    return result


def check_nograd_wdict(wdict, path: str = "") -> List[str]:
    """Recursively check that all tensors in a wdict have no gradient.

    Checks both requires_grad=False AND grad_fn is None to catch tensors
    that are still connected to the computation graph.

    Args:
        wdict: A nested wdict structure. Can be None.
        path: Internal use for building the key path in error messages.

    Returns:
        A list of error strings. Empty list means all checks passed.
    """
    errors = []
    if wdict is None:
        return errors
    if isinstance(wdict, dict):
        for key, value in wdict.items():
            current_path = f"{path}.{key}" if path else str(key)
            if isinstance(value, Tensor):
                if value.requires_grad:
                    errors.append(
                        f"  [{current_path}] requires_grad=True, "
                        f"shape={list(value.shape)}, grad_fn={value.grad_fn}"
                    )
                elif value.grad_fn is not None:
                    errors.append(
                        f"  [{current_path}] requires_grad=False but grad_fn={value.grad_fn}, "
                        f"shape={list(value.shape)} — tensor not properly detached!"
                    )
            elif isinstance(value, dict):
                errors.extend(check_nograd_wdict(value, current_path))
    return errors


def collect_wdict_tensors(wdict) -> List[Tensor]:
    """Recursively collect all tensors from a wdict into a flat list.

    Args:
        wdict: A nested wdict structure. Can be None.

    Returns:
        A list of all tensors found in the wdict.
    """
    tensors = []
    if wdict is None:
        return tensors
    if isinstance(wdict, dict):
        for key, value in wdict.items():
            if isinstance(value, Tensor):
                tensors.append(value)
            elif isinstance(value, dict):
                tensors.extend(collect_wdict_tensors(value))
    return tensors


def zeros_wdict_like(reference, device=None, dtype=None) -> Optional[dict]:
    """Create a zero-initialized wdict with the same structure as a reference.

    The reference can be either a wdict or a loradict. If it's a loradict,
    the output shape is inferred from A and B dimensions.

    Args:
        reference: A nested wdict or loradict structure. Can be None.
        device: Device for the created tensors. If None, uses the reference's device.
        dtype: Dtype for the created tensors. If None, uses the reference's dtype.

    Returns:
        A zero-initialized wdict, or None if reference is None.
    """
    if reference is None:
        return None
    if _is_wdict_leaf(reference):
        W = reference["W"]
        _device = device or W.device
        _dtype = dtype or W.dtype
        W_zero = torch.zeros_like(W, device=_device, dtype=_dtype)
        C = reference.get("C", None)
        C_zero = None
        if C is not None:
            C_zero = torch.zeros_like(C, device=_device, dtype=_dtype)
        return {"W": W_zero, "C": C_zero}
    if _is_loradict_leaf(reference):
        A = reference["A"]  # [Lb, in, r]
        B = reference["B"]  # [Lb, r, out]
        Lb, in_dim, _ = A.shape
        _, _, out_dim = B.shape
        _device = device or A.device
        _dtype = dtype or A.dtype
        W_zero = torch.zeros(Lb, in_dim, out_dim, device=_device, dtype=_dtype)
        C = reference.get("C", None)
        C_zero = None
        if C is not None:
            C_zero = torch.zeros(Lb, out_dim, device=_device, dtype=_dtype)
        return {"W": W_zero, "C": C_zero}
    # Recurse into sub-dicts
    result = {}
    for key, value in reference.items():
        if value is None:
            result[key] = None
        elif isinstance(value, dict):
            result[key] = zeros_wdict_like(value, device=device, dtype=dtype)
        else:
            result[key] = value
    return result


def merge_loradicts_into_wdict_add(
    loradict_list: List[Optional[dict]],
    wdict: Optional[dict] = None,
) -> Optional[dict]:
    """Merge multiple loradicts into a wdict by accumulating A@B one by one.

    For each loradict, computes A@B and adds it to the running wdict.
    This approach uses less peak memory (no concat of A/B needed), but
    may have lower GPU utilization when individual ranks are small.

    If wdict is None, a zero wdict is created from the first non-None loradict.

    Args:
        loradict_list: A list of loradicts to merge. None entries are skipped.
        wdict: An existing wdict to accumulate into. If None, creates a new one.

    Returns:
        The accumulated wdict, or None if all inputs are None and wdict is None.
    """
    non_none = [ld for ld in loradict_list if ld is not None]
    if len(non_none) == 0:
        return wdict

    # Initialize wdict if needed
    if wdict is None:
        wdict = zeros_wdict_like(non_none[0])

    # Accumulate each loradict's A@B into wdict
    for loradict in non_none:
        _accumulate_loradict_into_wdict(loradict, wdict)

    return wdict


def _accumulate_loradict_into_wdict(loradict: dict, wdict: dict):
    """In-place accumulate a single loradict's A@B into a wdict."""
    if _is_loradict_leaf(loradict) and _is_wdict_leaf(wdict):
        A = loradict["A"]  # [Lb, in, r]
        B = loradict["B"]  # [Lb, r, out]
        C = loradict.get("C", None)
        wdict["W"].add_(torch.bmm(A, B))  # In-place add
        if C is not None:
            if wdict["C"] is None:
                wdict["C"] = C.clone()
            else:
                wdict["C"].add_(C)
        return

    # Recurse into sub-dicts
    for key in loradict:
        if loradict[key] is None:
            continue
        if key not in wdict or wdict[key] is None:
            # Create zero leaf from loradict leaf
            wdict[key] = zeros_wdict_like(loradict[key])
        _accumulate_loradict_into_wdict(loradict[key], wdict[key])


def merge_loradicts_into_wdict_concat(
    loradict_list: List[Optional[dict]],
    wdict: Optional[dict] = None,
) -> Optional[dict]:
    """Merge multiple loradicts into a wdict by concatenating then multiplying.

    First concatenates all loradicts along the rank dimension (A along dim 2,
    B along dim 1), then performs a single large A_big @ B_big matmul.
    This approach has higher GPU utilization for many small-rank loradicts,
    but uses more peak memory (needs to allocate concatenated A and B).

    If wdict is None, a zero wdict is created from the concatenated result.

    Args:
        loradict_list: A list of loradicts to merge. None entries are skipped.
        wdict: An existing wdict to accumulate into. If None, creates a new one.

    Returns:
        The accumulated wdict, or None if all inputs are None and wdict is None.
    """
    non_none = [ld for ld in loradict_list if ld is not None]
    if len(non_none) == 0:
        return wdict

    # Concat all loradicts along rank dimension, then convert to wdict
    merged_loradict = concat_loradict(non_none)
    new_wdict = loradict_to_wdict(merged_loradict)

    if wdict is None:
        return new_wdict

    # Add the new wdict to the existing one
    return add_wdict([wdict, new_wdict])
