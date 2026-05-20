import xarray as xr
import numpy as np
import gc
try:
    from datatree import DataTree
except ImportError:
    try:
        from xarray import DataTree
    except ImportError:
        raise ImportError("DataTree not found. Install with: pip install datatree==0.0.12")
from typing import Optional, Callable, List, Dict


def filter_datatree_by_cost_ratio_memeff(
    dt: DataTree,
    threshold: float,
    fill_value=np.nan,
    remove_filtered: bool = False,
    node_filter: Optional[Callable[[DataTree], bool]] = None,
    skip_nodes_without_cost: bool = True,
    process_order: str = 'sequential',
    clear_after_each_node: bool = True,
    verbose: bool = True,
) -> DataTree:
    """
    Memory-efficient DataTree filtering for datatree 0.0.12.
    
    This version minimizes memory usage by:
    - Processing nodes one at a time
    - Forcing garbage collection between nodes
    - Avoiding full dataset copies
    - Providing progress feedback
    
    Parameters
    ----------
    dt : DataTree
        Input DataTree with nodes containing datasets.
    threshold : float
        Maximum allowed ratio of final to initial cost.
    fill_value : scalar, optional
        Value to use for filling filtered ensemble members.
    remove_filtered : bool, optional
        If True, remove filtered ensemble members from each node.
    node_filter : callable, optional
        Function that takes a DataTree node and returns True if filtering should
        be applied to that node.
    skip_nodes_without_cost : bool, optional
        If True (default), skip nodes that don't have 'cost_hist' variable.
    process_order : str, optional
        'sequential' (default): process nodes in tree order
        'memory_sorted': process smallest nodes first to avoid memory spikes
    clear_after_each_node : bool, optional
        If True (default), force garbage collection after each node.
    verbose : bool, optional
        If True (default), print progress messages.
    
    Returns
    -------
    DataTree
        Filtered DataTree.
    
    Examples
    --------
    >>> # Memory-efficient filtering with progress
    >>> dt_filtered = filter_datatree_by_cost_ratio_memeff(
    ...     dt, 
    ...     threshold=0.5,
    ...     verbose=True
    ... )
    
    >>> # Ultra-low memory: process smallest nodes first
    >>> dt_filtered = filter_datatree_by_cost_ratio_memeff(
    ...     dt,
    ...     threshold=0.5,
    ...     process_order='memory_sorted',
    ...     clear_after_each_node=True
    ... )
    """
    
    # Collect all nodes to process
    nodes_to_process = []
    _collect_nodes(dt, nodes_to_process)
    
    if verbose:
        print(f"Found {len(nodes_to_process)} nodes in tree")
    
    # Sort by memory if requested
    if process_order == 'memory_sorted':
        if verbose:
            print("Sorting nodes by memory usage (smallest first)...")
        nodes_to_process = _sort_nodes_by_memory(nodes_to_process)
    
    # Process each node
    processed_count = 0
    for node in nodes_to_process:
        # Check if we should process this node
        if node.ds is None or len(node.ds.data_vars) == 0:
            continue
        
        if node_filter is not None and not node_filter(node):
            continue
        
        if 'cost_hist' not in node.ds or 'ens_mem' not in node.ds.dims:
            if skip_nodes_without_cost:
                continue
            else:
                raise ValueError(f"Node '{node.path}' missing required variables")
        
        # Process this node
        processed_count += 1
        if verbose:
            print(f"Processing node {processed_count}: {node.path}")
            if hasattr(node.ds, 'nbytes'):
                mem_mb = node.ds.nbytes / (1024**2)
                print(f"  Memory: {mem_mb:.1f} MB")
        
        # Filter the dataset at this node
        filtered_ds = filter_by_cost_ratio_memeff(
            node.ds,
            threshold=threshold,
            fill_value=fill_value,
            remove_filtered=remove_filtered,
            in_place=False,  # Don't modify original
        )
        
        # Update the node's dataset
        filtered_ds.attrs['filtered_node_path'] = node.path
        
        # Replace the dataset at this node
        # In datatree 0.0.12, we need to create a new tree structure
        # This is handled by _rebuild_tree below
        
        if verbose:
            kept = filtered_ds.attrs.get('n_members_kept', 'N/A')
            filtered = filtered_ds.attrs.get('n_members_filtered', 'N/A')
            print(f"  Result: {kept} kept, {filtered} filtered")
        
        # Store the filtered dataset
        node._ds_filtered = filtered_ds
        
        # Clean up
        if clear_after_each_node:
            gc.collect()
    
    # Rebuild the tree with filtered datasets
    if verbose:
        print(f"\nRebuilding tree with {processed_count} filtered nodes...")
    
    dt_filtered = _rebuild_tree_v012(dt, nodes_to_process)
    
    # Final cleanup
    if clear_after_each_node:
        gc.collect()
    
    if verbose:
        print("✓ Filtering complete")
    
    return dt_filtered


def _collect_nodes(node: DataTree, node_list: List[DataTree]):
    """Collect all nodes in tree."""
    node_list.append(node)
    if hasattr(node, 'children'):
        for child in node.children.values():
            _collect_nodes(child, node_list)


def _sort_nodes_by_memory(nodes: List[DataTree]) -> List[DataTree]:
    """Sort nodes by memory size (smallest first)."""
    def get_size(node):
        if node.ds is None:
            return 0
        return node.ds.nbytes if hasattr(node.ds, 'nbytes') else 0
    
    return sorted(nodes, key=get_size)


def _rebuild_tree_v012(original_tree: DataTree, processed_nodes: List[DataTree]) -> DataTree:
    """Rebuild tree with filtered datasets for datatree 0.0.12."""
    # Create mapping of paths to filtered datasets
    filtered_map = {}
    for node in processed_nodes:
        if hasattr(node, '_ds_filtered'):
            filtered_map[node.path] = node._ds_filtered
    
    def rebuild_node(node: DataTree) -> DataTree:
        # Get the filtered dataset if available, otherwise original
        if node.path in filtered_map:
            ds = filtered_map[node.path]
        elif node.ds is not None:
            ds = node.ds
        else:
            ds = None
        
        # Create new node
        new_node = DataTree(data=ds, name=node.name)
        
        # Copy attributes
        if hasattr(node, 'attrs'):
            new_node.attrs.update(node.attrs)
        
        # Recursively rebuild children
        if hasattr(node, 'children'):
            for child_name, child_node in node.children.items():
                new_node[child_name] = rebuild_node(child_node)
        
        return new_node
    
    return rebuild_node(original_tree)


def filter_datatree_node_by_node(
    dt: DataTree,
    threshold: float,
    node_paths: List[str],
    fill_value=np.nan,
    remove_filtered: bool = False,
    save_intermediate: bool = False,
    intermediate_dir: Optional[str] = None,
) -> DataTree:
    """
    Filter specific nodes one at a time with maximum memory efficiency.
    
    This is the most memory-efficient approach: process one node, optionally
    save to disk, clear memory, process next node.
    
    Parameters
    ----------
    dt : DataTree
        Input DataTree.
    threshold : float
        Cost ratio threshold.
    node_paths : list of str
        Specific node paths to filter (e.g., ['/forecast', '/analysis/surface']).
    fill_value : scalar, optional
        Fill value for filtered members.
    remove_filtered : bool, optional
        Whether to remove or fill filtered members.
    save_intermediate : bool, optional
        If True, save each filtered node to disk before processing next.
    intermediate_dir : str, optional
        Directory to save intermediate results (required if save_intermediate=True).
    
    Returns
    -------
    DataTree
        Filtered DataTree.
    
    Examples
    --------
    >>> # Process only specific nodes
    >>> dt_filtered = filter_datatree_node_by_node(
    ...     dt,
    ...     threshold=0.5,
    ...     node_paths=['/forecast', '/analysis/surface'],
    ... )
    
    >>> # Save intermediate results to avoid re-computation
    >>> dt_filtered = filter_datatree_node_by_node(
    ...     dt,
    ...     threshold=0.5,
    ...     node_paths=['/forecast', '/analysis'],
    ...     save_intermediate=True,
    ...     intermediate_dir='./filtered_nodes/'
    ... )
    """
    import os
    
    if save_intermediate and intermediate_dir is None:
        raise ValueError("intermediate_dir required when save_intermediate=True")
    
    if save_intermediate:
        os.makedirs(intermediate_dir, exist_ok=True)
    
    filtered_datasets = {}
    
    for i, path in enumerate(node_paths):
        print(f"\n[{i+1}/{len(node_paths)}] Processing: {path}")
        
        # Get the node
        try:
            node = dt[path]
        except KeyError:
            print(f"  ⚠️  Node not found: {path}")
            continue
        
        if node.ds is None:
            print(f"  ⚠️  Node has no dataset")
            continue
        
        # Check memory
        if hasattr(node.ds, 'nbytes'):
            mem_mb = node.ds.nbytes / (1024**2)
            print(f"  Memory: {mem_mb:.1f} MB")
        
        # Filter this node
        print(f"  Filtering...")
        filtered_ds = filter_by_cost_ratio_memeff(
            node.ds,
            threshold=threshold,
            fill_value=fill_value,
            remove_filtered=remove_filtered,
        )
        
        filtered_ds.attrs['filtered_node_path'] = path
        
        kept = filtered_ds.attrs.get('n_members_kept', 'N/A')
        filtered = filtered_ds.attrs.get('n_members_filtered', 'N/A')
        print(f"  ✓ Complete: {kept} kept, {filtered} filtered")
        
        # Save intermediate if requested
        if save_intermediate:
            # Create safe filename from path
            safe_name = path.replace('/', '_').lstrip('_') or 'root'
            filepath = os.path.join(intermediate_dir, f"{safe_name}.nc")
            print(f"  Saving to: {filepath}")
            filtered_ds.to_netcdf(filepath)
        
        # Store in memory
        filtered_datasets[path] = filtered_ds
        
        # Force cleanup
        del filtered_ds
        gc.collect()
    
    # Rebuild tree with filtered nodes
    print(f"\nRebuilding tree...")
    
    def rebuild_with_filtered(node: DataTree) -> DataTree:
        # Use filtered dataset if available, otherwise original
        if node.path in filtered_datasets:
            ds = filtered_datasets[node.path]
        elif node.ds is not None:
            ds = node.ds
        else:
            ds = None
        
        new_node = DataTree(data=ds, name=node.name)
        if hasattr(node, 'attrs'):
            new_node.attrs.update(node.attrs)
        
        if hasattr(node, 'children'):
            for child_name, child_node in node.children.items():
                new_node[child_name] = rebuild_with_filtered(child_node)
        
        return new_node
    
    dt_filtered = rebuild_with_filtered(dt)
    print("✓ Tree rebuilt")
    
    return dt_filtered


def get_memory_efficient_summary(
    dt: DataTree,
    compute_cost_stats: bool = True,
) -> Dict:
    """
    Get memory usage summary without loading all data.
    
    Parameters
    ----------
    dt : DataTree
        Input DataTree.
    compute_cost_stats : bool, optional
        If True, compute cost ratio statistics (requires loading cost_hist).
        If False, only report memory estimates.
    
    Returns
    -------
    dict
        Memory and statistics summary.
    """
    summary = {
        'nodes': {},
        'total_memory_mb': 0,
    }
    
    def collect_info(node: DataTree):
        if node.ds is None:
            return
        
        node_info = {'path': node.path}
        
        # Memory estimate
        if hasattr(node.ds, 'nbytes'):
            mem_mb = node.ds.nbytes / (1024**2)
            node_info['memory_mb'] = mem_mb
            summary['total_memory_mb'] += mem_mb
        
        # Dataset info
        if 'ens_mem' in node.ds.dims:
            node_info['n_members'] = len(node.ds.ens_mem)
        
        if 'iter' in node.ds.dims:
            node_info['n_iterations'] = len(node.ds.iter)
        
        # Cost statistics (only if requested and available)
        if compute_cost_stats and 'cost_hist' in node.ds:
            try:
                cost_initial = node.ds.cost_hist.isel(iter=0)
                cost_final = node.ds.cost_hist.isel(iter=-1)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    cost_ratio = cost_final / cost_initial
                    cost_ratio = cost_ratio.where(cost_initial != 0, np.inf)
                
                ratios = cost_ratio.values
                valid_ratios = ratios[np.isfinite(ratios)]
                
                if len(valid_ratios) > 0:
                    node_info['mean_cost_ratio'] = float(np.mean(valid_ratios))
                    node_info['min_cost_ratio'] = float(np.min(valid_ratios))
                    node_info['max_cost_ratio'] = float(np.max(valid_ratios))
                
                # Clean up
                del cost_initial, cost_final, cost_ratio, ratios, valid_ratios
                gc.collect()
            except Exception as e:
                node_info['cost_stats_error'] = str(e)
        
        summary['nodes'][node.path] = node_info
    
    # Traverse tree
    def traverse(node):
        collect_info(node)
        if hasattr(node, 'children'):
            for child in node.children.values():
                traverse(child)
    
    traverse(dt)
    
    summary['total_memory_gb'] = summary['total_memory_mb'] / 1024
    summary['n_nodes'] = len(summary['nodes'])
    
    return summary


def print_memory_summary(summary: Dict):
    """Pretty print memory summary."""
    print("=" * 70)
    print("MEMORY SUMMARY")
    print("=" * 70)
    print(f"Total nodes: {summary['n_nodes']}")
    print(f"Total memory: {summary['total_memory_mb']:.1f} MB ({summary['total_memory_gb']:.2f} GB)")
    print("\nPer-node breakdown:")
    print("-" * 70)
    
    for path, info in summary['nodes'].items():
        mem_mb = info.get('memory_mb', 0)
        n_mem = info.get('n_members', 'N/A')
        print(f"\n{path}:")
        print(f"  Memory: {mem_mb:.1f} MB")
        print(f"  Members: {n_mem}")
        
        if 'mean_cost_ratio' in info:
            print(f"  Mean cost ratio: {info['mean_cost_ratio']:.3f}")
            print(f"  Range: [{info['min_cost_ratio']:.3f}, {info['max_cost_ratio']:.3f}]")
    
    print("=" * 70)

import xarray as xr
import numpy as np
import gc
from typing import Optional, List


def filter_by_cost_ratio_memeff(
    ds: xr.Dataset,
    threshold: float,
    fill_value=np.nan,
    remove_filtered: bool = False,
    in_place: bool = False,
    chunk_size: Optional[int] = None,
    variables_to_filter: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Memory-efficient version of filter_by_cost_ratio.
    
    This function avoids creating full dataset copies and processes variables
    one at a time or in chunks to minimize memory usage.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with a 'cost_hist' variable having dimensions (ens_mem, iter).
    threshold : float
        Maximum allowed ratio of final to initial cost.
    fill_value : scalar, optional
        Value to use for filling filtered ensemble members. Default is np.nan.
    remove_filtered : bool, optional
        If True, remove filtered ensemble members entirely.
        If False (default), fill them with fill_value.
    in_place : bool, optional
        If True, modify the dataset in place (saves memory but destructive).
        If False (default), create minimal copy. Only works when remove_filtered=False.
    chunk_size : int, optional
        Process variables in chunks of this size to reduce memory.
        If None, process all at once (faster but more memory).
    variables_to_filter : list of str, optional
        Specific variables to filter. If None, filter all variables with ens_mem.
        Useful for selective filtering to save memory.
    
    Returns
    -------
    xr.Dataset
        Filtered dataset.
    
    Notes
    -----
    Memory-saving strategies used:
    - Avoids full dataset copy
    - Processes variables individually or in chunks
    - Forces garbage collection between operations
    - Option for in-place modification
    - Can filter only specific variables
    
    Examples
    --------
    >>> # Memory-efficient filtering
    >>> ds_filtered = filter_by_cost_ratio_memeff(ds, threshold=0.5)
    
    >>> # Ultra-low memory: in-place + chunked processing
    >>> ds_filtered = filter_by_cost_ratio_memeff(
    ...     ds, 
    ...     threshold=0.5, 
    ...     in_place=True,
    ...     chunk_size=5
    ... )
    
    >>> # Filter only specific large variables
    >>> ds_filtered = filter_by_cost_ratio_memeff(
    ...     ds,
    ...     threshold=0.5,
    ...     variables_to_filter=['state_vars', 'observations']
    ... )
    """
    # Validate input
    if 'cost_hist' not in ds:
        raise ValueError("Dataset must contain 'cost_hist' variable")
    
    if 'ens_mem' not in ds.cost_hist.dims:
        raise ValueError("'cost_hist' must have 'ens_mem' dimension")
    
    if 'iter' not in ds.cost_hist.dims:
        raise ValueError("'cost_hist' must have 'iter' dimension")
    
    # Calculate cost ratios (small memory footprint)
    cost_initial = ds.cost_hist.isel(iter=0)
    cost_final = ds.cost_hist.isel(iter=-1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cost_ratio = cost_final / cost_initial
        cost_ratio = cost_ratio.where(cost_initial != 0, np.inf)
    
    # Compute the mask immediately (small array)
    keep_mask = cost_ratio <= threshold
    keep_mask_values = keep_mask.values  # Convert to numpy for faster access
    
    # Clean up intermediate arrays
    del cost_initial, cost_final, cost_ratio
    gc.collect()
    
    if remove_filtered:
        # Memory-efficient approach: select without copying everything first
        valid_ens_members = ds.ens_mem.values[keep_mask_values]
        
        # Use sel with drop=True to avoid carrying filtered indices
        ds_filtered = ds.sel(ens_mem=valid_ens_members, drop=True)
        
        # Add metadata
        ds_filtered.attrs['filtered_by_cost_ratio'] = True
        ds_filtered.attrs['cost_ratio_threshold'] = threshold
        ds_filtered.attrs['n_members_kept'] = len(valid_ens_members)
        ds_filtered.attrs['n_members_filtered'] = int((~keep_mask).sum().values)
        
        # Clean up
        del valid_ens_members
        gc.collect()
        
        return ds_filtered
    
    else:
        # For filling: process variables individually to avoid full copy
        
        # Determine which variables to filter
        if variables_to_filter is None:
            vars_to_process = [v for v in ds.data_vars if 'ens_mem' in ds[v].dims]
        else:
            vars_to_process = variables_to_filter
        
        if in_place:
            # Modify in place - DESTRUCTIVE but saves memory
            ds_filtered = ds
            print("⚠️  WARNING: Modifying dataset in-place")
        else:
            # Create a shallow copy (only copies metadata, not data)
            ds_filtered = ds.copy(deep=False)
        
        # Process variables in chunks if requested
        if chunk_size:
            for i in range(0, len(vars_to_process), chunk_size):
                chunk_vars = vars_to_process[i:i+chunk_size]
                print(f"Processing variables {i+1}-{min(i+chunk_size, len(vars_to_process))} of {len(vars_to_process)}")
                
                for var_name in chunk_vars:
                    ds_filtered[var_name] = ds[var_name].where(keep_mask, fill_value)
                
                # Force garbage collection after each chunk
                gc.collect()
        else:
            # Process all at once
            for var_name in vars_to_process:
                ds_filtered[var_name] = ds[var_name].where(keep_mask, fill_value)
        
        # Add the mask as a variable
        ds_filtered['cost_ratio_mask'] = keep_mask
        ds_filtered['cost_ratio_mask'].attrs['description'] = (
            f'Boolean mask: True where cost_hist[-1]/cost_hist[0] <= {threshold}'
        )
        
        # Add metadata
        ds_filtered.attrs['filtered_by_cost_ratio'] = True
        ds_filtered.attrs['cost_ratio_threshold'] = threshold
        ds_filtered.attrs['cost_ratio_fill_value'] = fill_value
        ds_filtered.attrs['n_members_kept'] = int(keep_mask.sum().values)
        ds_filtered.attrs['n_members_filtered'] = int((~keep_mask).sum().values)
        
        # Clean up
        gc.collect()
        
        return ds_filtered


def filter_by_cost_ratio_lazy(
    ds: xr.Dataset,
    threshold: float,
    fill_value=np.nan,
    remove_filtered: bool = False,
) -> xr.Dataset:
    """
    Lazy evaluation version - works with dask arrays.
    
    This version doesn't trigger computation until explicitly requested,
    allowing for deferred execution and better memory management.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (can use dask arrays).
    threshold : float
        Maximum allowed ratio of final to initial cost.
    fill_value : scalar, optional
        Value to use for filling filtered ensemble members.
    remove_filtered : bool, optional
        If True, remove filtered ensemble members.
    
    Returns
    -------
    xr.Dataset
        Filtered dataset with lazy evaluation.
    
    Notes
    -----
    - Operations are lazy and won't execute until .compute() or .load() is called
    - Works seamlessly with dask-backed datasets
    - Use .compute() on the result when ready to execute
    
    Examples
    --------
    >>> # Create lazy filtered dataset
    >>> ds_filtered = filter_by_cost_ratio_lazy(ds, threshold=0.5)
    >>> 
    >>> # Trigger computation when ready
    >>> ds_filtered = ds_filtered.compute()
    """
    # Validate input
    if 'cost_hist' not in ds:
        raise ValueError("Dataset must contain 'cost_hist' variable")
    
    # Calculate cost ratios lazily
    cost_initial = ds.cost_hist.isel(iter=0)
    cost_final = ds.cost_hist.isel(iter=-1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cost_ratio = cost_final / cost_initial
        cost_ratio = cost_ratio.where(cost_initial != 0, np.inf)
    
    keep_mask = cost_ratio <= threshold
    
    if remove_filtered:
        # Need to compute mask to get valid indices
        keep_mask_computed = keep_mask.compute()
        valid_ens_members = ds.ens_mem.values[keep_mask_computed.values]
        ds_filtered = ds.sel(ens_mem=valid_ens_members, drop=True)
        
        ds_filtered.attrs['filtered_by_cost_ratio'] = True
        ds_filtered.attrs['cost_ratio_threshold'] = threshold
        ds_filtered.attrs['n_members_kept'] = len(valid_ens_members)
        ds_filtered.attrs['n_members_filtered'] = int((~keep_mask_computed).sum().values)
    else:
        # Lazy masking - doesn't execute until compute()
        ds_filtered = ds.copy(deep=False)
        for var_name in ds.data_vars:
            if 'ens_mem' in ds[var_name].dims:
                ds_filtered[var_name] = ds[var_name].where(keep_mask, fill_value)
        
        ds_filtered['cost_ratio_mask'] = keep_mask
        
        ds_filtered.attrs['filtered_by_cost_ratio'] = True
        ds_filtered.attrs['cost_ratio_threshold'] = threshold
        ds_filtered.attrs['cost_ratio_fill_value'] = fill_value
    
    return ds_filtered


def estimate_memory_usage(ds: xr.Dataset, variables: Optional[List[str]] = None) -> dict:
    """
    Estimate memory usage of dataset or specific variables.
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.
    variables : list of str, optional
        Specific variables to check. If None, check all.
    
    Returns
    -------
    dict
        Memory usage information in MB.
    
    Examples
    --------
    >>> mem_info = estimate_memory_usage(ds)
    >>> print(f"Total: {mem_info['total_mb']:.1f} MB")
    >>> for var, size in mem_info['by_variable'].items():
    ...     print(f"  {var}: {size:.1f} MB")
    """
    vars_to_check = variables if variables else list(ds.data_vars)
    
    mem_by_var = {}
    total_bytes = 0
    
    for var in vars_to_check:
        if var in ds:
            var_bytes = ds[var].nbytes
            mem_by_var[var] = var_bytes / (1024**2)  # Convert to MB
            total_bytes += var_bytes
    
    return {
        'total_mb': total_bytes / (1024**2),
        'total_gb': total_bytes / (1024**3),
        'by_variable': mem_by_var,
    }


def batch_filter_datasets(
    datasets: List[xr.Dataset],
    threshold: float,
    fill_value=np.nan,
    remove_filtered: bool = False,
    clear_between: bool = True,
) -> List[xr.Dataset]:
    """
    Filter multiple datasets with memory management between each.
    
    Useful when processing multiple ensemble runs or time periods.
    
    Parameters
    ----------
    datasets : list of xr.Dataset
        List of datasets to filter.
    threshold : float
        Cost ratio threshold.
    fill_value : scalar, optional
        Fill value for filtered members.
    remove_filtered : bool, optional
        Whether to remove or fill filtered members.
    clear_between : bool, optional
        If True, force garbage collection between datasets.
    
    Returns
    -------
    list of xr.Dataset
        Filtered datasets.
    """
    filtered_datasets = []
    
    for i, ds in enumerate(datasets):
        print(f"Filtering dataset {i+1}/{len(datasets)}...")
        
        ds_filtered = filter_by_cost_ratio_memeff(
            ds,
            threshold=threshold,
            fill_value=fill_value,
            remove_filtered=remove_filtered,
        )
        
        filtered_datasets.append(ds_filtered)
        
        if clear_between:
            # Clean up before next iteration
            del ds_filtered
            gc.collect()
    
    return filtered_datasets

def filter_by_max_conceivable(angles, MAX_CONCEIVABLE_ANGLE):
    return np.where(angles > MAX_CONCEIVABLE_ANGLE, np.nan, angles)