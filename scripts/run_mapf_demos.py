#!/usr/bin/env python3
"""
Run MAPF on multiple maps with many agents and generate visualization GIFs.
Uses a simple prioritized A* planner.
"""

import argparse
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Set
import numpy as np

from mapf_env.io.movingai_map import load_map
from mapf_env.io.movingai_scene import load_scen
from core.instance import MAPFInstance, instance_from_scen
from mapf_env.viz.animate import animate_paths


def astar_path(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    reserved_cells: Optional[Set[Tuple[int, int]]] = None,
    motion: str = "4",
) -> Optional[List[Tuple[int, int]]]:
    """
    A* pathfinding from start to goal.
    
    Args:
        grid: 2D array (0=free, 1=obstacle)
        start: (row, col) start position
        goal: (row, col) goal position
        reserved_cells: Set of (row, col) cells to avoid (for prioritized planning)
        motion: "4" or "8" connected
    
    Returns:
        List of (row, col) positions from start to goal, or None if no path found
    """
    if reserved_cells is None:
        reserved_cells = set()
    
    H, W = grid.shape
    sr, sc = start
    gr, gc = goal
    
    # Check if start/goal are valid
    if grid[sr, sc] == 1 or grid[gr, gc] == 1:
        return None
    
    # Heuristic: Manhattan distance
    def heuristic(r, c):
        return abs(r - gr) + abs(c - gc)
    
    # Get neighbors based on motion model
    if motion == "4":
        deltas = [(0, 1), (1, 0), (-1, 0), (0, -1)]  # right, down, up, left
    else:  # 8-connected
        deltas = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    # A* search
    open_set = [(0, sr, sc)]  # (f, r, c)
    came_from = {}
    g_score = {(sr, sc): 0}
    f_score = {(sr, sc): heuristic(sr, sc)}
    visited = set()
    
    while open_set:
        open_set.sort()  # Simple priority queue (could use heapq for better performance)
        current_f, r, c = open_set.pop(0)
        
        if (r, c) in visited:
            continue
        visited.add((r, c))
        
        if r == gr and c == gc:
            # Reconstruct path
            path = []
            while (r, c) in came_from:
                path.append((r, c))
                r, c = came_from[(r, c)]
            path.append((sr, sc))
            path.reverse()
            return path
        
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                continue
            
            # Check obstacle
            if grid[nr, nc] == 1:
                continue
            
            # Check reserved cells (for prioritized planning)
            if (nr, nc) in reserved_cells:
                continue
            
            tentative_g = g_score.get((r, c), float('inf')) + 1
            
            if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                came_from[(nr, nc)] = (r, c)
                g_score[(nr, nc)] = tentative_g
                f = tentative_g + heuristic(nr, nc)
                f_score[(nr, nc)] = f
                open_set.append((f, nr, nc))
    
    return None  # No path found


def random_movement_planner(
    instance: MAPFInstance,
    motion: str = "4",
    max_timesteps: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Simple random movement - agents move randomly on the map.
    Just for visualization purposes.
    
    Returns:
        paths: (T, N, 2) array of (row, col) positions
    """
    grid = instance.grid
    starts = instance.starts
    N = instance.num_agents
    H, W = grid.shape
    
    np.random.seed(seed)
    
    # Get allowed deltas based on motion model
    if motion == "4":
        deltas = [(0, 1), (1, 0), (-1, 0), (0, -1), (0, 0)]  # right, down, up, left, wait
    else:  # 8-connected
        deltas = [(0, 1), (1, 0), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0)]
    
    # Initialize paths
    paths = np.zeros((max_timesteps, N, 2), dtype=np.int32)
    current_pos = starts.copy()
    
    for t in range(max_timesteps):
        # Store current positions
        paths[t] = current_pos.copy()
        
        # Move each agent randomly
        next_pos = current_pos.copy()
        
        for agent_id in range(N):
            r, c = int(current_pos[agent_id, 0]), int(current_pos[agent_id, 1])
            
            # Try random moves until we find a valid one
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts:
                # Pick a random direction
                dr, dc = deltas[np.random.randint(len(deltas))]
                nr, nc = r + dr, c + dc
                
                # Check bounds
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    attempts += 1
                    continue
                
                # Check obstacle
                if grid[nr, nc] == 1:
                    attempts += 1
                    continue
                
                # Valid move found
                next_pos[agent_id, 0] = nr
                next_pos[agent_id, 1] = nc
                break
            
            # If no valid move found after max_attempts, stay put
            if attempts >= max_attempts:
                next_pos[agent_id, 0] = r
                next_pos[agent_id, 1] = c
        
        current_pos = next_pos
    
    return paths


def sample_random_starts_goals(
    grid: np.ndarray,
    num_agents: int,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample random start and goal positions on free cells.
    """
    if seed is not None:
        np.random.seed(seed)
    
    H, W = grid.shape
    free_cells = np.argwhere(grid == 0)
    
    if len(free_cells) < num_agents * 2:
        raise ValueError(f"Not enough free cells ({len(free_cells)}) for {num_agents * 2} positions")
    
    # Sample without replacement
    indices = np.random.choice(len(free_cells), size=num_agents * 2, replace=False)
    selected = free_cells[indices]
    
    starts = selected[:num_agents]
    goals = selected[num_agents:]
    
    return starts, goals


def main():
    parser = argparse.ArgumentParser(
        description="Run MAPF on multiple maps and generate visualization GIFs."
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        required=True,
        help="Map basenames (without .map extension), e.g. 'warehouse-20-40-10-2-2'",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of agents (default: 50)",
    )
    parser.add_argument(
        "--maps_dir",
        type=str,
        default="data/mapf-map",
        help="Directory containing .map files",
    )
    parser.add_argument(
        "--scen_dir",
        type=str,
        default="data/scens",
        help="Directory containing .scen files",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for GIF (default: 8 - fast but visible)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal stride for GIF (default: 1 = all frames)",
    )
    parser.add_argument(
        "--motion",
        choices=["4", "8"],
        default="4",
        help="Motion model: 4-connected or 8-connected",
    )
    parser.add_argument(
        "--use_scenarios",
        action="store_true",
        help="Use scenario files if available, otherwise sample random starts/goals",
    )
    parser.add_argument(
        "--max_timesteps",
        type=int,
        default=1000,
        help="Maximum timesteps for paths (default: 1000)",
    )
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    
    maps_dir = Path(args.maps_dir)
    scen_dir = Path(args.scen_dir)
    
    for map_name in args.maps:
        print(f"\n{'='*60}")
        print(f"Processing map: {map_name}")
        print(f"{'='*60}")
        
        map_path = maps_dir / f"{map_name}.map"
        if not map_path.exists():
            print(f"Warning: Map file not found: {map_path}, skipping...")
            continue
        
        # Load map
        grid = load_map(map_path)
        H, W = grid.shape
        print(f"Loaded map: {H}x{W}")
        
        # Try to load scenario file
        instance = None
        if args.use_scenarios:
            scen_files = list(scen_dir.glob(f"{map_name}-random-*.scen"))
            if scen_files:
                scen_path = scen_files[0]
                print(f"Using scenario: {scen_path}")
                try:
                    scen_starts, scen_goals = load_scen(scen_path)
                    # Filter valid entries
                    valid_mask = (
                        (scen_starts[:, 0] >= 0) & (scen_starts[:, 0] < H) &
                        (scen_starts[:, 1] >= 0) & (scen_starts[:, 1] < W) &
                        (scen_goals[:, 0] >= 0) & (scen_goals[:, 0] < H) &
                        (scen_goals[:, 1] >= 0) & (scen_goals[:, 1] < W)
                    )
                    valid_starts = scen_starts[valid_mask]
                    valid_goals = scen_goals[valid_mask]
                    
                    # Check free cells
                    starts_on_free = grid[valid_starts[:, 0], valid_starts[:, 1]] == 0
                    goals_on_free = grid[valid_goals[:, 0], valid_goals[:, 1]] == 0
                    valid_mask2 = starts_on_free & goals_on_free
                    
                    if valid_mask2.sum() >= args.k:
                        instance = instance_from_scen(
                            grid=grid,
                            scen_starts=valid_starts[valid_mask2],
                            scen_goals=valid_goals[valid_mask2],
                            k=args.k,
                            offset=0,
                        )
                        print(f"Loaded instance with {instance.num_agents} agents from scenario")
                except Exception as e:
                    print(f"Warning: Could not load scenario: {e}")
        
        # If no scenario, sample random starts/goals
        if instance is None:
            print("Sampling random starts/goals...")
            try:
                starts, goals = sample_random_starts_goals(grid, args.k, seed=42)
                instance = MAPFInstance(
                    grid=grid,
                    starts=starts,
                    goals=goals,
                    num_agents=args.k,
                )
                print(f"Created instance with {instance.num_agents} agents")
            except Exception as e:
                print(f"Error creating instance: {e}, skipping map...")
                continue
        
        # Generate random movement paths (just for visualization)
        print(f"Generating random movement paths ({args.motion}-connected)...")
        paths = random_movement_planner(instance, motion=args.motion, max_timesteps=args.max_timesteps, seed=42)
        T, N, _ = paths.shape
        print(f"Generated paths: T={T}, N={N}")
        
        # Save paths
        paths_file = results_dir / f"{map_name}_paths.npy"
        np.save(str(paths_file), paths)
        print(f"Saved paths to: {paths_file}")
        
        # Generate GIF
        gif_file = results_dir / f"{map_name}_demo.gif"
        print(f"Generating GIF: {gif_file} (fps={args.fps}, stride={args.stride})...")
        
        animate_paths(
            grid=grid,
            paths=paths,
            starts=instance.starts,
            goals=instance.goals,
            out=str(gif_file),
            fps=args.fps,
            stride=args.stride,
            highlight_collisions=True,
        )
        
        print(f"Saved GIF to: {gif_file}")
        print(f"✓ Completed: {map_name}")
    
    print(f"\n{'='*60}")
    print(f"All done! Results saved to: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

