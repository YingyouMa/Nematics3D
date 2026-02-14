    def _helper_detect_defects(self, directors, threshold: float=0):
        
        def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
            """Wrap angles to (-pi, pi]."""
            return (angle + np.pi) % (2.0 * np.pi) - np.pi
        
        plane_grid = self._entity_plane
        points = plane_grid._entity_grid_all
        polar = plane_grid._entity_polar
        ring_offsets = plane_grid._calc_ring_offsets
        
        n_rings = ring_offsets.shape[0] - 1
        adjacent_mask = np.zeros((points.shape[0],), dtype=bool)
        defect_centers_chunks = []
    
        # If ring 0 is origin block (size 1 at r=0), skip it for quad loops
        start_ring = 0
        if n_rings >= 1:
            s0, e0 = ring_offsets[0], ring_offsets[1]
            if (e0 - s0) == 1 and np.isclose(polar[s0, 0], 0.0):
                start_ring = 1
    
        # ----------------------------
        # Helper: process one ring-pair (outer -> inner)
        # ----------------------------
        def _process_outer_to_inner(s_outer: int, e_outer: int, s_inner: int, e_inner: int) -> None:
            n_outer = e_outer - s_outer
            n_inner = e_inner - s_inner
            if n_outer < 2 or n_inner < 2:
                return
    
            theta_outer = polar[s_outer:e_outer, 1]  # (n_outer,)
            theta_inner = polar[s_inner:e_inner, 1]  # (n_inner,)
    
            # base edges on OUTER ring
            j = np.arange(n_outer, dtype=np.int64)
            jn = (j + 1) % n_outer
    
            idx_a = s_outer + j
            idx_b = s_outer + jn
    
            theta_a = theta_outer[j]   # (n_outer,)
            theta_b = theta_outer[jn]  # (n_outer,)
    
            # c: nearest on INNER ring to theta_b (no sorted assumption)
            diff_b = _wrap_to_pi(theta_inner[None, :] - theta_b[:, None])      # (n_outer, n_inner)
            c_local = np.argmin(np.abs(diff_b), axis=1).astype(np.int64)       # (n_outer,)
    
            # inner-ring adjacency via sorting theta_inner into a circular order
            order = np.argsort(theta_inner)                                    # (n_inner,)
            rank_of = np.empty_like(order)
            rank_of[order] = np.arange(n_inner, dtype=np.int64)                # local_index -> rank
    
            c_rank = rank_of[c_local]                                          # (n_outer,)
            prev_rank = (c_rank - 1) % n_inner
            next_rank = (c_rank + 1) % n_inner
    
            prev_local = order[prev_rank]                                      # (n_outer,)
            next_local = order[next_rank]                                      # (n_outer,)
    
            # d: choose neighbor-of-c closer to theta_a
            d_prev = np.abs(_wrap_to_pi(theta_inner[prev_local] - theta_a))    # (n_outer,)
            d_next = np.abs(_wrap_to_pi(theta_inner[next_local] - theta_a))    # (n_outer,)
            d_local = np.where(d_prev <= d_next, prev_local, next_local).astype(np.int64)
    
            idx_c = s_inner + c_local
            idx_d = s_inner + d_local
    
            # loop vertex coords
            pa = points[idx_a]
            pb = points[idx_b]
            pc = points[idx_c]
            pd = points[idx_d]
    
            # loop directors
            a = directors[idx_a]
            b_raw = directors[idx_b]
            c_raw = directors[idx_c]
            d_raw = directors[idx_d]
    
            # align along loop: a -> b -> c -> d
            b = align_directors(a, b_raw)
            c = align_directors(b, c_raw)
            d = align_directors(c, d_raw)
    
            test = np.einsum("...i,...i->...", a, d)  # (n_outer,)
            hit = test < threshold
            if not np.any(hit):
                return
    
            centers = (pa + pb + pc + pd) * 0.25
            defect_centers_chunks.append(centers[hit])
    
            adjacent_mask[idx_a[hit]] = True
            adjacent_mask[idx_b[hit]] = True
            inner_idx = np.unique(np.concatenate([idx_c[hit], idx_d[hit]]))
            adjacent_mask[inner_idx] = True
            
        # ----------------------------
        # Outer -> inner traversal with stop rule (inner ring < 6)
        # ----------------------------
        outermost = n_rings - 1
        last_good_ring = None  # the ring index (k) of the last ring with >= 6 points before stopping
    
        # Walk inward: (outer=r, inner=r-1)
        for r in range(outermost, start_ring, -1):
            s_outer, e_outer = ring_offsets[r], ring_offsets[r + 1]
            s_inner, e_inner = ring_offsets[r - 1], ring_offsets[r]
    
            n_inner = e_inner - s_inner
    
            # stop condition: we are about to step into a ring with < 6 points
            if n_inner < 6:
                last_good_ring = r  # the current outer ring is the last ring still >= 6 (in practice)
                break
    
            # process this ring-pair (outer -> inner)
            _process_outer_to_inner(s_outer, e_outer, s_inner, e_inner)
    
        # If we never hit n_inner < 6, then the innermost ring we reached is start_ring
        if last_good_ring is None:
            last_good_ring = start_ring if (ring_offsets[start_ring + 1] - ring_offsets[start_ring]) >= 6 else None
    
        # ----------------------------
        # Final-ring internal closure check (only if that ring has >= 6 points)
        # ----------------------------
        if last_good_ring is not None:
            s, e = ring_offsets[last_good_ring], ring_offsets[last_good_ring + 1]
            n_last = e - s
    
            if n_last >= 6:
                v = directors[s:e]  # (n_last, 3)
    
                # Vectorized "sequential alignment" using cumulative signs of neighbor dots.
                # NOTE: avoid sign==0 zeroing by using a strict <0 test here.
                dots = np.einsum("ij,ij->i", v[:-1], v[1:])                    # (n_last-1,)
                step_sign = np.where(dots < 0.0, -1.0, 1.0).astype(v.dtype)     # (n_last-1,)
    
                cum_sign = np.concatenate(
                    [np.ones((1,), dtype=v.dtype), np.cumprod(step_sign)]
                )  # (n_last,)
    
                v_aligned_last = v[-1] * cum_sign[-1]
                closure = float(np.dot(v[0], v_aligned_last))
    
                if closure < threshold:
                    # mark this whole ring as adjacent-to-defect
                    adjacent_mask[s:e] = True
    
                    # put one defect center for this ring (use mean position)
                    defect_centers_chunks.append(points[s:e].mean(axis=0, keepdims=True))
    
        defect_centers = (
            np.concatenate(defect_centers_chunks, axis=0).astype(float)
            if defect_centers_chunks
            else np.zeros((0, 3), dtype=float)
        )
    
        return defect_centers, adjacent_mask
    