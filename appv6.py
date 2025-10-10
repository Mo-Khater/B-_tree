# app.py
"""
Streamlit B+ Tree Visualization (single-file)
Run with: streamlit run app.py

Features:
- Configurable order (m)
- Insert / Delete / Search with step-by-step generators
- Visualization with Graphviz (HTML-like node labels)
- Highlights: current node (yellow), comparisons (red), promotions/borrow (blue)
- Pseudocode panel with highlighted line, operation log, example sequences
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Generator, Dict
import uuid
import math
import copy
import streamlit as st
import time
import html

# -----------------------
# Step dataclass
# -----------------------
@dataclass
class Step:
    id: int
    action: str                  # e.g., "compare", "descend", "insert_leaf", "split", "promote", "borrow", "merge", "delete", "search"
    description: str             # one-line explanation
    detailed_explanation: str = ""
    pseudocode_line: Optional[int] = None
    tree_snapshot: Any = None           # serialized snapshot for visualization
    highlight_nodes: List[str] = field(default_factory=list)   # node IDs currently highlighted
    highlight_keys: List[Tuple[str, int]] = field(default_factory=list)  # (node_id, key_index)
    meta: dict = field(default_factory=dict)

# -----------------------
# B+ Tree Node
# -----------------------
class BPlusNode:
    _id_counter = 0

    def __init__(self, leaf: bool = False):
        self.keys: List[int] = []
        self.children: List['BPlusNode'] = []  # for internal nodes: len(children) = len(keys)+1
        self.leaf = leaf
        self.next: Optional['BPlusNode'] = None  # for leaves: pointer to next leaf
        self.id = f"n{BPlusNode._id_counter}"
        BPlusNode._id_counter += 1

    def __repr__(self):
        return f"<{'Leaf' if self.leaf else 'Int'} {self.id} keys={self.keys}>"

# -----------------------
# Utility: snapshot the tree to a simple serializable structure
# -----------------------
def tree_to_snapshot(root: Optional[BPlusNode]) -> Dict[str, Any]:
    nodes = {}

    def rec(node: BPlusNode):
        if node is None or node.id in nodes:
            return
        nodes[node.id] = {
            "id": node.id,
            "keys": list(node.keys),
            "leaf": node.leaf,
            "children": [c.id for c in node.children] if not node.leaf else [],
            "next": node.next.id if (node.leaf and node.next) else None,
        }
        if not node.leaf:
            for c in node.children:
                rec(c)
        else:
            if node.next:
                rec(node.next)

    if root:
        rec(root)

    snapshot = {"root": root.id if root else None, "nodes": nodes}

    # --- FIX: detect split stage and add pseudo-parent for layout ---
    # Find leaf pairs with next pointers but no internal parent
    if root and root.leaf:
        # root is a leaf, but there may be split siblings chained by next
        leaves = list(nodes.keys())
        if len(leaves) >= 2:
            # Add a pseudo parent to align them
            pseudo_id = f"pseudo_{uuid.uuid4().hex[:6]}"
            snapshot["nodes"][pseudo_id] = {
                "id": pseudo_id,
                "keys": [],  # visually neutral
                "leaf": False,
                "children": leaves,
                "next": None,
            }
            snapshot["root"] = pseudo_id

    return snapshot


# -----------------------
# B+ Tree Implementation with step generators
# -----------------------
class BPlusTree:
    def __init__(self, order: int = 4, allow_duplicates: bool = False):
        if order < 3:
            raise ValueError("Order must be at least 3")
        self.order = order
        self.root: Optional[BPlusNode] = None
        self.allow_duplicates = allow_duplicates
        # derived
        self.max_keys = self.order - 1
        self.min_children = math.ceil(self.order / 2)
        self.min_keys = self.min_children -1
        # step id counter
        self._step_id = 0

    def _next_step_id(self) -> int:
        self._step_id += 1
        return self._step_id

    # -----------------------
    # Validation helper (invariants)
    # -----------------------
    def validate(self) -> Tuple[bool, List[str]]:
        errors = []
        if not self.root:
            return True, []
        # all leaves same depth
        leaf_depths = []
        def rec(node: BPlusNode, depth: int):
            if node.leaf:
                leaf_depths.append(depth)
            else:
                if len(node.keys) > self.max_keys:
                    errors.append(f"{node} has {len(node.keys)} keys > max {self.max_keys}")
                if node is self.root and len(node.children) < 2:
                    # root can be smaller, but if it's internal must have at least 2 children
                    errors.append(f"root internal node must have >=2 children")
                for c in node.children:
                    rec(c, depth+1)
        rec(self.root, 0)
        if leaf_depths and max(leaf_depths) != min(leaf_depths):
            errors.append("Leaves are not at the same depth: " + str(leaf_depths))
        return (len(errors) == 0, errors)

    # -----------------------
    # Helper: produce step (captures snapshot)
    # -----------------------
    def _make_step(self, action, description, detailed="", pseudocode_line=None,
                highlight_nodes=None, highlight_keys=None, meta=None) -> Step:
        """
        Produce a Step containing a snapshot and auto-fix floating nodes by inferring
        their missing parent connections (for split steps and similar transitions).
        """
        snap = tree_to_snapshot(self.root)

        try:
            nodes = snap["nodes"]
            # --- Build a set of all nodes that already appear as children ---
            attached = set()
            for n in nodes.values():
                if not n["leaf"]:
                    attached.update(n.get("children", []))

            # --- Try to find floating nodes (usually new right splits) ---
            floating = [nid for nid in nodes if nid not in attached and nid != snap.get("root")]

            # Try to pair each floating node with a plausible parent:
            for fid in floating:
                fnode = nodes[fid]
                # skip if it's root or already attached
                if fid == snap.get("root"):
                    continue

                # Look for a left neighbor that points to this one (leaf next or internal next)
                left_id = None
                for nid, nd in nodes.items():
                    if nd.get("next") == fid:
                        left_id = nid
                        break
                    if not nd["leaf"] and "children" in nd:
                        ch = nd["children"]
                        for i in range(len(ch) - 1):
                            if ch[i] == fid:
                                left_id = ch[i]
                                break

                if left_id:
                    # Find the parent of the left sibling
                    parent_id = None
                    for nid, nd in nodes.items():
                        if not nd["leaf"] and left_id in nd.get("children", []):
                            parent_id = nid
                            break

                    if parent_id and fid not in nodes[parent_id]["children"]:
                        idx = nodes[parent_id]["children"].index(left_id)
                        nodes[parent_id]["children"].insert(idx + 1, fid)
                    elif not parent_id:
                        # no parent yet → create pseudo parent
                        pseudo_id = f"pseudo_{uuid.uuid4().hex[:6]}"
                        nodes[pseudo_id] = {
                            "id": pseudo_id,
                            "keys": [],
                            "leaf": False,
                            "children": [left_id, fid],
                            "next": None
                        }
                        snap["root"] = pseudo_id
        except Exception:
            pass  # avoid breaking step capture

        return Step(
            id=self._next_step_id(),
            action=action,
            description=description,
            detailed_explanation=detailed,
            pseudocode_line=pseudocode_line,
            tree_snapshot=snap,
            highlight_nodes=highlight_nodes or [],
            highlight_keys=highlight_keys or [],
            meta=meta or {}
        )






    # -----------------------
    # Search Steps
    # -----------------------
    def search_steps(self, key) -> Generator[Step, None, Optional[bool]]:
        """
        Yields detailed steps of searching for key.
        """
        if not self.root:
            yield self._make_step("search", f"Tree is empty — '{key}' not found.", detailed="There is no root.", pseudocode_line=0)
            return False

        node = self.root
        while not node.leaf:
            # compare with each key (sequential)
            for idx, k in enumerate(node.keys):
                # compare step
                s = self._make_step("compare", f"Comparing search key {key} with node key {k}.",
                                    detailed=f"At internal node {node.id}, compare with key {k}.",
                                    pseudocode_line=1,
                                    highlight_nodes=[node.id],
                                    highlight_keys=[(node.id, idx)])
                yield s
                if key < k:
                    # descending between previous child and this one
                    child_idx = idx
                    s2 = self._make_step("descend", f"Key {key} < {k} — traverse to child {child_idx}.",
                                         detailed=f"Following child {child_idx} of node {node.id}.",
                                         pseudocode_line=2,
                                         highlight_nodes=[node.id],
                                         highlight_keys=[(node.id, idx)])
                    yield s2
                    node = node.children[child_idx]
                    break
            else:
                # key is greater than all keys in node -> descend to last child
                child_idx = len(node.keys)
                s2 = self._make_step("descend", f"Key {key} larger than all keys — traverse to rightmost child {child_idx}.",
                                     detailed=f"Following child {child_idx} of node {node.id}.",
                                     pseudocode_line=2,
                                     highlight_nodes=[node.id])
                yield s2
                node = node.children[child_idx]

        # now at leaf
        # comparisons at leaf
        for idx, k in enumerate(node.keys):
            s = self._make_step("compare", f"Comparing search key {key} with leaf key {k}.",
                                detailed=f"At leaf {node.id}.",
                                pseudocode_line=3,
                                highlight_nodes=[node.id],
                                highlight_keys=[(node.id, idx)])
            yield s
            if key == k:
                found = self._make_step("search_found", f"Found key {key} in leaf {node.id} at index {idx}.",
                                        detailed=f"Search successful.",
                                        pseudocode_line=4,
                                        highlight_nodes=[node.id],
                                        highlight_keys=[(node.id, idx)])
                yield found
                return True
        # not found
        notfound = self._make_step("search_not_found", f"Key {key} not found in leaf {node.id}.",
                                   detailed="Reached leaf and key not present.",
                                   pseudocode_line=5,
                                   highlight_nodes=[node.id])
        yield notfound
        return False

    # -----------------------
    # Insert Steps
    # -----------------------
    def insert_steps(self, key, value=None) -> Generator[Step, None, None]:
        """
        Yields step-by-step actions for insertion.
        For simplicity this version handles integer keys and focuses on the structural steps.
        """
        # If tree empty -> create leaf
        if not self.root:
            leaf = BPlusNode(leaf=True)
            leaf.keys.append(key)
            self.root = leaf
            yield self._make_step("insert_root", f"Tree empty — create root leaf and insert {key}.",
                                  detailed="Inserted as root leaf.",
                                  pseudocode_line=0,
                                  highlight_nodes=[leaf.id],
                                  highlight_keys=[(leaf.id, 0)])
            return

        # Step 1: Find leaf and record path for later split propagation
        path: List[BPlusNode] = []
        node = self.root
        while not node.leaf:
            path.append(node)
            # compare with each key
            decided_child_idx = None
            for idx, k in enumerate(node.keys):
                # compare
                yield self._make_step("compare", f"Comparing key {key} with node key {k} in node {node.id}.",
                                      detailed=f"At internal node {node.id}.",
                                      pseudocode_line=1,
                                      highlight_nodes=[node.id],
                                      highlight_keys=[(node.id, idx)])
                if key < k:
                    decided_child_idx = idx
                    yield self._make_step("descend", f"Key {key} < {k} — traverse to child {decided_child_idx}.",
                                          detailed=f"Following child {decided_child_idx} pointer of node {node.id}.",
                                          pseudocode_line=2,
                                          highlight_nodes=[node.id],
                                          highlight_keys=[(node.id, idx)])
                    node = node.children[decided_child_idx]
                    break
            if decided_child_idx is None:
                # go to rightmost child
                decided_child_idx = len(node.keys)
                yield self._make_step("descend", f"Key {key} greater than all keys — traverse to rightmost child {decided_child_idx}.",
                                      detailed=f"Following child {decided_child_idx} of node {node.id}.",
                                      pseudocode_line=2,
                                      highlight_nodes=[node.id])
                node = node.children[decided_child_idx]

        # node is the leaf where insertion happens
        leaf = node
        path.append(leaf)
        # highlight leaf
        yield self._make_step("insert_leaf", f"Reached leaf {leaf.id}; will insert {key} in sorted order.",
                              detailed=f"Leaf before insert: {leaf.keys}",
                              pseudocode_line=3,
                              highlight_nodes=[leaf.id])

        # Insert key in sorted position (handle duplicates)
        insert_idx = 0
        while insert_idx < len(leaf.keys) and leaf.keys[insert_idx] < key:
            # show comparison with existing leaf keys as steps
            yield self._make_step("compare", f"Compare insert key {key} with leaf key {leaf.keys[insert_idx]} at index {insert_idx}.",
                                  detailed=f"At leaf {leaf.id}.",
                                  pseudocode_line=3,
                                  highlight_nodes=[leaf.id],
                                  highlight_keys=[(leaf.id, insert_idx)])
            insert_idx += 1

        # If duplicates not allowed and already present:
        if not self.allow_duplicates and insert_idx < len(leaf.keys) and leaf.keys[insert_idx] == key:
            yield self._make_step("duplicate", f"Key {key} already exists in leaf {leaf.id}; duplicate not allowed.",
                                  detailed="No insertion performed.",
                                  pseudocode_line=3,
                                  highlight_nodes=[leaf.id],
                                  highlight_keys=[(leaf.id, insert_idx)])
            return

        # show insertion point
        yield self._make_step("insert_at", f"Inserting {key} into leaf {leaf.id} at index {insert_idx}.",
                              detailed=f"Leaf keys before: {leaf.keys}",
                              pseudocode_line=4,
                              highlight_nodes=[leaf.id])

        leaf.keys.insert(insert_idx, key)

        # snapshot after actual insertion
        yield self._make_step("after_insert", f"Inserted {key} into leaf {leaf.id}. New keys: {leaf.keys}",
                              detailed="Check for overflow.",
                              pseudocode_line=5,
                              highlight_nodes=[leaf.id],
                              highlight_keys=[(leaf.id, insert_idx)])

        # If no overflow -> done
        if len(leaf.keys) <= self.max_keys:
            yield self._make_step("done", f"Insertion complete; no overflow in leaf {leaf.id}.",
                                  detailed=f"Leaf keys: {leaf.keys}",
                                  pseudocode_line=6,
                                  highlight_nodes=[leaf.id])
            return

        # Overflow happened -> split leaf
        # split point: ceil((max_keys+1)/2) or median of current count
        total = len(leaf.keys)
        split_index = math.ceil((total- 1) / 2)
        left_keys = leaf.keys[:split_index]
        right_keys = leaf.keys[split_index:]

        new_leaf = BPlusNode(leaf=True)
        new_leaf.keys = right_keys
        new_leaf.next = leaf.next
        leaf.keys = left_keys
        leaf.next = new_leaf

        # The promoted key is the first key of the right node
        promoted_key = new_leaf.keys[0]

        yield self._make_step("split_leaf", f"Leaf {leaf.id} overflowed — split into {leaf.id} and {new_leaf.id}. Promote {promoted_key}.",
                              detailed=f"Left: {leaf.keys} Right: {new_leaf.keys}",
                              pseudocode_line=7,
                              highlight_nodes=[leaf.id, new_leaf.id],
                              highlight_keys=[(new_leaf.id, 0)],
                              meta={"promoted_key": promoted_key})

        # Attach the new_leaf into parent
        # If split at root (root is leaf)
        if leaf is self.root:
            new_root = BPlusNode(leaf=False)
            new_root.keys = [promoted_key]
            new_root.children = [leaf, new_leaf]
            self.root = new_root
            yield self._make_step("new_root", f"Old root leaf split — create new root {new_root.id} with key {promoted_key}.",
                                  detailed="Root updated.",
                                  pseudocode_line=8,
                                  highlight_nodes=[new_root.id],
                                  highlight_keys=[(new_root.id, 0)])
            return

        # Otherwise, propagate to parents using path (excluding root at path[0] maybe)
        # path holds internal nodes then leaf at end
        # Remove leaf from end of path
        # We'll walk backwards handling internal insertions and possible splits
        parent_index = len(path) - 2  # index of parent node in path
        child_left = leaf
        child_right = new_leaf
        key_to_insert = promoted_key

        while parent_index >= 0:
            parent = path[parent_index]
            # insert key_to_insert into parent at correct position
            insert_idx = 0
            while insert_idx < len(parent.keys) and parent.keys[insert_idx] < key_to_insert:
                yield self._make_step("compare", f"Comparing promoted key {key_to_insert} with parent key {parent.keys[insert_idx]} in {parent.id}.",
                                      detailed=f"Inserting into parent {parent.id}.",
                                      pseudocode_line=9,
                                      highlight_nodes=[parent.id],
                                      highlight_keys=[(parent.id, insert_idx)])
                insert_idx += 1

            yield self._make_step("insert_internal", f"Inserting promoted key {key_to_insert} into internal node {parent.id} at index {insert_idx}.",
                                  detailed=f"Parent keys before: {parent.keys}",
                                  pseudocode_line=10,
                                  highlight_nodes=[parent.id])

            parent.keys.insert(insert_idx, key_to_insert)
            # children must insert child_right after the child_left
            # find index of child_left in children
            try:
                child_pos = parent.children.index(child_left)
            except ValueError:
                # fallback: insert by position (shouldn't happen)
                child_pos = insert_idx
            parent.children.insert(child_pos + 1, child_right)

            yield self._make_step("after_insert_internal", f"Inserted key {key_to_insert} into {parent.id}; keys now {parent.keys}.",
                                  detailed="Check for overflow in parent.",
                                  pseudocode_line=11,
                                  highlight_nodes=[parent.id])

            # check for overflow
            if len(parent.keys) <= self.max_keys:
                # done
                yield self._make_step("done_propagate", f"No overflow at parent {parent.id}; insertion complete.",
                                      detailed=f"Parent keys: {parent.keys}",
                                      pseudocode_line=12,
                                      highlight_nodes=[parent.id])
                return
            else:
                # split internal node
                total = len(parent.keys)
                mid = (total-1) // 2  # promote parent.keys[mid]
                promoted = parent.keys[mid]
                left_keys = parent.keys[:mid]
                right_keys = parent.keys[mid+1:]
                left_children = parent.children[:mid+1]
                right_children = parent.children[mid+1:]

                new_internal = BPlusNode(leaf=False)
                new_internal.keys = right_keys
                new_internal.children = right_children

                # mutate parent to become left node
                parent.keys = left_keys
                parent.children = left_children

                yield self._make_step("split_internal", f"Internal node {parent.id} overflowed — split into {parent.id} and {new_internal.id}. Promote {promoted}.",
                                      detailed=f"Left keys: {parent.keys} Right keys: {new_internal.keys}",
                                      pseudocode_line=13,
                                      highlight_nodes=[parent.id, new_internal.id],
                                      highlight_keys=[(new_internal.id, 0)],
                                      meta={"promoted_key": promoted})

                # prepare to propagate upwards
                key_to_insert = promoted
                child_left = parent
                child_right = new_internal
                parent_index -= 1
                if parent_index < 0:
                    # we split the root's child -> create a new root
                    new_root = BPlusNode(leaf=False)
                    new_root.keys = [key_to_insert]
                    new_root.children = [child_left, child_right]
                    self.root = new_root
                    yield self._make_step("new_root_internal", f"Create new root {new_root.id} with key {key_to_insert}.",
                                          detailed="Updated root after internal split.",
                                          pseudocode_line=14,
                                          highlight_nodes=[new_root.id],
                                          highlight_keys=[(new_root.id, 0)])
                    return
                # else loop to insert promoted into higher parent
        # End while
        return

    # -----------------------
    # Delete Steps
    # -----------------------
    def _find_subtree_min(self, node: BPlusNode):
        """Return the smallest key in the subtree rooted at `node` (or None)."""
        if node is None:
            return None
        cur = node
        while not cur.leaf:
            if not cur.children:
                return None
            cur = cur.children[0]
        return cur.keys[0] if cur.keys else None

    def _node_contains(self, node: BPlusNode, target: BPlusNode) -> bool:
        """Return True if `target` is contained (descendant) in subtree `node`."""
        if node is target:
            return True
        if node.leaf:
            return False
        for c in node.children:
            if self._node_contains(c, target):
                return True
        return False

    def _replace_in_ancestors(self, removed_value: int, leaf: BPlusNode):
        """
        Walk the path from root down to the leaf and update any internal separator
        keys equal to `removed_value`. Returns list of changes [(parent_id, old, new), ...].
        """
        changes = []
        if not self.root or not leaf or not leaf.leaf:
            return changes

        # Find path: list of parent nodes from root down to the parent of `leaf`.
        path_parents = []
        node = self.root
        # If root is the leaf itself, nothing to update
        if node is leaf:
            return changes

        while node is not leaf:
            found = False
            # Search child that contains the leaf
            for i, child in enumerate(node.children):
                if self._node_contains(child, leaf):
                    # store parent node (node)
                    path_parents.append(node)
                    node = child
                    found = True
                    break
            if not found:
                # cannot locate leaf from current root (shouldn't happen) -> bail
                return changes

        # For each parent in path_parents (bottom-up or top-down both work, use top-down),
        # replace any keys equal to removed_value by the min key of the right child.
        for parent in path_parents:
            # replace any occurrences of removed_value in this parent
            for j, k in enumerate(list(parent.keys)):  # copy because we may mutate
                if k == removed_value:
                    # separator k corresponds to parent.children[j+1] (right subtree)
                    if j + 1 < len(parent.children):
                        right_child = parent.children[j + 1]
                        new_min = self._find_subtree_min(right_child)
                        if new_min is not None and new_min != k:
                            parent.keys[j] = new_min
                            changes.append((parent.id, k, new_min))
                        elif new_min is None:
                            # if right subtree has no keys (rare after merges), try left or skip
                            # safe fallback: remove key if parent becomes invalid — skip here
                            pass
        return changes



    def delete_steps(self, key) -> Generator[Step, None, Optional[bool]]:
        """
        Step-by-step delete with borrow/merge handling.
        Updated to recompute parents dynamically after merges so cascade merges don't leave nodes without parents.
        """
        if not self.root:
            yield self._make_step("delete_empty", f"Tree empty; cannot delete {key}.", detailed="No action.", pseudocode_line=0)
            return False

        # small helper to find current parent of a node and the child's index
        def find_parent_and_index(target: BPlusNode):
            """Return (parent_node, child_index) or (None, None) if target is root or not found."""
            if self.root is None or self.root is target:
                return None, None
            stack = [self.root]
            while stack:
                node = stack.pop()
                if not node.leaf:
                    for i, c in enumerate(node.children):
                        if c is target:
                            return node, i
                    # push children to search deeper
                    for c in reversed(node.children):
                        stack.append(c)
            return None, None

        # Find leaf (similar to search) but without relying on stored path for parent pointers
        path: List[BPlusNode] = []
        node = self.root
        while not node.leaf:
            path.append(node)
            decided_child_idx = None
            for idx, k in enumerate(node.keys):
                yield self._make_step("compare", f"Comparing {key} with node key {k} in {node.id}.",
                                    detailed=f"At internal node {node.id}.",
                                    pseudocode_line=1,
                                    highlight_nodes=[node.id],
                                    highlight_keys=[(node.id, idx)])
                if key < k:
                    decided_child_idx = idx
                    node = node.children[decided_child_idx]
                    yield self._make_step("descend", f"Traverse to child {decided_child_idx} of node {node.id}.",
                                        detailed=f"Following pointer.",
                                        pseudocode_line=2,
                                        highlight_nodes=[node.id])
                    break
            if decided_child_idx is None:
                decided_child_idx = len(node.keys)
                node = node.children[decided_child_idx]
                yield self._make_step("descend", f"Traverse to rightmost child {decided_child_idx}.",
                                    detailed="Following pointer to rightmost child.",
                                    pseudocode_line=2,
                                    highlight_nodes=[node.id])

        leaf = node
        path.append(leaf)
        # find key in leaf
        found_idx = None
        for idx, k in enumerate(leaf.keys):
            yield self._make_step("compare", f"Comparing delete key {key} with leaf key {k}.",
                                detailed=f"At leaf {leaf.id}.",
                                pseudocode_line=3,
                                highlight_nodes=[leaf.id],
                                highlight_keys=[(leaf.id, idx)])
            if k == key:
                found_idx = idx
                break

        if found_idx is None:
            yield self._make_step("not_found", f"Key {key} not found; deletion aborted.",
                                detailed="Reached leaf and key not present.",
                                pseudocode_line=4,
                                highlight_nodes=[leaf.id])
            return False

        # delete in leaf
        yield self._make_step("delete_leaf", f"Deleting key {key} from leaf {leaf.id} at index {found_idx}.",
                            detailed=f"Leaf before: {leaf.keys}",
                            pseudocode_line=5,
                            highlight_nodes=[leaf.id],
                            highlight_keys=[(leaf.id, found_idx)])
        leaf.keys.pop(found_idx)
        yield self._make_step("after_delete", f"Deleted {key}. Leaf keys now: {leaf.keys}",
                            detailed="Check underflow.",
                            pseudocode_line=6,
                            highlight_nodes=[leaf.id])
        replacements = self._replace_in_ancestors(key, leaf)
        if replacements:
            # prepare a human-friendly description and highlight the changed parents
            changed_parents = [p for (p, old, new) in replacements]
            desc = "Updated ancestor separators: " + ", ".join([f"{p}: {old} -> {new}" for (p, old, new) in replacements])
            yield self._make_step("update_parents", desc,
                                detailed="Replaced removed key in ancestor separator(s).",
                                pseudocode_line=6,
                                highlight_nodes=changed_parents,
                                meta={"replacements": replacements})

        # If root leaf and now empty -> make tree empty
        if leaf is self.root:
            if len(leaf.keys) == 0:
                self.root = None
                yield self._make_step("root_empty", "Tree became empty after deletion.",
                                    detailed="Root removed.",
                                    pseudocode_line=7)
            else:
                yield self._make_step("done_delete", "Deletion complete; root leaf ok.", pseudocode_line=8, highlight_nodes=[leaf.id])
            return True

        # If leaf has enough keys -> done
        # use min_keys invariant (min_children - 1) for minimum keys in a node
        min_keys = max(1, self.min_children)  # root can have fewer, but non-root must have at least this many
        if len(leaf.keys) >= min_keys:
            yield self._make_step("done_delete", f"Leaf {leaf.id} has enough keys after deletion; done.",
                                detailed=f"Leaf size {len(leaf.keys)} >= min {min_keys}.",
                                pseudocode_line=9,
                                highlight_nodes=[leaf.id])
            return True

        # Underflow -> try borrow or merge
        # We'll walk upward, but compute parent dynamically after each modification to avoid stale references
        current = leaf
        idx = 0 
        while True:
            if idx >0 :
                if len(current.keys) >= math.ceil(self.order/2 -1):
                    yield self._make_step("done_delete", f"current node {current.id} has enough keys after deletion; done.",
                                        detailed=f"current node size {len(leaf.keys)} >= min {min_keys}.",
                                        pseudocode_line=9,
                                        highlight_nodes=[current.id])
                    return True
            idx += 1
            parent, child_pos = find_parent_and_index(current)
            if parent is None:
                # current is root (after repeated merges) -> if root has no keys and has single child, shrink
                if current is self.root:
                    # if root internal with one child, collapse
                    if not current.leaf and len(current.keys) == 0 and len(current.children) == 1:
                        newroot = current.children[0]
                        self.root = newroot
                        yield self._make_step("root_shrink", f"Root had 0 keys; promote child {newroot.id} to root.",
                                            detailed=f"New root is {newroot.id}.",
                                            pseudocode_line=15,
                                            highlight_nodes=[newroot.id])
                    return True
                else:
                    # shouldn't happen but guard
                    yield self._make_step("error", "Parent not found during rebalance — unexpected state.",
                                        detailed="Invariant failure or corrupted path.",
                                        pseudocode_line=16,
                                        highlight_nodes=[current.id])
                    return False

            # compute siblings based on current parent children list
            left_sib = parent.children[child_pos - 1] if child_pos - 1 >= 0 else None
            right_sib = parent.children[child_pos + 1] if child_pos + 1 < len(parent.children) else None

            yield self._make_step("underflow", f"Node {current.id} underflows (size {len(current.keys)}). Check siblings for borrow/merge at parent {parent.id}.",
                                detailed=f"Parent {parent.id} children: {[c.id for c in parent.children]}",
                                pseudocode_line=10,
                                highlight_nodes=[current.id, parent.id])

            # Try borrow from left sibling (if leaf)
            if left_sib and len(left_sib.keys) > min_keys:
                # borrow rightmost from left_sib
                if current.leaf:
                    borrowed = left_sib.keys.pop(-1)
                    current.keys.insert(0, borrowed)
                    parent_key_idx = child_pos - 1
                    parent.keys[parent_key_idx] = current.keys[0]
                    yield self._make_step("borrow_left", f"Borrowed key {borrowed} from left sibling {left_sib.id} into {current.id}; updated parent key.",
                                        detailed=f"Left sibling now {left_sib.keys}; node now {current.keys}; parent {parent.id} keys {parent.keys}",
                                        pseudocode_line=11,
                                        highlight_nodes=[current.id, left_sib.id, parent.id],
                                        highlight_keys=[(left_sib.id, len(left_sib.keys)), (current.id, 0), (parent.id, parent_key_idx)],
                                        meta={"borrowed": borrowed})
                    return True
                else:
                    # internal-node borrow: move separator down and pull child's last pointer
                    borrowed_key = left_sib.keys.pop(-1)
                    borrowed_child = left_sib.children.pop(-1)
                    # promote parent's separator down to current
                    sep_idx = child_pos - 1
                    parent_sep = parent.keys[sep_idx]
                    current.keys.insert(0, parent_sep)
                    current.children.insert(0, borrowed_child)
                    # update parent's separator to borrowed_key
                    parent.keys[sep_idx] = borrowed_key
                    yield self._make_step("borrow_left_internal", f"Borrowed key/child from left internal sibling {left_sib.id} into {current.id}; updated parent key.",
                                        detailed=f"Left {left_sib.keys}; current {current.keys}; parent {parent.keys}",
                                        pseudocode_line=11,
                                        highlight_nodes=[current.id, left_sib.id, parent.id])
                    return True

            # Try borrow from right sibling
            if right_sib and len(right_sib.keys) > min_keys:
                if current.leaf:
                    borrowed = right_sib.keys.pop(0)
                    current.keys.append(borrowed)
                    parent_key_idx = child_pos
                    parent.keys[parent_key_idx] = right_sib.keys[0] if right_sib.keys else borrowed
                    yield self._make_step("borrow_right", f"Borrowed key {borrowed} from right sibling {right_sib.id} into {current.id}; updated parent key.",
                                        detailed=f"Right sibling now {right_sib.keys}; node now {current.keys}; parent keys {parent.keys}",
                                        pseudocode_line=12,
                                        highlight_nodes=[current.id, right_sib.id, parent.id],
                                        highlight_keys=[(right_sib.id, 0), (current.id, len(current.keys)-1), (parent.id, parent_key_idx)],
                                        meta={"borrowed": borrowed})
                    return True
                else:
                    # internal node borrow from right sibling
                    borrowed_key = right_sib.keys.pop(0)
                    borrowed_child = right_sib.children.pop(0)
                    # move parent's separator down
                    sep_idx = child_pos
                    parent_sep = parent.keys[sep_idx]
                    current.keys.append(parent_sep)
                    current.children.append(borrowed_child)
                    parent.keys[sep_idx] = borrowed_key
                    yield self._make_step("borrow_right_internal", f"Borrowed key/child from right internal sibling {right_sib.id} into {current.id}; updated parent key.",
                                        detailed=f"Right {right_sib.keys}; current {current.keys}; parent {parent.keys}",
                                        pseudocode_line=12,
                                        highlight_nodes=[current.id, right_sib.id, parent.id])
                    return True

            # Cannot borrow -> merge with a sibling. Prefer left if exists, else right.
            if right_sib:
                # merge right_sib into current (current <- current + right)
                yield self._make_step("merge", f"Merging right sibling {right_sib.id} into node {current.id}. Remove parent separator key.",
                                    detailed=f"Left {current.keys} + Right {right_sib.keys}",
                                    pseudocode_line=13,
                                    highlight_nodes=[current.id, right_sib.id, parent.id])
                if current.leaf:
                    current.keys.extend(right_sib.keys)
                    current.next = right_sib.next
                else:
                    sep_idx = child_pos
                    separator = parent.keys[sep_idx]
                    current.keys.append(separator)
                    current.keys.extend(right_sib.keys)
                    current.children.extend(right_sib.children)
                sep_idx = child_pos
                removed_key = parent.keys.pop(sep_idx)
                parent.children.pop(child_pos + 1)
                yield self._make_step("after_merge", f"After merge node {current.id} keys: {current.keys}. Parent {parent.id} removed key {removed_key}.",
                                    detailed=f"Parent keys now: {parent.keys}",
                                    pseudocode_line=14,
                                    highlight_nodes=[current.id, parent.id],
                                    meta={"removed_parent_key": removed_key})

                if parent is self.root and len(parent.keys) == 0:
                    self.root = current
                    yield self._make_step("root_shrink", "Parent was root and is empty — promote child to root.",
                                        detailed=f"New root is {current.id}.",
                                        pseudocode_line=15,
                                        highlight_nodes=[current.id])
                    return True

                current = parent
                continue
            elif left_sib:
                # merge current into left_sib (left <- left + current)
                yield self._make_step("merge", f"Merging node {current.id} into left sibling {left_sib.id}. Remove parent separator key.",
                                    detailed=f"Left {left_sib.keys} + Right {current.keys}",
                                    pseudocode_line=13,
                                    highlight_nodes=[current.id, left_sib.id, parent.id])
                if current.leaf:
                    # leaf merge: extend keys and fix next pointer
                    left_sib.keys.extend(current.keys)
                    left_sib.next = current.next
                else:
                    # internal merge: move separator from parent down, then append keys and children
                    sep_idx = child_pos - 1
                    separator = parent.keys[sep_idx]
                    left_sib.keys.append(separator)
                    left_sib.keys.extend(current.keys)
                    left_sib.children.extend(current.children)
                # remove separator and child reference from parent
                sep_idx = child_pos - 1
                removed_key = parent.keys.pop(sep_idx)
                parent.children.pop(child_pos)
                yield self._make_step("after_merge", f"After merge left sibling {left_sib.id} keys: {left_sib.keys}. Parent {parent.id} removed key {removed_key}.",
                                    detailed=f"Parent keys now: {parent.keys}",
                                    pseudocode_line=14,
                                    highlight_nodes=[left_sib.id, parent.id],
                                    meta={"removed_parent_key": removed_key})
                # after merges, ensure parent separators are updated for the merged node(s)
                # if you have a `current` or `left_sib` node that now holds combined keys, call:
                # self._replace_in_ancestors(old_separator_value, current)
                # but in merges it's often simpler to call with the merged node's new first key:
                # e.g., new_first = current.keys[0] (or left_sib.keys[0]); then:
                # self._replace_in_ancestors(old_value, merged_node)


                # If parent was root and now empty -> promote merged node
                if parent is self.root and len(parent.keys) == 0:
                    self.root = left_sib
                    yield self._make_step("root_shrink", "Parent was root and is empty — promote child to root.",
                                        detailed=f"New root is {left_sib.id}.",
                                        pseudocode_line=15,
                                        highlight_nodes=[left_sib.id])
                    return True

                # continue upwards: set current = parent (we must recompute its parent on next loop)
                current = parent
                # continue loop to check parent's underflow; loop recomputes parent via find_parent_and_index
                continue

            else:
                # No sibling -> shouldn't happen for B+ with proper invariants
                yield self._make_step("error", "No siblings to borrow/merge — unexpected state.",
                                    detailed="Invariants violated.",
                                    pseudocode_line=16,
                                    highlight_nodes=[parent.id])
                return False
    # end while


    # -----------------------
    # Convenience wrappers
    # -----------------------
    def insert(self, key, value=None):
        for _ in self.insert_steps(key, value):
            pass

    def delete(self, key):
        for _ in self.delete_steps(key):
            pass

    def search(self, key) -> bool:
        for step in self.search_steps(key):
            pass

# -----------------------
# Visualization utilities (Graphviz DOT string builder)
# -----------------------
def render_tree_dot(snapshot: Dict[str, Any], highlight_nodes: List[str], highlight_keys: List[Tuple[str, int]]) -> str:
    nodes = snapshot["nodes"]
    root_id = snapshot["root"]
    dot_lines = [
        "digraph BPlus {",
        "  graph [rankdir=TB, splines=true];",
        "  node [shape=plaintext];"
    ]

    # map of highlighted key indices
    highlight_map = {}
    for node_id, key_idx in highlight_keys:
        highlight_map.setdefault(node_id, set()).add(key_idx)

    # draw nodes
    for nid, data in nodes.items():
        keys = data["keys"]
        is_leaf = data["leaf"]
        bgcolor = "#b9f6ca" if is_leaf else "#b3e5fc"
        outer_bg = "#fff59d" if nid in highlight_nodes else bgcolor

        cell_tds = []
        for i, k in enumerate(keys):
            cell_bg = "#ffcdd2" if (nid in highlight_map and i in highlight_map[nid]) else "white"
            cell_tds.append(f'<td port="k{i}" bgcolor="{cell_bg}" border="0" cellspacing="0" cellpadding="6">{html.escape(str(k))}</td>')
        cell_tds_html = "".join(cell_tds) if cell_tds else '<td bgcolor="white" cellpadding="6"> </td>'

        label = f'''<
            <table BORDER="0" CELLBORDER="1" CELLSPACING="0">
              <tr><td bgcolor="{outer_bg}" colspan="{max(1, len(keys))}"><b>{nid}{' (L)' if is_leaf else ''}</b></td></tr>
              <tr>{cell_tds_html}</tr>
            </table>
        >'''
        dot_lines.append(f'  {nid} [label={label}];')

    # edges for internal nodes
    for nid, data in nodes.items():
        if not data["leaf"]:
            for child_id in data["children"]:
                dot_lines.append(f'  {nid} -> {child_id} [arrowhead=none];')

    # dashed arrows between leaves
    for nid, data in nodes.items():
        if data["leaf"] and data.get("next"):
            dot_lines.append(f'  {nid} -> {data["next"]} [style=dashed, color="#616161", constraint=false, arrowhead=normal];')

    # --- NEW: keep sibling leaves horizontally aligned if no parent exists yet ---
    leaves = [nid for nid, d in nodes.items() if d["leaf"]]
    if len(leaves) >= 2:
        # if there are two or more leaves with no internal parent yet (root is leaf or None)
        if not root_id or nodes[root_id]["leaf"]:
            dot_lines.append(f'  {{ rank=same; {" ".join(leaves)}; }}')

    # emphasize root rank
    if root_id:
        dot_lines.append(f'  {{ rank=source; {root_id}; }}')

    dot_lines.append("}")
    return "\n".join(dot_lines)


# -----------------------
# Pseudocode snippets (simple educational ones)
# -----------------------
INSERT_PSEUDOCODE = [
    "1. if root is None: create leaf with key",
    "2. else: traverse from root, comparing keys to choose child",
    "3. when reach leaf: insert key in sorted position",
    "4. if leaf overflows: split leaf, promote smallest key of right node",
    "5. insert promoted key to parent; if parent overflows split and propagate",
    "6. if root must split: create new root",
    "7. finish"
]

SEARCH_PSEUDOCODE = [
    "1. if root is None: return not found",
    "2. start at root, compare with each key",
    "3. if key < node.key: follow left pointer; else continue",
    "4. when reach leaf, scan keys sequentially",
    "5. if found return true else false"
]

DELETE_PSEUDOCODE = [
    "1. find leaf containing key",
    "2. delete key from leaf",
    "3. if leaf underflows: try borrow from sibling",
    "4. if borrow not possible: merge with sibling and remove parent separator",
    "5. propagate underflow up if parent underflows",
    "6. if root becomes empty: shrink tree"
]

# -----------------------
# Streamlit App UI
# -----------------------
st.set_page_config(page_title="B+ Tree Visualizer", layout="wide")
st.title("📚 B+ Tree Step-by-Step Visualizer")
st.markdown("""
This interactive app visualizes **B+ tree** insertion, deletion, and search *step-by-step*.
Use the controls on the right to perform operations. Use **Next** / **Previous** to move between conceptual steps.
""")

# Initialize session state
if "tree" not in st.session_state:
    st.session_state["tree"] = BPlusTree(order=4)
if "steps" not in st.session_state:
    st.session_state["steps"] = []  # list[Step]
if "step_index" not in st.session_state:
    st.session_state["step_index"] = 0
if "mode" not in st.session_state:
    st.session_state["mode"] = None
if "log" not in st.session_state:
    st.session_state["log"] = []

# Sidebar controls
# -----------------------
# Sidebar controls (REPLACE the original sidebar block with this)
# -----------------------
with st.sidebar:
    st.header("Controls")

    # group the main inputs into a form so pressing Enter submits
    with st.form("op_form"):
        order = st.selectbox("Tree order (m)",
                             options=[3, 4, 5, 6, 7, 8],
                             index=1,
                             help="Order m: max children per internal node. Max keys per node = m-1",
                             key="order_select")
        step_granularity = st.selectbox("Step granularity", options=["fine", "coarse"], index=0, key="granularity_select")
        operation = st.selectbox("Operation", options=["Insert", "Delete", "Search"], index=0, key="operation_select")

        # these text_inputs are inside the form; pressing Enter will submit the form
        key_input = st.text_input("Key (integer)", value="", key="key_input_form")
        value_input = st.text_input("Value (optional)", value="", key="value_input_form")

        # single submit button for the form — pressing Enter will trigger this
        submitted = st.form_submit_button("Execute")

    # handle the submitted form (Enter or click)
    if submitted:
        try:
            k = int(st.session_state["key_input_form"].strip())
        except Exception:
            st.error("Please enter an integer key.")
        else:
            # if order changed, reset tree
            if st.session_state["tree"].order != order:
                st.session_state["tree"] = BPlusTree(order=order)
                st.session_state["log"].append(f"Order changed to {order} — tree reset.")

            if operation == "Insert":
                st.session_state["mode"] = "insert"
                gen = st.session_state["tree"].insert_steps(k, st.session_state["value_input_form"] or None)
                st.session_state["steps"] = list(gen)
                st.session_state["step_index"] = 0
                st.session_state["log"].append(f"Insert requested: {k}")
            elif operation == "Delete":
                st.session_state["mode"] = "delete"
                gen = st.session_state["tree"].delete_steps(k)
                st.session_state["steps"] = list(gen)
                st.session_state["step_index"] = 0
                st.session_state["log"].append(f"Delete requested: {k}")
            elif operation == "Search":
                st.session_state["mode"] = "search"
                gen = st.session_state["tree"].search_steps(k)
                st.session_state["steps"] = list(gen)
                st.session_state["step_index"] = 0
                st.session_state["log"].append(f"Search requested: {k}")

    st.markdown("---")

    # Navigation (keep these outside the form so they aren't triggered by Enter)
    nav_cols = st.columns([1, 1, 1])
    if nav_cols[0].button("Previous"):
        if st.session_state["steps"]:
            st.session_state["step_index"] = max(0, st.session_state["step_index"] - 1)
    if nav_cols[1].button("Next"):
        if st.session_state["steps"]:
            st.session_state["step_index"] = min(len(st.session_state["steps"]) - 1, st.session_state["step_index"] + 1)
    if nav_cols[2].button("Reset"):
        st.session_state["tree"] = BPlusTree(order=order)
        st.session_state["steps"] = []
        st.session_state["step_index"] = 0
        st.session_state["mode"] = None
        st.session_state["log"].append("Tree reset.")

    st.markdown("---")
    if st.button("Load demo sequence (insert sample keys)"):
        demo_keys = [10, 20, 5, 6, 12, 30, 7, 17]
        st.session_state["tree"] = BPlusTree(order=order)
        st.session_state["log"].append("Loaded demo; inserting sample keys: " + str(demo_keys))
        for dk in demo_keys:
            for _ in st.session_state["tree"].insert_steps(dk):
                pass
        st.session_state["steps"] = []
        st.session_state["step_index"] = 0

    if st.button("Validate invariants"):
        ok, errs = st.session_state["tree"].validate()
        if ok:
            st.success("B+ tree invariants hold.")
            st.session_state["log"].append("Validation: OK")
        else:
            st.error("Invariants violated: see console")
            st.session_state["log"].append("Validation: ERR - " + "; ".join(errs))

    st.markdown("---")
    st.write("Operation Log (latest first):")
    for entry in reversed(st.session_state["log"][-30:]):
        st.write(f"- {entry}")


# -----------------------
# Main visualization and explanation area
# -----------------------
left_col, right_col = st.columns([2, 1])
with left_col:
    st.subheader("Tree visualization")
    if st.session_state["steps"]:
        step = st.session_state["steps"][st.session_state["step_index"]]
        dot = render_tree_dot(step.tree_snapshot, step.highlight_nodes, step.highlight_keys)
        st.graphviz_chart(dot)
        st.markdown(f"**Step {st.session_state['step_index']+1}/{len(st.session_state['steps'])}** — **{step.action}**")
        st.info(step.description)
    else:
        # nothing to show: render current tree snapshot
        snap = tree_to_snapshot(st.session_state["tree"].root) if st.session_state["tree"].root else {"root": None, "nodes": {}}
        dot = render_tree_dot(snap, [], [])
        st.graphviz_chart(dot)
        st.write("No active operation. Use the controls to create steps (Insert / Delete / Search).")

with right_col:
    st.subheader("Explanation & Pseudocode")
    if st.session_state["steps"]:
        step = st.session_state["steps"][st.session_state["step_index"]]
        st.markdown(f"**Action:** {step.action}")
        st.write(step.detailed_explanation or step.description)
        st.markdown("---")
        # pseudocode lines based on mode
        if st.session_state["mode"] == "insert":
            pseudocode = INSERT_PSEUDOCODE
        elif st.session_state["mode"] == "delete":
            pseudocode = DELETE_PSEUDOCODE
        elif st.session_state["mode"] == "search":
            pseudocode = SEARCH_PSEUDOCODE
        else:
            pseudocode = []

        # render pseudocode with highlighted line if present
        for idx, line in enumerate(pseudocode):
            if step.pseudocode_line is not None and idx == step.pseudocode_line:
                st.markdown(f"<div style='background:#fff59d;padding:6px;border-radius:4px'><code>{html.escape(line)}</code></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<code>{html.escape(line)}</code>", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("Meta")
        st.json(step.meta)
    else:
        st.write("No step selected. Perform an operation to see step-by-step explanation.")
    st.markdown("---")
    st.subheader("Operation Log")
    for entry in reversed(st.session_state["log"][-50:]):
        st.write(f"- {entry}")

# -----------------------
# Footer: quick help + examples
# -----------------------
st.markdown("---")
st.markdown("**Usage tips**")
st.markdown("""
- Enter integer keys in the Key field.
- Use *Insert* / *Delete* / *Search* to start an operation — the app builds a sequence of conceptual steps.
- Move with **Next** / **Previous** only. This app intentionally provides manual, educational navigation.
- Use `Validate invariants` to check basic tree consistency.
""")
