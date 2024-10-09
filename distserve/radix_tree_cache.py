from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
from distserve.logger import init_logger

logger = init_logger(__name__)

class TreeNode:
    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent = None
        self.key = None
        self.value = None
        self.lock_ref = 0
        self.last_access_time = time.time()

    def __lt__(self, other: "TreeNode"):
        return self.last_access_time < other.last_access_time


def _key_match(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


class RadixCache():
    def __init__(
        self,
        disable: bool = False,
    ):
        self.disable = disable
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0

    def match_prefix(self, key: List, **kwargs):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        if value:
            value = sum(value, [])
        return value, last_node[0]

    def insert(self, key: List, value=None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value)

    def cache_prefill_req(self, last_node: TreeNode, token_ids: List[int], kv_indices: List[int]):
        """Cache request when it is unfinished. 
        
        This function should only be invoked by prefill requests???
        """

        if self.disable:
            return

        assert (token_ids is not None), "Req token_id is None"

        # Insert this req into tree_cache
        new_prefix_len = self.insert(token_ids, kv_indices)

        new_indices, new_last_node = self.match_prefix(key=token_ids)
        assert len(new_indices) == len(token_ids), f"{len(new_indices)}, {len(token_ids)}"

        self.dec_lock_ref(last_node)
        self.inc_lock_ref(new_last_node)
        # req.prefix_indices = new_indices
        # req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def evict(self, num_tokens: int, evict_callback: Callable):
        if self.disable:
            return

        leaves = self._collect_leaves()
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue

            evict_callback(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0:
                heapq.heappush(leaves, x.parent)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: TreeNode
    ):
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)
            if prefix_len < len(child.key):
                # Split original node to two nodes in order to 
                # align prefill prompt to tree node
                #
                # Note that the difference with <insert> method is that when 
                # there is a node mismatch <insert> method will not only split
                # original node but also create a new node to store the new kv.
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                last_node[0] = new_node
            else:
                value.append(child.value)
                last_node[0] = child
                self._match_prefix_helper(child, key[prefix_len:], value, last_node)

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {key[split_len:][0]: child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[key[:split_len][0]] = new_node
        return new_node

    def _insert_helper(self, node: TreeNode, key: List, value):
        """ Insert a seq into tree_cache

        Args:
            node (TreeNode): default is root node, but it's recursive
            key (List): token_ids
            value (List): kv indices of tokens

        Returns:
            int: prefix_match_length

        Recursively insertion. Assign keys(token_ids) and values(kv indices) to 
        tree_nodes, and split tree_node is necessary.
        """
        node.last_access_time = time.time()
        if len(key) == 0:
            return 0

        # Start from checking first token_id in key 
        if key[0] in node.children.keys():
            child = node.children[key[0]]
            prefix_len = _key_match(child.key, key)

            # prefix_len == len(child.key) means this child node is perfectly matched
            if prefix_len == len(child.key):
                # prefix_len == len(key) means they are perfectly aligned and 
                # child node doesn't need to split
                if prefix_len == len(key):
                    return prefix_len
                else:
                    # len(key) > len(child.key) && prefix_len == len(child.key)
                    # need to recursively insert to child's child node
                    key = key[prefix_len:]
                    value = value[prefix_len:]
                    return prefix_len + self._insert_helper(child, key, value)

            # prefix_len < len(child.key), so child node needs to split to match prefix
            new_node = self._split_node(child.key, child, prefix_len)
            return prefix_len + self._insert_helper(
                new_node, key[prefix_len:], value[prefix_len:]
            )

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[key[0]] = new_node
            self.evictable_size_ += len(value)
        return 0

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key[:10], f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.value)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _collect_leaves(self):
        ret_list = []

        def dfs_(cur_node):
            if len(cur_node.children) == 0:
                ret_list.append(cur_node)

            for x in cur_node.children.values():
                dfs_(x)

        dfs_(self.root_node)
        return ret_list
