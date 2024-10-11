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
    """ TreeNode: containing maximum block_size tokens"""

    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: List = None # token_ids
        self.value: int = None # block_id
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
        block_size: int = 16,
    ):
        self.disable = disable
        self.node_size = block_size
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
        last_node = self.root_node
        self._match_prefix_helper(self.root_node, key, value, last_node)
        return value, last_node

    def insert(self, key: List, value:List):
        if self.disable:
            return 0

        return self._insert_helper(self.root_node, key, value)

    def cache_prefill_req(self, last_node: TreeNode, token_ids: List[int], kv_indices: List[int]):
        """Cache request when it is unfinished. 
        
        This function should only be invoked by prefill requests???
        """

        if self.disable:
            return

        assert (token_ids is not None), "Req token_id is None"

        # Insert this req into tree_cache
        self.insert(key=token_ids, value=kv_indices)

        new_indices, new_last_node = self.match_prefix(key=token_ids)

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

        search_key = ''.join(map(str, key[:self.node_size]))
        if search_key in node.children.keys():
            child = node.children[search_key]
            prefix_len = _key_match(child.key, key)
            if prefix_len == self.node_size:
                value.append(child.value)
                last_node = child
                if len(key) > prefix_len:
                    self._match_prefix_helper(child, key[self.node_size:], value, last_node)

    def _insert_helper(self, node: TreeNode, key: List, value: List):
        """ Insert a seq into tree_cache

        Args:
            node (TreeNode): default is root node, but it's recursive
            key (List): token_ids
            value (List): kv indices of tokens

        Returns:
            int: prefix_match_length

        Recursively insertion. Store keys(token_ids) and values(kv indices) to 
        tree_nodes, tree node size = block size.
        """
        node.last_access_time = time.time()
        if len(key) == 0:
            return

        search_key = ''.join(map(str, key[:self.node_size]))
        # Start from checking first token_id in key 
        if search_key in node.children.keys():
            child = node.children[search_key]
            prefix_len = _key_match(child.key, key)

            # If matched prefix length == node_size (block_size)
            if prefix_len == self.node_size:
                # prefix_len == len(key) means they are perfectly aligned and 
                # child node doesn't need to split
                if prefix_len == len(key):
                    return
                else:
                    # len(key) > len(child.key) && prefix_len == len(child.key)
                    # need to recursively insert to child's child node
                    assert (len(key) > prefix_len)
                    key = key[self.node_size:]
                    value = value[1:]
                    return self._insert_helper(child, key, value)

        if len(key):
            assert (len(value) > 0)
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key[:self.node_size]
            new_node.value = value[0]
            
            node.children[search_key] = new_node
            self.evictable_size_ += len(new_node.key)

            if len(key) > self.node_size:
                self._insert_helper(new_node, key[self.node_size:], value[1:])

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key, child.value, f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self, node: TreeNode):
        x = len(node.key)
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
