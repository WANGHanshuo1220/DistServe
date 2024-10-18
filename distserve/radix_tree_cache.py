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
from enum import Enum
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, List, Optional

import torch
import ray
from distserve.logger import init_logger
from distserve.utils import BlockLocation

logger = init_logger(__name__)

# class BlockLocation(Enum):
#     """The location of a block"""

#     GPU = "gpu"
#     CPU = "cpu"

class TreeNode:
    """ TreeNode: containing maximum block_size tokens"""

    def __init__(self):
        self.children = defaultdict(TreeNode)
        self.parent: TreeNode = None
        self.key: List = None # token_ids
        self.value: int = None # block_id
        self.lock_ref = 0
        self.last_access_time = time.time()
        self.location = None
        self.is_root = False

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
        self.num_gpu_nodes = 0
        self.num_cpu_nodes = 0
        self.reset()

    ##### Public API #####

    def reset(self):
        self.num_gpu_nodes = 0
        self.num_cpu_nodes = 0
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.root_node.is_root = True
        self.evictable_size_ = 0

    def match_prefix(self, key: List, **kwargs):
        if self.disable:
            return [], self.root_node

        value = []
        last_node = [self.root_node]
        self._match_prefix_helper(self.root_node, key, value, last_node)
        return value, last_node[0]

    def insert(self, key: List, value:List):
        if self.disable:
            return None

        last_node = [self.root_node]
        self._insert_helper(self.root_node, key, value, last_node)
        return last_node[0]

    def cache_prefill_req(self, req, kv_indices: List[int]):
        """Cache request when it is unfinished. 
        
        This function should only be invoked by prefill requests???
        """

        if self.disable:
            return

        last_node = req.last_node
        token_ids = req.prompt_token_ids

        # Insert this req into tree_cache
        new_last_node = self.insert(token_ids, kv_indices.copy())

        self.inc_lock_ref(new_last_node)
        self.dec_lock_ref(last_node)

        req.last_node = new_last_node
        # req.prefix_indices = new_indices

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}, #nodes: {self.num_nodes}")

    def total_size(self):
        return self._total_size_helper(self.root_node)

    def total_gpu_nodes(self):
        return self._total_gpu_nodes_helper()

    def evict(self, num_nodes: int, evict_callback: Callable, location: BlockLocation):
        """ Evict num of GPU blocks(nodes)

        Args:
            num_nodes (int): num of nodes to evict
            evict_callback (Callable): block_manager_callback
            location (BlockLocation): indicate the location of blocks to be evicted
        
        Note:
            Deleted nodes will be stored on cpu memory, if cpu does
            not have enough blocks, then simply delete some without
            any migration
        """
        if self.disable:
            return

        # Second: Find to-delete gpu blocks
        
        # Third: Swap these blocks from gpu to cpu 

        # Finally: Set both gpu and cpu metadata correctly

        leaves = self._collect_leaves(location)
        heapq.heapify(leaves)

        def add_leaves_(node: TreeNode):
            p_node = node.parent
            if location == BlockLocation.CPU:
                if len(p_node.children) == 0 and p_node.location == location:
                    heapq.heappush(leaves, p_node)
            else:
                for child in p_node.children.values():
                    if child.location == BlockLocation.GPU:
                        return
                heapq.heappush(leaves, p_node)

        num_evicted = 0
        to_evict_nodes = []
        while num_evicted < num_nodes and len(leaves):
            node = heapq.heappop(leaves)

            if node == self.root_node:
                break
            if node.lock_ref > 0:
                continue

            # Some callback operations
            to_evict_nodes.append(node)

            num_evicted += 1
            if location == BlockLocation.CPU:
                self._delete_leaf(node)
            else:
                self._swap_gpu_leaf(node)

            # Add leaves
            add_leaves_(node)
        
        evict_callback(to_evict_nodes, location)

        if location == BlockLocation.GPU:
            self.num_gpu_nodes -= num_evicted
        else:
            self.num_cpu_nodes -= num_evicted

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    ##### Internal Helper Functions #####

    def _match_prefix_helper(
        self, node: TreeNode, key: List, value, last_node: list[TreeNode]
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
                last_node[0] = child
                if len(key) > prefix_len:
                    self._match_prefix_helper(child, key[self.node_size:], value, last_node)
                else:
                    return

    def _insert_helper(self, node: TreeNode, key: List, value: List, last_node: list[TreeNode]):
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
                    return self._insert_helper(child, key, value, last_node)

        if len(key):
            assert (len(value) > 0)

            self.num_gpu_nodes += 1
            
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key[:self.node_size]
            new_node.value = value[0]
            new_node.location = BlockLocation.GPU
            
            node.children[search_key] = new_node
            self.evictable_size_ += len(new_node.key)

            last_node[0] = new_node

            if len(key) > self.node_size:
                self._insert_helper(new_node, key[self.node_size:], value[1:], last_node)

    def _print_helper(self, node: TreeNode, indent: int):
        for _, child in node.children.items():
            print(" " * indent, len(child.key), child.key, child.value, f"r={child.lock_ref}")
            self._print_helper(child, indent=indent + 2)

    def _delete_leaf(self, node: TreeNode):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _swap_gpu_leaf(self, node: TreeNode):
        """Swap gpu node to cpu logically"""
        node.location = BlockLocation.CPU

    def _total_size_helper(self, node: TreeNode):
        x = len(node.key)
        for child in node.children.values():
            x += self._total_size_helper(child)
        return x

    def _total_gpu_nodes_helper(self):
        return self.num_gpu_nodes

    def _collect_leaves(self, location: BlockLocation) -> List[TreeNode]:
        ret_list = []

        def is_cpu_leaves(cur_node: TreeNode) -> bool:
            if cur_node.location == BlockLocation.CPU and len(cur_node.children) == 0:
                return True
            else:
                return False

        def is_gpu_leaves(cur_node: TreeNode) -> bool:
            assert cur_node.location == BlockLocation.GPU
            if len(cur_node.children) == 0:
                return True
            else:
                for child_node in cur_node.children.values():
                    if child_node.location == BlockLocation.GPU:
                        return False
                return True

        def dfs_(cur_node: TreeNode, location: BlockLocation):
            if location == BlockLocation.CPU:
                if is_cpu_leaves(cur_node):
                    ret_list.append(cur_node)
            else:
                if is_gpu_leaves(cur_node):
                    ret_list.append(cur_node)
                    return

            for x in cur_node.children.values():
                dfs_(x, location)

        dfs_(self.root_node, location)
        return ret_list
