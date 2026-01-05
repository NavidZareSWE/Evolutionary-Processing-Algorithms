import random
from constants import TERMINALS, FUNCTIONS


class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def is_terminal(self):
        return self.value in TERMINALS

    def is_function(self):
        return self.value in FUNCTIONS

    def subtree_depth(self):
        if self.is_terminal():
            return 1
        return 1 + max(child.subtree_depth() for child in self.children)

    def subtree_size(self):
        if self.is_terminal():
            return 1
        return 1 + sum(child.subtree_size() for child in self.children)

    def copy(self):
        if self.is_terminal():
            return TreeNode(self.value)
        return TreeNode(self.value, [child.copy() for child in self.children])

    def __str__(self):
        if self.is_terminal():
            return self.value
        return f"({self.value} {' '.join(str(c) for c in self.children)})"

    def print_tree(self, indent=0):
        prefix = "  " * indent
        if self.is_terminal():
            print(f"{prefix}+-- {self.value}")
        else:
            print(f"{prefix}+-- {self.value}")
            if len(self.children) >= 2:
                print(f"{prefix}|   [IF TRUE]:")
                self.children[0].print_tree(indent + 2)
                print(f"{prefix}|   [IF FALSE]:")
                self.children[1].print_tree(indent + 2)


def generate_full_tree(max_depth, current_depth=0):
    if current_depth >= max_depth - 1:
        return TreeNode(random.choice(TERMINALS))
    else:
        func = random.choice(FUNCTIONS)
        children = [
            generate_full_tree(max_depth, current_depth + 1),
            generate_full_tree(max_depth, current_depth + 1)
        ]
        return TreeNode(func, children)


def generate_grow_tree(max_depth, current_depth=0):
    if current_depth >= max_depth - 1:
        return TreeNode(random.choice(TERMINALS))
    else:
        if random.random() < 0.3:  # 30% chance of early termination
            return TreeNode(random.choice(TERMINALS))
        else:
            func = random.choice(FUNCTIONS)
            children = [
                generate_grow_tree(max_depth, current_depth + 1),
                generate_grow_tree(max_depth, current_depth + 1)
            ]
            return TreeNode(func, children)


def get_all_nodes(tree, include_root=True):
    # Returns list of tuples: (node, parent, child_index)
    nodes = []

    def collect_nodes_preorder(node, parent, child_index):
        nodes.append((node, parent, child_index))
        for i, child in enumerate(node.children):
            collect_nodes_preorder(child, node, i)

    if include_root:
        collect_nodes_preorder(tree, None, -1)
    else:
        for i, child in enumerate(tree.children):
            collect_nodes_preorder(child, tree, i)

    return nodes
