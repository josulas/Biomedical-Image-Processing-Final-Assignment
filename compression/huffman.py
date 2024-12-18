from typing import Iterable, Any
from numbers import Real
from collections import OrderedDict, defaultdict
from numpy.typing import ArrayLike, NDArray
import numpy as np
from bitarray import bitarray


class HuffmanNode:
    def __init__(self, symbol, weight):
        self.symbol = symbol
        self.weight = weight
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.weight < other.weight
    def __str__(self):
        if self.symbol is not None:
            return f"({self.symbol}, {self.weight})"
        else:
            return f"(({self.left.__str__()}, {self.right.__str__()}), {self.weight})"
    def __rpr__(self):
        return self.__str__()


def get_histogram(elements: Iterable) -> dict[Real, int]:
    histogram = {}
    for element in elements:
        histogram[element] = histogram.get(element, 0) + 1
    return histogram


def sort_dict(dictionary: dict, **kwargs) -> OrderedDict:
    return OrderedDict(sorted(dictionary.items(), key=lambda item: item[1]), **kwargs)


def get_node_depth(node: HuffmanNode, depth=0) -> int:
    if node.symbol is not None:
        return depth
    else:
        return max(get_node_depth(node.left, depth + 1), get_node_depth(node.right, depth + 1))


def get_smallest_nodes(list_pair: list) -> list[HuffmanNode]:
    smallest = []
    # Choosing the first list when there is a tie ensures the most balanced tree
    for _ in range(2):
        if not len(list_pair[1]):
            smallest.append(list_pair[0].pop(0))
        elif not len(list_pair[0]):
            smallest.append(list_pair[1].pop(0))
        elif list_pair[1][0] < list_pair[0][0]:
            smallest.append(list_pair[1].pop(0))
        else:
            smallest.append(list_pair[0].pop(0))
    return smallest


def build_huffman_tree(histogram: OrderedDict) -> HuffmanNode:
    list_pair = [[HuffmanNode(symbol, weight) for symbol, weight in histogram.items()], []]
    if len(list_pair[0]) == 1:
        return list_pair[0][0]
    while len(list_pair[0]) + len(list_pair[1]) > 1:
        left_node, right_node = get_smallest_nodes(list_pair)
        new_node = HuffmanNode(None, left_node.weight + right_node.weight)
        new_node.left = left_node
        new_node.right = right_node
        list_pair[1].append(new_node)
    if not len(list_pair[0]) == 0 or not len(list_pair[1]) == 1:
        raise NotImplementedError('Huffman Tree construction did not run as expected')
    return list_pair[1][0]


def generate_code_from_tree(root: HuffmanNode) -> defaultdict:
    def get_code_from_tree(node: HuffmanNode, prefix=bitarray(), codebook=None):
        if codebook is None:
            codebook = defaultdict(bitarray)
        if node.symbol is not None:  # Leaf node
            codebook[node.symbol] = prefix
        else:
            get_code_from_tree(node.left, prefix + bitarray('0'), codebook)
            get_code_from_tree(node.right, prefix + bitarray('1'), codebook)
        return codebook
    if root.symbol is not None:
        codebook = defaultdict(bitarray)
        codebook[root.symbol] = bitarray('0')
        return codebook
    return get_code_from_tree(root)


def generate_tree_from_code(codebook: defaultdict):
    root = HuffmanNode(None, None)
    for symbol, code in codebook.items():
        node = root
        for bit in code:
            if bit == 0:
                if node.left is None:
                    node.left = HuffmanNode(None, None)
                node = node.left
            else:
                if node.right is None:
                    node.right = HuffmanNode(None, None)
                node = node.right
        node.symbol = symbol
    return root


def encode_data_with_codebook(data: Iterable, codebook: defaultdict) -> bitarray:
    encoded_data = bitarray()
    for symbol in data:
        encoded_data += codebook[symbol]
    return encoded_data


def get_encoded_list(data: Iterable, codebook: defaultdict) -> list:
    encoded_list = []
    for symbol in data:
        encoded_list.append(codebook[symbol])
    return encoded_list


def get_huffman_tree(array: Iterable) -> HuffmanNode:
    sorted_histogram = sort_dict(get_histogram(array))
    huffman_tree = build_huffman_tree(sorted_histogram)
    return huffman_tree


def get_abstract_tree(root: HuffmanNode) -> list[tuple[Any, int]]:
    symbols_and_code_lengths = []
    def abstract_tree(node: HuffmanNode, code_length=0):
        if node.symbol is not None:
            symbols_and_code_lengths.append((node.symbol, code_length))
        else:
            abstract_tree(node.left, code_length + 1)
            abstract_tree(node.right, code_length + 1)
    if root.symbol is not None:
        symbols_and_code_lengths.append((root.symbol, 1))
        symbols_and_code_lengths.append((root.symbol + 1, 1))
    else:
        abstract_tree(root)
    return sorted(symbols_and_code_lengths, key=lambda x: x[1])


def limit_huffman_code_length(sorted_symbol_code_lengths: list[tuple[Any, int]],
                              max_code_length: int = 16) -> list[tuple[Any, int]]:
    def get_tree_weight(length_counts: list[int]):
        tree_weight = 0
        for count_index, count in enumerate(length_counts):
            tree_weight += count * 2 ** (max_code_length - count_index - 1) # A 1 bit code halves the tree capacity
        return tree_weight
    def get_lengths(length_counts: list[int]) -> list[int]:
        lengths = []
        for count_index, count in enumerate(length_counts):
            if count:
                lengths.extend([count_index + 1] * count)
        return lengths
    max_length = sorted_symbol_code_lengths[-1][1]
    if max_length > max_code_length:
        valid_lengths_counts = [0 for _ in range(max_code_length)]
        max_tree_weight = 2 ** max_code_length
        actual_lengths_counts = [0 for _ in range(max_length)]
        for _, length in sorted_symbol_code_lengths:
            actual_lengths_counts[length - 1] += 1
        for length_index, actual_length_count in enumerate(actual_lengths_counts):
            if length_index >= max_code_length:
                valid_lengths_counts[max_code_length - 1] += actual_length_count
            else:
                valid_lengths_counts[length_index] += actual_length_count
        while get_tree_weight(valid_lengths_counts) > max_tree_weight:
            length_index = max_code_length - 2
            while not actual_lengths_counts[length_index]:
                length_index -= 1
            valid_lengths_counts[length_index] -= 1
            valid_lengths_counts[length_index + 1] += 1
        new_lengths = get_lengths(valid_lengths_counts)
        new_symbol_code_lengths = []
        for new_length, (symbol, _) in zip(new_lengths, sorted_symbol_code_lengths):
            new_symbol_code_lengths.append((symbol, new_length))
        return new_symbol_code_lengths
    else:
        return sorted_symbol_code_lengths


def index_to_bit_array(unsigned_int: int, length: int) -> bitarray:
    bitarray_ = bitarray('0') * length
    for position_index in range(length - 1, -1, -1):
        bitarray_[position_index] = bool(unsigned_int % 2)
        unsigned_int //= 2
    return bitarray_


def build_sorted_codebook_from_abstraction(abstract_tree: list[tuple[Any, int]]) -> defaultdict:
    sorted_abstract_tree = sorted(abstract_tree, key=lambda el: el[1])
    codebook = defaultdict(bitarray)
    max_length = sorted_abstract_tree[-1][1]
    raw_code = 0
    for (symbol, code_length) in sorted_abstract_tree:
        code = raw_code // 2 ** (max_length - code_length)
        bitcode = index_to_bit_array(code, code_length)
        codebook[symbol] = bitcode
        raw_code += 2 ** (max_length - code_length)
    return codebook


def huffman_encode(array: Iterable) -> tuple[HuffmanNode, bitarray]:
    huffman_tree = get_huffman_tree(array)
    code = generate_code_from_tree(huffman_tree)
    encoded = encode_data_with_codebook(array, code)
    return huffman_tree, encoded


def huffman_decode(huffman_tree: HuffmanNode, encoded_data: bitarray) -> list:
    decoded_data = []
    node = huffman_tree
    for bit in encoded_data:
        node = node.right if bit else node.left
        if node.symbol is not None:  # Leaf node
            decoded_data.append(node.symbol)
            node = huffman_tree
    return decoded_data


def huffman_encode_image(img: ArrayLike) -> tuple[tuple[int], HuffmanNode, bitarray]:
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    shape = img.shape
    huffman_tree, encoded_data = huffman_encode(img.flatten())
    return shape, huffman_tree, encoded_data


def huffman_decode_image(shape: tuple, huffman_tree: HuffmanNode, encoded_data: bitarray) -> NDArray:
    return np.array(huffman_decode(huffman_tree, encoded_data)).reshape(shape)