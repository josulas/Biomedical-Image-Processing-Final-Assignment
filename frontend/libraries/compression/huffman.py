"""
Huffman coding implementation for data compression.
"""


from typing import Iterable, Any, Self
from numbers import Real
from collections import OrderedDict, defaultdict

from numpy.typing import ArrayLike, NDArray
import numpy as np
from bitarray import bitarray


class HuffmanNode:
    """A node in the Huffman tree."""
    def __init__(self, symbol, weight):
        self.symbol: Any = symbol
        self.weight: int = weight
        self.left: Self | None = None
        self.right: Self | None = None
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
    """Returns a histogram of the elements in the iterable."""
    histogram = {}
    for element in elements:
        histogram[element] = histogram.get(element, 0) + 1
    return histogram


def sort_dict(dictionary: dict, **kwargs) -> OrderedDict:
    """Sorts a dictionary by its values and returns an OrderedDict."""
    return OrderedDict(sorted(dictionary.items(), key=lambda item: item[1]), **kwargs)


def get_node_depth(node: HuffmanNode | None, depth=0) -> int:
    """Returns the depth of the node in the Huffman tree."""
    if node is None:
        return depth
    elif node.symbol is not None:
        return depth
    else:
        return max(get_node_depth(node.left, depth + 1), get_node_depth(node.right, depth + 1))


def get_smallest_nodes(list_pair: list[list]) -> list[HuffmanNode]:
    """Returns the two smallest nodes from the list_pair."""
    smallest = []
    # Choosing the first list when there is a tie ensures the most balanced tree
    for _ in range(2):
        if not list_pair[1]:
            smallest.append(list_pair[0].pop(0))
        elif not list_pair[0]:
            smallest.append(list_pair[1].pop(0))
        elif list_pair[1][0] < list_pair[0][0]:
            smallest.append(list_pair[1].pop(0))
        else:
            smallest.append(list_pair[0].pop(0))
    return smallest


def build_huffman_tree(histogram: OrderedDict) -> HuffmanNode:
    """Builds a Huffman tree from the histogram."""
    list_pair = [[HuffmanNode(symbol, weight) for symbol, weight in histogram.items()], []]
    if len(list_pair[0]) == 1:
        return list_pair[0][0]
    while len(list_pair[0]) + len(list_pair[1]) > 1:
        left_node, right_node = get_smallest_nodes(list_pair)
        new_node = HuffmanNode(None, left_node.weight + right_node.weight)
        new_node.left = left_node
        new_node.right = right_node
        list_pair[1].append(new_node)
    if not len(list_pair[0]) == 0 or len(list_pair[1]) != 1:
        raise NotImplementedError('Huffman Tree construction did not run as expected')
    return list_pair[1][0]


def generate_code_from_tree(root: HuffmanNode) -> defaultdict:
    """Generates a codebook from the Huffman tree."""
    def get_code_from_tree(node: HuffmanNode, prefix=bitarray(), codebook=None):
        if codebook is None:
            codebook = defaultdict(bitarray)
        if node.symbol is not None:  # Leaf node
            codebook[node.symbol] = prefix
        else:
            if node.left is None or node.right is None:
                raise ValueError("Huffman tree is not complete")
            get_code_from_tree(node.left, prefix + bitarray('0'), codebook)
            get_code_from_tree(node.right, prefix + bitarray('1'), codebook)
        return codebook
    if root.symbol is not None:
        codebook = defaultdict(bitarray)
        codebook[root.symbol] = bitarray('0')
        return codebook
    return get_code_from_tree(root)


def generate_tree_from_code(codebook: defaultdict):
    """Generates a Huffman tree from the codebook."""
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
    """Encodes the data using the provided codebook."""
    encoded_data = bitarray()
    for symbol in data:
        encoded_data += codebook[symbol]
    return encoded_data


def get_encoded_list(data: Iterable, codebook: defaultdict) -> list:
    """Encodes the data into a list of codes using the provided codebook."""
    encoded_list = []
    for symbol in data:
        encoded_list.append(codebook[symbol])
    return encoded_list


def get_huffman_tree(array: Iterable) -> HuffmanNode:
    """Generates a Huffman tree from the given array."""
    sorted_histogram = sort_dict(get_histogram(array))
    huffman_tree = build_huffman_tree(sorted_histogram)
    return huffman_tree


def get_abstract_tree(root: HuffmanNode) -> list[tuple[Any, int]]:
    """Generates an abstract representation of the Huffman tree."""
    symbols_and_code_lengths = []
    def abstract_tree(node: HuffmanNode, code_length=0):
        if node.symbol is not None:
            symbols_and_code_lengths.append((node.symbol, code_length))
        else:
            if node.left is None or node.right is None:
                raise ValueError("Huffman tree is not complete")
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
    """
    Limits the maximum code length of the Huffman codes to the specified maximum.
    If the maximum code length is exceeded, the code lengths are adjusted to fit within the limit.
    """
    def get_tree_weight(length_counts: list[int]):
        tree_weight = 0
        for count_index, count in enumerate(length_counts):
            tree_weight += count * 2 ** (max_code_length - count_index - 1)
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
    """Converts an unsigned integer to a bitarray of a specified length."""
    bitarray_ = bitarray('0') * length
    for position_index in range(length - 1, -1, -1):
        bitarray_[position_index] = bool(unsigned_int % 2)
        unsigned_int //= 2
    return bitarray_


def build_sorted_codebook_from_abstraction(abstract_tree: list[tuple[Any, int]]) -> defaultdict:
    """
    Builds a codebook from the abstract representation of the Huffman tree.
    The abstract tree is a list of tuples where each tuple contains a symbol and its corresponding code length.
    The codebook is a defaultdict where the keys are the symbols and the values are the corresponding bitarrays.
    """
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
    """
    Encodes the given array using Huffman encoding.
    Returns a tuple containing the Huffman tree and the encoded data as a bitarray.
    """
    huffman_tree = get_huffman_tree(array)
    code = generate_code_from_tree(huffman_tree)
    encoded = encode_data_with_codebook(array, code)
    return huffman_tree, encoded


def huffman_decode(huffman_tree: HuffmanNode, encoded_data: bitarray) -> list:
    """
    Decodes the encoded data using the provided Huffman tree.
    Returns a list of decoded symbols.
    """
    decoded_data = []
    node = huffman_tree
    for bit in encoded_data:
        if node is None:
            raise ValueError("Huffman tree is not complete or encoded data is invalid")
        node = node.right if bit else node.left
        if node is None:
            raise ValueError("Huffman tree is not complete or encoded data is invalid")
        if node.symbol is not None:  # Leaf node
            decoded_data.append(node.symbol)
            node = huffman_tree
    return decoded_data


def huffman_encode_image(img: ArrayLike) -> tuple[tuple[int, ...], HuffmanNode, bitarray]:
    """
    Encodes a 2D image using Huffman encoding.
    The image must be a 2D array of integers.
    Returns a tuple containing the shape of the image, 
    the Huffman tree, and the encoded data as a bitarray.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if not issubclass(img.dtype.type, np.integer) or not img.ndim == 2:
        raise ValueError("Image must be an integer 2D array")
    shape = tuple(img.shape)
    huffman_tree, encoded_data = huffman_encode(img.flatten())
    return shape, huffman_tree, encoded_data


def huffman_decode_image(shape: tuple, huffman_tree: HuffmanNode, encoded_data: bitarray) -> NDArray:
    """
    Decodes the encoded data of a 2D image using the provided Huffman tree.
    The shape of the image must be provided to reshape the decoded data.
    Returns a 2D numpy array of the decoded image.
    """
    return np.array(huffman_decode(huffman_tree, encoded_data)).reshape(shape)
