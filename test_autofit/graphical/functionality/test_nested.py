import collections 

import pytest


from jax import tree_util

from autofit.graphical import utils


NTuple = collections.namedtuple("NTuple", "first, last")

def jax_nested_zip(tree, *rest):
    leaves, treedef = tree_util.tree_flatten(tree)
    return zip(leaves, *(treedef.flatten_up_to(r) for r in rest))


def jax_key_to_val(key):
    if isinstance(key, tree_util.SequenceKey):
        return key.idx 
    elif isinstance(key, (tree_util.DictKey, tree_util.FlattenedIndexKey)):
        return key.key 
    elif isinstance(key, tree_util.GetAttrKey):
        return key.name 
    return key

def jax_path_to_key(path):
    return tuple(map(jax_key_to_val, path))


def test_nested_getitem():
    obj = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, 'd': (3, {'e': [4, 5]})}

    assert utils.nested_get(obj, ('b',)) == 2
    assert utils.nested_get(obj, ('c', 'a')) == 1
    assert utils.nested_get(obj, ('d', 0)) == 3
    assert utils.nested_get(obj, ('d', 1, 'e', 1)) == 5


def test_nested_setitem():
    obj = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, 'd': (3, {'e': [4, 5]})}

    utils.nested_set(obj, ('b',), 3)
    assert utils.nested_get(obj, ('b',))

    utils.nested_set(obj, ('c', 'a'), 2) 
    assert utils.nested_get(obj, ('c', 'a')) == 2

    utils.nested_set(obj, ('d', 1, 'e', 1), 6) 
    assert utils.nested_get(obj, ('d', 1, 'e', 1)) == 6

    with pytest.raises(TypeError):
        utils.nested_set(obj, ('d', 0), 4) 


def test_nested_order():

    obj1 = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, 'd': (3, {'e': [4, 5]})}
    obj2 = {"a": 1, "b": 2, 'd': (3, {'e': [4, 5]}), "c": {"b": 2, "a": 1}}

    assert all(v1 == v2 for (v1, v2) in utils.nested_zip(obj1, obj2))
    assert all(utils.nested_filter(lambda x, y: x == y, obj1, obj2))
    assert list(utils.nested_zip(obj1)) == list(utils.nested_zip(obj2))
    assert list(utils.nested_zip(obj1, obj2)) == list(jax_nested_zip(obj1, obj2))

    obj1 = {"b": 2, "a": 1, "c": {"b": 2, "a": 1}, 'd': (3, {'e': NTuple(4, 5)})}
    obj2 = {"a": 1, "b": 2, 'd': (3, {'e': NTuple(4, 5)}), "c": {"b": 2, "a": 1}}

    assert all(v1 == v2 for (v1, v2) in utils.nested_zip(obj1, obj2))
    assert all(utils.nested_filter(lambda x, y: x == y, obj1, obj2))
    assert list(utils.nested_zip(obj1)) == list(utils.nested_zip(obj2))
    assert list(utils.nested_zip(obj1, obj2)) == list(jax_nested_zip(obj1, obj2))


def test_nested_items():
    
    obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': [4, 5]})}

    for (k1, v1), (p2, v2) in zip(
            utils.nested_items(obj1), 
            tree_util.tree_flatten_with_path(obj1)[0]
    ):
        assert k1 == jax_path_to_key(p2)
        assert v1 == v2


def test_nested_filter():
    obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': [4, 5]})}
    assert list(utils.nested_filter(lambda x: x % 2 == 0, obj1)) == [(2,), (4,), (2,)]

    obj1 = {"b": 2, "a": 1,  'c': (3, {'e': [4, 5]}), "d": {"b": 2, "a": 1}}
    assert list(utils.nested_filter(lambda x: x % 2 == 0, obj1)) == [(2,), (4,), (2,)]


def test_nested_map():
    obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': [4, 5]})}
    obj2 = {'a': 2, 'b': 4, 'c': (6, {'e': [8, 10]}), 'd': {'a': 2, 'b': 4}}
    obj12 = utils.nested_map(lambda x: x*2, obj1)
    assert obj12 == obj2

    obj3 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': (4, 5)})}
    obj4 = {'a': 2, 'b': 4, 'c': (6, {'e': (8, 10)}), 'd': {'a': 2, 'b': 4}}
    obj32 = utils.nested_map(lambda x: x*2, obj3)
    assert obj32 == obj4


    obj5 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': NTuple(4, 5)})}
    obj6 = {'a': 2, 'b': 4, 'c': (6, {'e': NTuple(8, 10)}), 'd': {'a': 2, 'b': 4}}
    obj52 = utils.nested_map(lambda x: x*2, obj5)
    assert obj52 == obj6 == obj4

    assert obj32 != obj2
    assert obj52 != obj2


    assert all(utils.nested_iter(utils.nested_map(
        lambda a, b, c: a == b == c, obj1, obj3, obj5
    )))
    assert all(utils.nested_iter(utils.nested_map(
        lambda a, b, c: a == b == c, obj2, obj4, obj6
    )))
    assert all(map(lambda x: x[0] == x[1] == x[2],  utils.nested_zip(obj1, obj3, obj5)))
    assert all(map(lambda x: x[0] == x[1] == x[2],  utils.nested_zip(obj2, obj32, obj52)))


def test_nested_update():
    assert utils.nested_update([1, (2, 3), [3, 2, {1, 2}]], {2: 'a'}) == [1, ('a', 3), [3, 'a', {1, 'a'}]]
    assert utils.nested_update([1, NTuple(2, 3), [3, 2, {1, 2}]], {2: 'a'}) == [1, ('a', 3), [3, 'a', {1, 'a'}]]
    assert isinstance(utils.nested_update([1, NTuple(2, 3), [3, 2, {1, 2}]], {2: 'a'})[1], NTuple)
    assert utils.nested_update([{2: 2}], {2: 'a'}) == [{2: 'a'}]
    

def test_nested_items():
    obj1 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': [4, 5]})}
    obj2 = {'a': 2, 'b': 4, 'c': (6, {'e': [8, 10]}), 'd': {'a': 2, 'b': 4}}

    obj3 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': (4, 5)})}
    obj4 = {'a': 2, 'b': 4, 'c': (6, {'e': (8, 10)}), 'd': {'a': 2, 'b': 4}}

    obj5 = {"b": 2, "a": 1, "d": {"b": 2, "a": 1}, 'c': (3, {'e': NTuple(4, 5)})}
    obj6 = {'a': 2, 'b': 4, 'c': (6, {'e': NTuple(8, 10)}), 'd': {'a': 2, 'b': 4}}

    for path, val in utils.nested_items(obj1):
        assert utils.nested_getitem(obj2, path) == utils.nested_getitem(obj4, path) == val * 2

    for path, val in utils.nested_items(obj3):
        assert utils.nested_getitem(obj4, path) == utils.nested_getitem(obj6, path) == val * 2

    for path, val in utils.nested_items(obj5):
        assert utils.nested_getitem(obj6, path) == utils.nested_getitem(obj2, path) == val * 2

    assert list(utils.nested_items([NTuple(1, 2), {2: 5, 1: 3}])) == [((0, 0), 1), ((0, 1), 2), ((1, 1), 3), ((1, 2), 5)]

    assert list(utils.nested_items([1, (2, 3), [3, {'a': 1, 'b': 2}]])) == list(utils.nested_items([1, (2, 3), [3, {'b': 2, 'a': 1}]]))
    assert list(utils.nested_items([1, (2, 3), [3, {'b': 2, 'a': 1, }]])) == list(utils.nested_items([1, (2, 3), [3, {'b': 2, 'a': 1}]]))

    obj1 = [1, (2, 3), [3, {'b': 2, 'a': 1, }]]
    obj2 = [1, (2, 3), [3, {'a': 1, 'b': 2, }]]
    obj3 = [1, NTuple(2, 3), [3, {'a': 1, 'b': 2, }]]

    # Need jax version > 0.4
    if hasattr(tree_util, "tree_flatten_with_path"):
        jax_flat = tree_util.tree_flatten_with_path(obj1)[0]
        af_flat = utils.nested_items(obj2)

        for (jpath, jval), (akey, aval) in zip(jax_flat, af_flat):
            jkey = jax_path_to_key(jpath) 
            assert jkey == akey 
            assert jval == aval 
            assert (
                utils.nested_get(obj2, jkey) 
                == utils.nested_get(obj1, jkey)
                == utils.nested_get(obj2, akey)
                == utils.nested_get(obj1, akey)
            )

        jax_flat = tree_util.tree_flatten_with_path(obj2)[0]
        af_flat = utils.nested_items(obj3)
        for (jpath, jval), (akey, aval) in zip(jax_flat, af_flat):
            jkey = jax_path_to_key(jpath) 
            assert jkey == akey 
            assert jval == aval 
            assert (
                utils.nested_get(obj2, jkey) 
                == utils.nested_get(obj1, jkey)
                == utils.nested_get(obj2, akey)
                == utils.nested_get(obj1, akey)
                == utils.nested_get(obj3, jkey)
                == utils.nested_get(obj3, akey)
            )