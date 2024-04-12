import numpy as np


class ItemPriorityBase:
    """
    The base class for inventory item priority logic.
    """

    def _split(self, items, forced_items, weight_capacity):
        '''
        returns a dict (container_item or None for inventory) ->
                       (list of counts to take corresponding to `items`)

        Lack of the container in the dict means "don't change the content except for
        items wanted by other containers"

        Order of `items` matters. First items are more important.
        Otherwise the agent will drop and pickup items repeatedly.

        The function should be monotonic (i.e. removing an item from the argument,
        shouldn't decrease counts of other items). Otherwise the agent may
        go to the item, don't take it, and repeat infinitely

        weight capacity can be exceeded. It's only a hint what the agent wants
        '''
        raise NotImplementedError()

    def split(self, items, forced_items, weight_capacity):
        ret = self._split(items, forced_items, weight_capacity)
        assert None in ret
        counts = np.array(list(ret.values())).sum(0)
        assert all((0 <= count <= item.count for count, item in zip(counts, items)))
        assert all((0 <= c <= item.count for cs in ret.values() for c, item in zip(cs, items)))
        assert all((item not in ret or item.is_container() for item in items))
        assert all((item not in ret or ret[item][i] == 0 for i, item in enumerate(items)))
        return ret
