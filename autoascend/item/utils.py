def flatten_items(iterable):
    ret = []
    for item in iterable:
        ret.append(item)
        if item.is_container():
            ret.extend(flatten_items(item.content))
    return ret


def find_equivalent_item(item, iterable):
    assert item.text
    for i in iterable:
        assert i.text
        if i.text == item.text:
            return i
    assert 0, (item, iterable)


def check_if_triggered_container_trap(message):
    return ('A cloud of ' in message and ' gas billows from ' in message) or \
           'Suddenly you are frozen in place!' in message or \
           'A tower of flame bursts from ' in message or \
           'You are jolted by a surge of electricity!' in message or \
           'But luckily ' in message
