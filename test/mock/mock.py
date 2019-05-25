

class MockClassNLOx4(object):

    def __init__(self, one=1, two=2, three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockClassNLOx5(object):

    def __init__(self, one=1, two=2, three=3, four=4, five=5):
        self.one = one
        self.two = two
        self.three = three
        self.four = four
        self.five = five


class MockClassNLOx6(object):

    def __init__(self, one=(1, 2), two=(3, 4), three=3, four=4):
        self.one = one
        self.two = two
        self.three = three
        self.four = four


class MockAnalysis(object):
    def __init__(self):
        self.kwargs = None
        self.instance = None
        self.visualise_instance = None

    def fit(self, instance):
        self.instance = instance
        return 1.

    # noinspection PyUnusedLocal
    def visualize(self, instance, *args, **kwargs):
        self.visualise_instance = instance