from test_autofit.simulate.gaussian import makers

# Welcome to the PyAutoFit test_autoarray suite data_type maker. Here, we'll make the suite of data_type that we use to test_autoarray and profile
# PyAutoLens. This consists of the following sets of images:

sub_size = 1

# To simulate each lens, we pass it a name and call its maker. In the makers.py file, you'll see the
makers.make__gaussian(sub_size=sub_size)
