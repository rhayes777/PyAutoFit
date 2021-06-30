from autofit import graphical as g


def test():
    mean_field = g.MeanField({

    })

    print(mean_field.prior_count)
    mean_field.instance_for_arguments({})
