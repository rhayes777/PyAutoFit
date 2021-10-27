import autofit as af

model = af.Model(
    af.Gaussian
)
print(model.prior_count)
