
# import package
using StructuralIdentifiability

model = @ODEmodel(
        s'(t) = (1-g) * sigma * Nw - (b * c(t) / Nw) * s(t)  - delta * s(t) + tau * c(t),
        c'(t) = g * sigma * Nw     +  (b * c(t) / Nw) * s(t) - delta * c(t) - tau * c(t),
        y(t)  = ro * (c(t) / Nw)
    )

assess_identifiability(model, [b, g, ro])