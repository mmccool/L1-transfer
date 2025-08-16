# Shadowed LEO to L1 Transfer
Compute and plot rampled bang-bang transfer flight path from LEO to L1 while remaining in the shadow of a planetary-scale solar shade, allowing for configurable transfer times and g-force limits.

Initial code generated in ChapGPT 5 using the following prompts:

* Develop an orbital flight path from earth orbit to L1, with the spacecraft arriving at L1, or a point just before it on the line between the earth and the L1 point, with relative zero velocity.  The spacecraft should use an acceleration and a deceleration burn, not constant acceleration, and should also stay in the shadow of a proposed solar shade at the L1 point that provides shade for the entire earth as well as near-earth orbit, e.g. planetary scale.  The maximum G-forces of the burn should be specified as input, to stay within human tolerances, but the duration may change to fit orbital transfer requirements.  Other inputs will be the total time taken for the trip.  I need python code with a documented API to be able to explore various options for burn times, maximum G-forces, and time of flight.

Extended later with the following:

* Update the code to include departure from near-earth orbit, with a configurable altitude.  Find the optimal departure point to minimize fuel costs, assuming the plane of the orbit is the same as the vector between the L1 point and the center of the Earth.  Update the default maximum G value to 3 gravities, but with configurable smooth ramp up and down.  Ignore perturbations due to the Moon.

and

* Extend to a planar (x,y) model.
