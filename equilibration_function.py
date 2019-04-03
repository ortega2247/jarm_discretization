import numpy as np
from itertools import compress


def pre_equilibration(solver, time_search=None, param_values=None, tolerance=1e-6):
    """

    Parameters
    ----------
    solver : pysb solver
        a pysb solver object
    time_search : np.array, optional
        Time span array used to find the equilibrium. If not provided, function will use tspan from the solver
    param_values :  dict or np.array, optional
        Model parameters used to find the equilibrium, it can be an array with all model parameters
        (this array must have the same order as model.parameters) or it can be a dictionary where the
        keys are the parameter names thatpara want to be changed and the values are the new parameter
        values. If not provided, function will use param_values from the solver
    tolerance : float
        Tolerance to define when the equilibrium has been reached

    Returns
    -------

    """
    # Solve system for the time span provided
    sims = solver.run(tspan=time_search, param_values=param_values)
    if time_search is None:
        time_search = solver.tspan

    if sims.nsims == 1:
        simulations = [sims.species]
    else:
        simulations = sims.species

    dt = time_search[1] - time_search[0]

    all_times_eq = [0] * sims.nsims
    all_conc_eq = [0] * sims.nsims
    for n, y in enumerate(simulations):
        time_to_equilibration = [0, 0]
        for idx in range(y.shape[1]):
            sp_eq = False
            derivative = np.diff(y[:, idx]) / dt
            derivative_range = ((derivative < tolerance) & (derivative > -tolerance))
            # Indexes of values less than tolerance and greater than -tolerance
            derivative_range_idxs = list(compress(range(len(derivative_range)), derivative_range))
            for i in derivative_range_idxs:
                # Check if derivative is close to zero in the time points ahead
                if i + 3 > len(time_search):
                    raise Exception('Equilibrium can not be reached within the time_search input')
                if (derivative[i + 3] < tolerance) | (derivative[i + 3] > -tolerance):
                    sp_eq = True
                    if time_search[i] > time_to_equilibration[0]:
                        time_to_equilibration[0] = time_search[i]
                        time_to_equilibration[1] = i
                if not sp_eq:
                    raise Exception('Equilibrium can not be reached within the time_search input')
                if sp_eq:
                    break
            else:
                raise Exception('Species s{0} has not reached equilibrium'.format(idx))

        conc_eq = y[time_to_equilibration[1]]
        all_times_eq[n] = time_to_equilibration
        all_conc_eq[n] = conc_eq

    return all_times_eq, all_conc_eq