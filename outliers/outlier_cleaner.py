#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    #cleaned_data = []

    ### your code goes here
    residual_errors = (predictions - net_worths)**2
    tuple_list = zip(ages, net_worths, residual_errors)
    dtype = [('age', int), ('net_worth', float), ('residual_error', float)]
    structured_array = np.array(tuple_list, dtype = dtype)
    structured_array = np.sort(structured_array, order='residual_error')
    cleaned_data = structured_array[:int( 0.9*len(structured_array) )]
    
    return cleaned_data

