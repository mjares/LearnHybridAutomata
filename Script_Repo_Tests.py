from infer_ha.infer_HA import infer_model
from infer_ha.model_printer.print_HA import print_HA
from utils.parse_parameters import parse_trajectories
from utils.commandline_parser import process_type_annotation_parameters

################# Learn HA Oscillator without Annotate  #############################
parameters = {}
print("Running test runLearnHA module")

parameters['input_filename'] = "data/simu_oscillator_2.txt"
parameters['output_filename'] = "HA_oscillator_2_withoutAnnotate.txt"

parameters['clustering_method'] = 1
parameters['methods'] = "dtw"

parameters['ode_degree'] = 1
parameters['modes'] = 4
parameters['guard_degree'] = 1
parameters['segmentation_error_tol'] = 0.1
parameters['segmentation_fine_error_tol'] = 0.1
parameters['threshold_distance'] = 1.0
parameters['threshold_correlation'] = 0.89
parameters['dbscan_eps_dist'] = 0.01  # default value
parameters['dbscan_min_samples'] = 2  # default value
parameters['size_input_variable'] = 0
parameters['size_output_variable'] = 2
parameters['variable_types'] = ''
parameters['pool_values'] = ''
parameters['ode_speedup'] = 50
parameters['is_invariant'] = 0
parameters['stepsize'] = 0.01
parameters['filter_last_segment'] = 1
parameters['lmm_step_size'] = 5

input_filename = parameters['input_filename']
output_filename = parameters['output_filename']
list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
# print("list of trajectories is ", list_of_trajectories)
variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
    variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

parameters['stepsize'] = stepsize  # we assume trajectories are sampled at fixed size time-step
parameters['variableType_datastruct'] = variableType_datastruct
P, G, mode_inv, transitions, position = infer_model(list_of_trajectories, parameters)
# print("Number of modes learned = ", len(P))
print_HA(P, G, mode_inv, transitions, position, parameters, output_filename)  # prints an HA model file

################# Learn HA Oscillator with Annotate #####################
parameters = {}
print("Running test runLearnHA module with Oscillator model with type annotation")

parameters['input_filename'] = "data/simu_oscillator_2.txt"
parameters['output_filename'] = "HA_oscillator_2_withAnnotate.txt"

parameters['clustering_method'] = 1
parameters['methods'] = "dtw"

parameters['ode_degree'] = 1
parameters['modes'] = 4
parameters['guard_degree'] = 1
parameters['segmentation_error_tol'] = 0.1
parameters['segmentation_fine_error_tol'] = 0.1
parameters['threshold_distance'] = 1.0
parameters['threshold_correlation'] = 0.89
parameters['dbscan_eps_dist'] = 0.01  # default value
parameters['dbscan_min_samples'] = 2  # default value
parameters['size_input_variable'] = 0
parameters['size_output_variable'] = 2
parameters['variable_types'] = 'x0=t1,x1=t1'
parameters['pool_values'] = ''
parameters['constant_value'] = ''
parameters['ode_speedup'] = 50
parameters['is_invariant'] = 0
parameters['stepsize'] = 0.01
parameters['filter_last_segment'] = 1
parameters['lmm_step_size'] = 5

input_filename = parameters['input_filename']
output_filename = parameters['output_filename']
list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
# print("list of trajectories is ", list_of_trajectories)
variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
    variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

parameters['stepsize'] = stepsize  # we assume trajectories are sampled at fixed size time-step
parameters['variableType_datastruct'] = variableType_datastruct
P, G, mode_inv, transitions, position = infer_model(list_of_trajectories, parameters)
# print("Number of modes learned = ", len(P))

print_HA(P, G, mode_inv, transitions, position, parameters, output_filename) # prints an HA model file

################## Learn HA with Bouncing Ball model ###########################
parameters = {}
print("Running test runLearnHA module with Bouncing Ball model with type annotation")
parameters['input_filename'] = "data/simu_bball_2.txt"
parameters['output_filename'] = "HA_bball_2.txt"

parameters['clustering_method'] = 1
parameters['methods'] = "dtw"

parameters['ode_degree'] = 1
parameters['modes'] = 1
parameters['guard_degree'] = 1
parameters['segmentation_error_tol'] = 0.1
parameters['segmentation_fine_error_tol'] = 0.9
parameters['threshold_distance'] = 9.0
parameters['threshold_correlation'] = 0.8
parameters['dbscan_eps_dist'] = 0.01  # default value
parameters['dbscan_min_samples'] = 2  # default value
parameters['size_input_variable'] = 1
parameters['size_output_variable'] = 2
parameters['variable_types'] = 'x0=t1,x1=t3'
parameters['constant_value'] = 'x1=0'
parameters['lmm_step_size'] = 5
parameters['pool_values'] = ''
parameters['ode_speedup'] = 50
parameters['is_invariant'] = 2
parameters['stepsize'] = 0.01
parameters['filter_last_segment'] = 1

input_filename = parameters['input_filename']
output_filename = parameters['output_filename']
list_of_trajectories, stepsize, system_dimension = parse_trajectories(input_filename)
# print("list of trajectories is ", list_of_trajectories)
variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
    variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

parameters['stepsize'] = stepsize  # we assume trajectories are sampled at fixed size time-step
parameters['variableType_datastruct'] = variableType_datastruct
P, G, mode_inv, transitions, position = infer_model(list_of_trajectories, parameters)
# print("Number of modes learned = ", len(P))

print_HA(P, G, mode_inv, transitions, position, parameters, output_filename) # prints an HA model file
