from infer_ha.infer_HA import infer_model
from infer_ha.model_printer.print_HA import print_HA
from utils.parse_parameters import parse_trajectories
from utils.commandline_parser import process_type_annotation_parameters


# traces_input_file = "data/simu_oscillator_2.txt"
traces_input_file = "data/test.txt"
# hybrid_model_output_file = "HA_oscillator_2_withoutAnnotate.txt"
hybrid_model_output_file = "HA_test.txt"

print("Running test learnHA module...")
# Parameters
parameters = {}
parameters['input_filename'] = traces_input_file
parameters['output_filename'] = hybrid_model_output_file
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
variableType_datastruct = []  # structure that holds [var_index, var_name, var_type, pool_values]
if len(parameters['variable_types']) >= 1:  # if user supply annotation arguments
    variableType_datastruct = process_type_annotation_parameters(parameters, system_dimension)

parameters['stepsize'] = stepsize  # we assume trajectories are sampled at fixed size time-step
parameters['variableType_datastruct'] = variableType_datastruct
P, G, mode_inv, transitions, position = infer_model(list_of_trajectories, parameters)
print_HA(P, G, mode_inv, transitions, position, parameters, output_filename)  # prints an HA model file
