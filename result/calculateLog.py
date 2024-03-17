
import glob
import math
def std_list(list_input):
    # Calculation of the average
    mean = sum(list_input) / len(list_input)
    # Computing the variance
    variance = sum([((x - mean) ** 2) for x in list_input]) / len(list_input)
    # Calculate standard deviation
    std_deviation = math.sqrt(variance)
    return mean, std_deviation


# Loop processing case*
case_pattern = "case0_result"
case_list = glob.glob(case_pattern)

for case_dir in case_list:
    # Nest another loop within the case* loop to process the log*
    seed_pattern = case_dir + "/log*.txt"
    seed_list = glob.glob(seed_pattern)

    # For each case file, merge the five log*.txt files.
    roc_list = []  # Create an empty list for storing roc values
    aupr_list = []  # Create an empty list for storing aupr values
    accuracy_list = []  # Create an empty list for storing accuracy values
    precision_list = []  # Create an empty list for storing precision values
    sensitivity_list = []  # Create an empty list for storing sensitivity values
    specificity_list = []  # Create an empty list for storing specificity values
    f1_list = []  # Create an empty list for storing f1 values
    timeconsuming_list = []  # Create an empty list for storing time consuming values

    for file_path in seed_list:
        with open(file_path, 'r', encoding='UTF-8') as file:
            content = file.read()

        # Extraction of individual values
        lines = content.split('\n')
        num_nodes = int(lines[2].split('=')[1])
        cross_validation = lines[3]
        num_epochs = int(lines[4].split(':')[1].strip())
        learn_rate = float(lines[5].split('=')[1].strip())
        weight_decay = float(lines[6].split('=')[1].strip())
        node_dim = int(lines[7].split('=')[1].strip())
        start_time = float(lines[8].split('=')[1].strip())

        roc = float(lines[13].split('=')[1].strip())
        aupr = float(lines[14].split('=')[1].strip())
        accuracy = float(lines[15].split('=')[1].strip())
        precision = float(lines[16].split('=')[1].strip())
        sensitivity = float(lines[17].split('=')[1].strip())
        specificity = float(lines[18].split('=')[1].strip())
        f1 = float(lines[19].split('=')[1].strip())
        time_consuming = float(lines[20].split(':')[1].strip())

        # Extracted values and added to list
        roc_list.append(roc)
        aupr_list.append(aupr)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        f1_list.append(f1)
        timeconsuming_list.append(time_consuming)

        print("file_path:", file_path)

        # print result
        # ...
    F1, F1_std = std_list(f1_list)
    ROC, ROC_std = std_list(roc_list)
    AUPR, AUPR_std = std_list(aupr_list)
    Accuracy, Accuracy_std = std_list(accuracy_list)
    Sensitivity, Sensitivity_std = std_list(sensitivity_list)
    Specificity, Specificity_std = std_list(specificity_list)
    Precision, Precision_std = std_list(precision_list)

    print('ROC: {:.4f};          ROC: {:.4f}'.format(ROC, ROC_std))
    print('AUPR: {:.4f};         AUPR: {:.4f}'.format(AUPR, AUPR_std))
    print('F1: {:.4f};           F1: {:.4f}'.format(F1, F1_std))
    print('Precision: {:.4f};    Precision: {:.4f}'.format(Precision, Precision_std))
    print('Accuracy: {:.4f};     Accuracy: {:.4f}'.format(Accuracy, Accuracy_std))
    print('Sensitivity: {:.4f};  Sensitivity: {:.4f}'.format(Sensitivity, Sensitivity_std))
    print('Specificity: {:.4f};  Specificity: {:.4f}'.format(Specificity, Specificity_std))

