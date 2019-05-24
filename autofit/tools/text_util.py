
def label_and_label_string(label0, label1, whitespace):
    return label0 + label1.rjust(whitespace - len(label0) + len(label1))

def label_and_value_string(label, value, whitespace, format_str='{:.4f}'):
    value = format_str.format(value)
    return label + value.rjust(whitespace - len(label) + len(value))

def label_value_and_limits_string(label, value, lower_limit, upper_limit, whitespace, format_str='{:.4f}'):
    value = format_str.format(value)
    upper_limit = format_str.format(upper_limit)
    lower_limit = format_str.format(lower_limit)
    value = value + ' (' + lower_limit + ', ' + upper_limit + ')'
    return label + value.rjust(whitespace - len(label) + len(value))

def label_value_and_unit_string(label, value, unit, whitespace, format_str='{:.4f}'):
    value = (format_str + ' {}').format(value, unit)
    return label + value.rjust(whitespace - len(label) + len(value))

def output_list_of_strings_to_file(file, list_of_strings):

    file = open(file, 'w')
    file.write(''.join(list_of_strings))
    file.close()