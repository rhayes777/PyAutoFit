def label_and_label_string(label0, label1, whitespace):
    return label0 + label1.rjust(whitespace - len(label0) + len(label1))

def label_and_value_string(label, value, whitespace, format_str='{:.4f}'):
    value = format_str.format(value)
    return label + value.rjust(whitespace - len(label) + len(value))

def label_value_and_unit_string(label, value, unit, whitespace, format_str='{:.4f}'):
    value = (format_str + ' {}').format(value, unit)
    return label + value.rjust(whitespace - len(label) + len(value))