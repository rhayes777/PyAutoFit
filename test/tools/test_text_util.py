from autofit.tools import text_util

import shutil
import os

text_path = "{}/../test_files/text/".format(os.path.dirname(os.path.realpath(__file__)))

def test__label_and_label_string():

    string0 = text_util.label_and_label_string(label0='param0', label1='mass', whitespace=10)
    string1 = text_util.label_and_label_string(label0='param00', label1='mass0', whitespace=10)
    string2 = text_util.label_and_label_string(label0='param000', label1='mass111', whitespace=10)

    assert string0 == 'param0    mass'
    assert string1 == 'param00   mass0'
    assert string2 == 'param000  mass111'

    string0 = text_util.label_and_label_string(label0='param0', label1='mass', whitespace=20)

    assert string0 == 'param0              mass'

def test__label_and_value_string():

    string0 = text_util.label_and_value_string(label='param0', value=2.0, whitespace=10, format_str='{:.2f}')
    string1 = text_util.label_and_value_string(label='param00', value=2.0, whitespace=10, format_str='{:.2f}')
    string2 = text_util.label_and_value_string(label='param000', value=2.0, whitespace=10, format_str='{:.2f}')

    assert string0 == 'param0    2.00'
    assert string1 == 'param00   2.00'
    assert string2 == 'param000  2.00'

    string = text_util.label_and_value_string(label='param11', value=3.00, whitespace=20, format_str='{:.4f}')

    assert string == 'param11             3.0000'

    string = text_util.label_and_value_string(label='param11', value=3.00, whitespace=15, format_str='{:.2e}')

    assert string == 'param11        3.00e+00'

def test__label_value_and_limits_string():

    string0 = text_util.label_value_and_limits_string(label='param0', value=2.0, upper_limit=2.5, lower_limit=1.5,
                                                      whitespace=10, format_str='{:.2f}')
    string1 = text_util.label_value_and_limits_string(label='param00', value=2.0, upper_limit=2.7, lower_limit=1.3,
                                                      whitespace=10, format_str='{:.2f}')
    string2 = text_util.label_value_and_limits_string(label='param000', value=2.0, upper_limit=2.9, lower_limit=1.1,
                                                      whitespace=10, format_str='{:.2f}')

    assert string0 == 'param0    2.00 (1.50, 2.50)'
    assert string1 == 'param00   2.00 (1.30, 2.70)'
    assert string2 == 'param000  2.00 (1.10, 2.90)'

    string = text_util.label_value_and_limits_string(label='param11', value=3.00, upper_limit=40000.0, lower_limit=0.0001,
                                                     whitespace=20, format_str='{:.4f}')

    assert string == 'param11             3.0000 (0.0001, 40000.0000)'

    string = text_util.label_value_and_limits_string(label='param11', value=3.00, upper_limit=500.0, lower_limit=1.0,
                                                     whitespace=15, format_str='{:.2e}')

    assert string == 'param11        3.00e+00 (1.00e+00, 5.00e+02)'

def test__label_value_and_unit_string():

    string0 = text_util.label_value_and_unit_string(label='param0', value=2.0, unit='arcsec', whitespace=10, format_str='{:.2f}')
    string1 = text_util.label_value_and_unit_string(label='param00', value=2.0, unit='mass', whitespace=10, format_str='{:.2f}')
    string2 = text_util.label_value_and_unit_string(label='param000', value=2.0, unit='kg', whitespace=10, format_str='{:.2f}')

    assert string0 == 'param0    2.00 arcsec'
    assert string1 == 'param00   2.00 mass'
    assert string2 == 'param000  2.00 kg'

    string = text_util.label_value_and_unit_string(label='param11', value=3.00, unit='kg', whitespace=20, format_str='{:.4f}')

    assert string == 'param11             3.0000 kg'

    string = text_util.label_value_and_unit_string(label='param11', value=3.00, unit='kgs', whitespace=15, format_str='{:.2e}')

    assert string == 'param11        3.00e+00 kgs'

def test__output_list_of_strings_to_file():

    if os.path.exists(text_path):
        shutil.rmtree(text_path)

    os.mkdir(text_path)

    results = ['hi\n', 'hello']
    text_util.output_list_of_strings_to_file(file=text_path + 'model.results', list_of_strings=results)

    file = open(text_path+'model.results', 'r')

    assert file.readlines() == ['hi\n', 'hello']