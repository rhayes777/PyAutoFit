### Module Mappings

# autofit/mapper -> VIS_AutoFit_Mapper
# autofit/mapper/prior -> VIS_AutoFit_Prior
# autofit/mapper/prior_model -> VIS_AutoFit_PriorModel
# autofit/text -> VIS_AutoFit_Text
# autofit/tools -> VIS_AutoFit_Tools

### Clean VIS_AutoFit project files

cd /home/jammy/PycharmProjects/PyEuclid/VIS_AutoFit/

rm -rf VIS_AutoFit_Mapper/python/VIS_AutoFit_Mapper/*
rm -rf VIS_AutoFit_Mapper/tests/python/*

rm -rf VIS_AutoFit_Prior/python/VIS_AutoFit_Prior/*
rm -rf VIS_AutoFit_Prior/tests/python/*

rm -rf VIS_AutoFit_PriorModel/python/VIS_AutoFit_PriorModel/*
rm -rf VIS_AutoFit_PriorModel/tests/python/*

rm -rf VIS_AutoFit_Text/python/VIS_AutoFit_Text/*
rm -rf VIS_AutoFit_Text/tests/python/*

rm -rf VIS_AutoFit_Tools/python/VIS_AutoFit_Tools/*
rm -rf VIS_AutoFit_Tools/tests/python/*

### Copy from PyAutoFit to VIS_AutoFit

cd /home/jammy/PycharmProjects/PyAuto/PyAutoFit/

# cp -r test_autoconf/unit/conftest.py ../../PyEuclid/VIS_AutoFit/

cp -r autofit/exc.py ../../PyEuclid/VIS_AutoFit/exc.py

cp -r autofit/mapper/*.py ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Mapper/python/VIS_AutoFit_Mapper/

cp -r autofit/mapper/prior/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Prior/python/VIS_AutoFit_Prior/
cp -r test_autofit/unit/mapper/test_abstract.py ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Prior/tests/python/
cp -r test_autofit/unit/mapper/mock.py ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Prior/tests/python/

cp -r autofit/mapper/prior_model/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_PriorModel/python/VIS_AutoFit_PriorModel/

cp -r autofit/text/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Text/python/VIS_AutoFit_Text/

cp -r autofit/tools/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Tools/python/VIS_AutoFit_Tools/

### Change import names

cd /home/jammy/PycharmProjects/PyEuclid/VIS_AutoFit/

rm VIS_AutoFit_Tools/python/VIS_AutoFit_Tools/phase.py

# Project Level

sed -i 's/from autofit import exc/import exc/g' */*/*/*.py

# PyAutoConf
sed -i 's/from autoconf/from VIS_AutoFit_Conf/g' */*/*/*.py

# The order prior_model -> prior -> mapper is required to change imports correctly.

# PriorModel
sed -i 's/from autofit.mapper.prior_model/from VIS_AutoFit_PriorModel/g' */*/*/*.py

# Prior
sed -i 's/from autofit.mapper.prior/from VIS_AutoFit_Prior/g' */*/*/*.py

# Mapper
sed -i 's/from autofit.mapper/from VIS_AutoFit_Mapper/g' */*/*/*.py

# Text
sed -i 's/from autofit.text/from VIS_AutoFit_Text/g' */*/*/*.py

# Tools
sed -i 's/from autofit.tools/from VIS_AutoFit_Tools/g' */*/*/*.py