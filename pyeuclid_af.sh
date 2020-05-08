cd /home/jammy/PycharmProjects/PyAuto/PyAutoFit/

# cp -r test_autoconf/unit/conftest.py ../../PyEuclid/VIS_AutoFit/

cp -r autofit/mapper/prior/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Prior/python/VIS_AutoFit_Prior/
cp -r autofit/mapper/prior_model/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_PriorModel/python/VIS_AutoFit_PriorModel/
cp -r autofit/mapper/*.py ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Model/python/VIS_AutoFit_Model/

# cp -r test_autofit/unit/mapper/prior/* ../../PyEuclid/VIS_CTI/VIS_CTI_Dataset/tests/python/

cd /home/jammy/PycharmProjects/PyEuclid/VIS_AutoFit/

# Conf

sed -i 's/from autofit.mapper.prior/from VIS_AutoFit_Prior/g' */*/*/*.py
sed -i 's/from autofit.mapper.prior/from VIS_AutoFit_Prior/g' */*/*/*/*.py

sed -i 's/from autofit.mapper.prior_model/from VIS_AutoFit_PriorModel/g' */*/*/*.py
sed -i 's/from autofit.mapper.prior_model/from VIS_AutoFit_PriorModel/g' */*/*/*/*.py

sed -i 's/from autofit.mapper/from VIS_AutoFit_Mapper/g' */*/*/*.py
sed -i 's/from autofit.mapper/from VIS_AutoFit_Mapper/g' */*/*/*/*.py
