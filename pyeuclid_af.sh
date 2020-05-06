
# cp -r test_autoconf/unit/conftest.py ../../PyEuclid/VIS_AutoFit/

cd /home/jammy/PycharmProjects/PyAuto/PyAutoConf/

cp -r autofit/mapper/prior/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Prior/python/VIS_AutoFit_Prior/
cp -r test_autofit/unit/mapper/prior/* ../../PyEuclid/VIS_CTI/VIS_CTI_Dataset/tests/python/

cd /home/jammy/PycharmProjects/PyEuclid/VIS_AutoFit/

cp -r VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/json_prior/config.py VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/config.py
cp -r VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/json_prior/converter.py VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/converter.py
cp -r VIS_AutoFit_Conf/tests/python/json_prior/* VIS_AutoFit_Conf/tests/python/

rm -rf cp -r VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/json_prior/
rm -rf cp -r VIS_AutoFit_Conf/tests/python/json_prior/

# Conf

sed -i 's/from autoconf/from VIS_AutoFit_Conf/g' */*/*/*.py

sed -i 's/import autoconf/import VIS_AutoFit_Conf/g' */*/*/*.py

sed -i 's/return autoconf/return VIS_AutoFit_Conf/g' */*/*/*.py

sed -i 's/from test_autoconf/from VIS_AutoFit_Conf.tests/g' */*/*/*.py

sed -i 's/.json_prior//g' */*/*/*.py

# Permissions

chmod -R 777 ../VIS_AutoFit
chmod -R 777 ../VIS_AutoFit/*
chmod -R 777 ../VIS_AutoFit/*/*
chmod -R 777 ../VIS_AutoFit/*/*/*
chmod -R 777 ../VIS_AutoFit/*/*/*/*
chmod -R 777 ../VIS_AutoFit/*/*/*/*/*
chmod -R 777 ../VIS_AutoFit/*/*/*/*/*/*

cd /home/jammy/PycharmProjects/PyAuto/PyAutoFit/
