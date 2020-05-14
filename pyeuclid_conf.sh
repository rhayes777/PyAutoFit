### Module Mappings

# autoconf/ -> VIS_AutoFit_Conf

### Clean VIS_AutoFit project files

cd /home/jammy/PycharmProjects/PyEuclid/VIS_AutoFit/

rm -rf VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/*
rm -rf VIS_AutoFit_Conf/tests/python/*

### Copy from PyAutoConf to VIS_AutoFit

cd /home/jammy/PycharmProjects/PyAuto/PyAutoConf/

cp -r autoconf/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/
# cp -r test_autoconf/unit/* ../../PyEuclid/VIS_AutoFit/VIS_AutoFit_Conf/tests/python/

# For autoconf, we move the json_prior files out of their sub-package to the main VIS_AutoFit_Conf file
cd /home/jammy/PycharmProjects/PyEuclid/VIS_AutoFit/

cp -r VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/json_prior/config.py VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/config.py
cp -r VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/json_prior/converter.py VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/converter.py
# cp -r VIS_AutoFit_Conf/tests/python/json_prior/* VIS_AutoFit_Conf/tests/python/

rm -rf cp -r VIS_AutoFit_Conf/python/VIS_AutoFit_Conf/json_prior/
# rm -rf cp -r VIS_AutoFit_Conf/tests/python/json_prior/

### Change import names

sed -i 's/from autoconf/from VIS_AutoFit_Conf/g' */*/*/*.py
sed -i 's/import autoconf/import VIS_AutoFit_Conf/g' */*/*/*.py
sed -i 's/return autoconf/return VIS_AutoFit_Conf/g' */*/*/*.py
sed -i 's/from test_autoconf/from VIS_AutoFit_Conf.tests/g' */*/*/*.py
sed -i 's/.json_prior//g' */*/*/*.py
