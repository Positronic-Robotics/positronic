import configuronic as cfgc
from configuronic.tests.support_package.subpkg.a import A

a_cfg_value1 = cfgc.Config(A, value=1)
a_cfg_value2 = cfgc.Config(A, value=2)
