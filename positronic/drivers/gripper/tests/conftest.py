"""Stand-in for the pymodbus vendor package.

``pymodbus`` ships only in the ``hardware`` extra, so the gripper drivers cannot be imported from a default
sync. The tests hand the driver their own fake client and never call into the vendor, so bare modules carrying
the names the driver binds at import are enough. Installed here, before any test module imports a driver, and
only when the real package is absent.
"""

import importlib.util
import sys
import types

PACKAGE = 'pymodbus'
CLIENT = f'{PACKAGE}.client'
EXCEPTIONS = f'{PACKAGE}.exceptions'

if importlib.util.find_spec(PACKAGE) is None:
    client = types.ModuleType(CLIENT)
    exceptions = types.ModuleType(EXCEPTIONS)
    modbus_exception = type('ModbusException', (Exception,), {})
    exceptions.__dict__.update(
        ModbusException=modbus_exception,
        ModbusIOException=type('ModbusIOException', (modbus_exception,), {}),
        ConnectionException=type('ConnectionException', (modbus_exception,), {}),
    )
    package = types.ModuleType(PACKAGE)
    package.__dict__.update(client=client, exceptions=exceptions)
    sys.modules[PACKAGE] = package
    sys.modules[CLIENT] = client
    sys.modules[EXCEPTIONS] = exceptions
