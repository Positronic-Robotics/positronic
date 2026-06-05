import subprocess
import sys


def test_registration_reaches_every_import_path():
    # The PUBLIC profile is registered as a side effect of importing positronic.utils. Both
    # entry points that resolve s3://PUBLIC@... must reach that registration on their own, in
    # a fresh interpreter:
    #   - inference servers import positronic.utils.checkpoints and never import cfg.ds;
    #   - dataset configs (e.g. cfg.ds.sim) import positronic.cfg.ds, which imports
    #     positronic.utils purely for the side effect — an import that looks unused and a
    #     future dev would happily delete.
    for import_stmt in ('import positronic.utils.checkpoints', 'import positronic.cfg.ds'):
        code = f'{import_stmt}\nfrom pos3.profiles import _resolve_profile\nassert _resolve_profile("PUBLIC").public\n'
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert result.returncode == 0, f'{import_stmt!r} did not register PUBLIC:\n{result.stderr}'
