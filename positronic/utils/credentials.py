"""A check that refuses text that looks like a credential before run metadata stores it.

It recognizes a key word next to a random value, an ``Authorization`` value, and an inline header set on the
command line. A credential under a header name with no key word, or with a low-entropy value, passes on any
flag except a ``headers`` one.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence

# The `generic-api-key` and `curl-auth-header` rules below are ported from gitleaks, commit 09242ce9,
# `config/gitleaks.toml`, with these changes: `\z` becomes `\Z`, Go's `$` becomes `\Z`, the stopwords and the
# `^[a-zA-Z_.-]+$` value allowlist are not ported, and the `Authorization` value is matched outside a curl command.
#
# MIT License
#
# Copyright (c) 2019 Zachary Rice
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
_GENERIC_API_KEY = re.compile(
    r"""(?i)[\w.-]{0,50}?(?:access|auth|(?-i:[Aa]pi|API)|credential|creds|key|passw(?:or)?d|secret|token)"""
    r"""(?:[ \t\w.-]{0,20})[\s'"]{0,3}(?:=|>|:{1,3}=|\|\||:|=>|\?=|,)[\x60'"\s=]{0,5}"""
    r"""([\w.=-]{10,150}|[a-z0-9][a-z0-9+/]{11,}={0,3})(?:[\x60'"\s;]|\\[nr]|\Z)""",
    re.ASCII,
)
_GENERIC_API_KEY_ENTROPY = 3.5
_GENERIC_API_KEY_ALLOWED_MATCH = re.compile(
    r"""(?i)(?:access(?:ibility|or)|access[_.-]?id|random[_.-]?access|api[_.-]?(?:id|name|version)|rapid|capital|"""
    r"""[a-z0-9-]*?api[a-z0-9-]*?:jar:|author|X-MS-Exchange-Organization-Auth|Authentication-Results|"""
    r"""(?:credentials?[_.-]?id|withCredentials)|"""
    r"""(?:bucket|foreign|hot|idx|natural|primary|pub(?:lic)?|schema|sequence)[_.-]?key|(?:turkey)|"""
    r"""key[_.-]?(?:alias|board|code|frame|id|length|mesh|name|pair|press(?:ed)?|ring|selector|signature|size|"""
    r"""stone|storetype|word|up|down|left|right)|key[_.-]?vault[_.-]?(?:id|name)|keyVaultToStoreSecrets|"""
    r"""key(?:store|tab)[_.-]?(?:file|path)|issuerkeyhash|(?-i:[DdMm]onkey|[DM]ONKEY)|keying|"""
    r"""(?:secret)[_.-]?(?:length|name|size)|UserSecretsId|(?:csrf)[_.-]?token|"""
    r"""(?:io\.jsonwebtoken[ \t]?:[ \t]?[\w-]+)|(?:api|credentials|token)[_.-]?(?:endpoint|ur[il])|"""
    r"""public[_.-]?token|(?:key|token)[_.-]?file|"""
    r"""(?-i:(?:[A-Z_]+=\n[A-Z_]+=|[a-z_]+=\n[a-z_]+=)(?:\n|\Z))|"""
    r"""(?-i:(?:[A-Z.]+=\n[A-Z.]+=|[a-z.]+=\n[a-z.]+=)(?:\n|\Z)))""",
    re.ASCII,
)
_AUTHORIZATION_VALUE = re.compile(
    r"""(?i)Authorization['"]?[ \t]{0,5}(?::|=>?|,)[ \t]{0,5}['"]?"""
    r"""(?:Basic[ \t]([a-z0-9+/]{8,}={0,3})|(?:Bearer|(?:Api-)?Token)[ \t]([\w=~@.+/-]{8,}))""",
    re.ASCII,
)
_AUTHORIZATION_VALUE_ENTROPY = 2.75

# The names in positronic's own configuration: `image.wrist_2`, `robot_state.q_7dof`. A value with no digit
# can still be a random token, so this passes lowercase names with a separator, and nothing broader. A part
# longer than 12 characters is not a name part: `sbp_<hex>` and `tok_<hex>` tokens have one.
_DOTTED_SNAKE_CASE = re.compile(r'[a-z][a-z0-9]{0,11}(?:[._][a-z0-9]{1,12})+', re.ASCII)

# `--policy.headers={...}` and `--policy.headers.Modal-Key=...` carry a header set inline;
# `--policy.headers.path=<file>` names the file that holds it.
_HEADERS_FLAG = re.compile(r'--(?:[\w-]+\.)*headers(?:\.(?!path\Z)[^=]+)?', re.ASCII)
# A configuronic reference names code that builds the header set, and `None` sends no headers.
_HEADERS_FROM_CODE = re.compile(r'[@.][A-Za-z_][\w.]*|None', re.ASCII)


class CredentialInRunMetadata(ValueError):
    """A credential, or text that looks like one, in a field that run metadata stores."""


def _shannon_entropy(text: str) -> float:
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def _holds_credential(text: str) -> bool:
    for match in _GENERIC_API_KEY.finditer(text):
        secret = match.group(1)
        if _shannon_entropy(secret) <= _GENERIC_API_KEY_ENTROPY or _DOTTED_SNAKE_CASE.fullmatch(secret):
            continue
        if not _GENERIC_API_KEY_ALLOWED_MATCH.search(match.group(0)):
            return True
    for match in _AUTHORIZATION_VALUE.finditer(text):
        if _shannon_entropy(match.group(1) or match.group(2)) > _AUTHORIZATION_VALUE_ENTROPY:
            return True
    return False


def _inline_headers_flag(arg: str) -> str | None:
    """The flag of ``--<x>.headers=<value>`` when the value is a header set, else None."""
    flag, sep, value = arg.partition('=')
    if sep and _HEADERS_FLAG.fullmatch(flag) and not _HEADERS_FROM_CODE.fullmatch(value):
        return flag
    return None


def check_command(argv: Sequence[str]) -> None:
    """Raise `CredentialInRunMetadata` when an argument of ``argv`` looks like a credential.

    ``--flag value`` is checked as ``--flag=value`` too. The error names the position, never the value.
    """
    for i, arg in enumerate(argv):
        joined = [arg]
        if arg.startswith('--') and '=' not in arg and i + 1 < len(argv) and not argv[i + 1].startswith('-'):
            joined.append(f'{arg}={argv[i + 1]}')
        for candidate in joined:
            flag = _inline_headers_flag(candidate)
            if flag is not None:
                headers = flag.partition('.headers')[0] + '.headers'
                raise CredentialInRunMetadata(
                    f'command: argv[{i}] gives {flag} inline, and run metadata stores the command line; '
                    f'put the header set in a JSON file and read it from there: '
                    f'{headers}=@positronic.cfg.policy.file_headers {headers}.path=<file>'
                )
            if _holds_credential(candidate):
                raise CredentialInRunMetadata(
                    f'command: argv[{i}] looks like a credential, and run metadata stores the command line; '
                    'pass the credential in a file or an environment variable'
                )


def check_text(text: str, field: str) -> None:
    """Raise `CredentialInRunMetadata` when a line of ``text`` looks like a credential.

    The error names ``field`` and the line number, never the value.
    """
    for number, line in enumerate(text.splitlines(), start=1):
        if _holds_credential(line):
            raise CredentialInRunMetadata(
                f'{field}: line {number} looks like a credential, and run metadata stores {field}; '
                'remove it from the uncommitted changes'
            )
