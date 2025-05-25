def dump(data, default_flow_style=False, sort_keys=False):
    def needs_quotes(obj: str) -> bool:
        special = any(ch in obj for ch in "@:*{}[]")
        return special or obj != obj.strip() or obj == ""

    def format_scalar(val):
        if isinstance(val, str):
            return f"'{val}'" if needs_quotes(val) else val
        return str(val)

    def dump_obj(obj, indent=0):
        lines = []
        prefix = "  " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"'{k}'" if needs_quotes(k) else k
                if isinstance(v, dict):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(dump_obj(v, indent + 1))
                elif isinstance(v, list):
                    lines.append(f"{prefix}{key}:")
                    for item in v:
                        lines.append(f"{'  '*indent}- {format_scalar(item)}")
                else:
                    lines.append(f"{prefix}{key}: {format_scalar(v)}")
        else:
            lines.append(prefix + format_scalar(obj))
        return lines

    return "\n".join(dump_obj(data)) + "\n"

