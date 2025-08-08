import inspect
import sys
import difflib

import tiny_llm
import tiny_llm_ref


def export_public_members(module):
    if not module.__name__.startswith("tiny_llm"):
        return []
    print(f"Processing {module.__name__}")

    public_members_info = []
    for name, member in inspect.getmembers(module):
        if not name.startswith("_"):
            if inspect.isfunction(member):
                if member.__module__ != module.__name__:
                    continue
                # only if this is function definition
                # Get the function type annotations
                annotations = member.__annotations__
                path = f"{module.__name__}.{name}"
                public_members_info.append((path, annotations))
            if inspect.isclass(member):
                if member.__module__ != module.__name__:
                    continue
                path = f"{module.__name__}.{name}"
                public_members_info.append((path, member.__annotations__))
                for attr_name, attr_value in member.__dict__.items():
                    if (
                        not attr_name.startswith("_")
                        or attr_name == "__init__"
                        or attr_name == "__call__"
                    ):
                        path = f"{module.__name__}.{name}.{attr_name}"
                        public_members_info.append((path, attr_value.__annotations__))
            if inspect.ismodule(member):
                public_members_info.extend(export_public_members(member))

    return sorted(public_members_info, key=lambda x: x[0])


def stringify_member(members):
    return [
        f"{member[0]}: {str(member[1])}\n".replace("tiny_llm_ref.", "tiny_llm.")
        for member in members
    ]


start_code = stringify_member(export_public_members(tiny_llm))
ref_sol = stringify_member(export_public_members(tiny_llm_ref))

print("--- tiny_llm/apis.txt ---", flush=True)
sys.stdout.writelines(start_code)
print("--- tiny_llm_ref/apis.txt ---", flush=True)
sys.stdout.writelines(ref_sol)

result = list(
    difflib.unified_diff(
        start_code,
        ref_sol,
        fromfile="tiny_llm/apis.txt",
        tofile="tiny_llm_ref/apis.txt",
        n=0,
    )
)

sys.stdout.writelines(result)

if len(result) > 0:
    sys.exit(1)
