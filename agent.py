from pathlib import Path
import importlib.util
import sys


def main():
    module_path = Path(__file__).resolve().parent / "agent" / "main.py"
    spec = importlib.util.spec_from_file_location("stage3_agent_main", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
