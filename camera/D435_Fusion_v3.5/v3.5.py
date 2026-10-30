"""D435 Fusion v3.5: 현재 PC/VS Code에서 선택한 Python으로 실행."""
from pathlib import Path
import importlib
import importlib.metadata
import sys

ROOT=Path(__file__).resolve().parent


def check_environment():
    # [추가] 카메라를 점유하지 않고 현재 인터프리터의 필수 라이브러리를 확인한다.
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]} | 64-bit")
    for module,package in (("numpy","numpy"),("cv2","opencv-python"),
                           ("pyrealsense2","pyrealsense2"),("torch","torch")):
        loaded=importlib.import_module(module)
        print(f"{package}: {importlib.metadata.version(package)} | {loaded.__file__}")
    print("Environment OK. Camera connection is checked when live mode starts.")


if __name__=="__main__":
    # [변경] 동봉 runtime으로 재실행하지 않는다. 선택한 Python을 그대로 사용한다.
    if sys.version_info<(3,12) or sys.maxsize<=2**32:
        raise SystemExit("Python 3.12+ 64-bit is required. Tested with Python 3.14.7 on Windows.")
    try:
        if sys.argv[1:]==["--check-env"]:
            check_environment()
        else:
            from d435map.application import main
            print(f"Python: {sys.executable}")
            main()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except ModuleNotFoundError as exc:
        print(f"Missing library: {exc.name}",file=sys.stderr)
        print("Install dependencies into THIS Python:",file=sys.stderr)
        print(f'"{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"',file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"v3.5 error: {exc}",file=sys.stderr)
        if "Device or resource busy" in str(exc):
            print("Camera is in use. Close its other window/debug session, then retry.",file=sys.stderr)
        if "Mutex_TotalFailure" in str(exc) or "CreateNamedMutex" in str(exc):
            print("Camera access blocked by this environment. Run from normal Windows or VS Code.",file=sys.stderr)
        raise SystemExit(1)
