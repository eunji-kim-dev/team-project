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
                           ("pyrealsense2","pyrealsense2")):
        loaded=importlib.import_module(module)
        # Jetson apt/소스 빌드 모듈에는 pip 배포 메타데이터가 없을 수 있다.
        try:version=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:version=getattr(loaded,"__version__","system package")
        print(f"{package}: {version} | {loaded.__file__}")
    from d435map.configuration import load_config
    from d435map.semantics import MODEL_SUFFIXES
    folder=Path(load_config()["semantics"]["model_dir"])
    if not folder.is_absolute():folder=ROOT/folder
    models=sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES) if folder.is_dir() else []
    print("Models:",[p.name for p in models])
    if len(models)==1:
        module="tensorrt" if models[0].suffix.lower()==".engine" else "torch"
        loaded=importlib.import_module(module)
        print(f"{module}: {loaded.__version__} | {loaded.__file__}")
        if module=="tensorrt":
            from d435map.tensorrt_backend import cuda_runtime,checked
            cuda=cuda_runtime()
            print("CUDA runtime:",checked(cuda.cudaRuntimeGetVersion()),"| devices:",checked(cuda.cudaGetDeviceCount()))
    print("Base environment OK. Use --check-model to test the selected model; camera is not checked here.")


if __name__=="__main__":
    # [변경] 동봉 runtime으로 재실행하지 않는다. 선택한 Python을 그대로 사용한다.
    # [engine 추가] Jetson 시스템 Python/TensorRT 바인딩을 사용할 수 있도록 3.10+ 허용.
    if sys.version_info<(3,10) or sys.maxsize<=2**32:
        raise SystemExit("Python 3.10+ 64-bit is required. Use the Python with your Jetson TensorRT bindings.")
    try:
        from d435map.performance import configure_native_threads
        configure_native_threads(ROOT,sys.argv[1:])
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
        if sys.platform=="win32":
            print(f'"{sys.executable}" -m pip install -r "{ROOT / "requirements.txt"}"',file=sys.stderr)
        else:
            print("Jetson: use its TensorRT/CUDA/RealSense Python environment; see requirements-jetson.txt.",file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"v3.5 error: {exc}",file=sys.stderr)
        if "Device or resource busy" in str(exc):
            print("Camera is in use. Close its other window/debug session, then retry.",file=sys.stderr)
        if "Mutex_TotalFailure" in str(exc) or "CreateNamedMutex" in str(exc):
            print("Camera access blocked by this environment. Run from normal Windows or VS Code.",file=sys.stderr)
        raise SystemExit(1)
