import subprocess
import sys
import os
import multiprocessing  # <-- 1. Impor library ini


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# vvv 2. Bungkus semua logika utama di dalam blok ini vvv
if __name__ == '__main__':
    # 3. Tambahkan baris ini SEGERA di dalam blok
    # Ini akan mencegah loop tak terbatas saat di-bundle menjadi .exe
    multiprocessing.freeze_support()

    main_script_path = resource_path('perhitungan_full.py')

    command = [sys.executable, "-m", "streamlit", "run", main_script_path, "--server.headless=true"]

    subprocess.run(command)