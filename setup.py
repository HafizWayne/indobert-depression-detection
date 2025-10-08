import sys
from cx_Freeze import setup, Executable

# --- OPSI UNTUK MENANGANI PUSTAKA BESAR & MASALAH UMUM ---
# build_exe_options adalah tempat kita mengontrol bagaimana cx_Freeze bekerja.
build_exe_options = {
    # Menyertakan semua file dan folder aset Anda.
    # Format: "path/sumber/file": "path/tujuan/di/hasil/build"
    "include_files": [
        "perhitungan_full.py",
        "finetuned_indobert_negation_combine/",
        "token_negasi.txt",
        "DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique.ttf",
        "DejaVuSansMono.ttf",
        "DejaVuSansMono-Bold.ttf",
        "splash.png"  # Jika Anda masih menggunakan splash screen
    ],

    # Memberitahu cx_Freeze untuk menyertakan pustaka yang mungkin terlewat.
    "packages": [
        "streamlit",
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "seaborn",
        "scipy",
        "fpdf"
    ],

    # Mengecualikan pustaka yang tidak kita perlukan.
    "excludes": [
        "tkinter",  # Streamlit tidak butuh Tkinter
        "pytest",
        "unittest"
    ],

    # cx_Freeze terkadang butuh path eksplisit ke file .dll penting
    # Biarkan kosong dulu, isi jika ada error "DLL not found".
    "bin_includes": []
}

# --- KONFIGURASI APLIKASI UTAMA ---
# Ini memberitahu cx_Freeze tentang file .exe yang akan dibuat.
# 'base' diatur ke "Win32GUI" agar tidak ada jendela konsol hitam yang muncul.
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="Aplikasi Analisis",
    version="1.0",
    description="Aplikasi Analisis Teks dengan Streamlit dan Transformers",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "run_app.py",  # File utama yang dijalankan
            base=base,
            target_name="AplikasiAnalisis.exe"  # Nama file .exe final
        )
    ]
)